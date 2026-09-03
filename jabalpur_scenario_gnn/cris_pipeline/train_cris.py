# -*- coding: utf-8 -*-
"""
cris_pipeline/train_cris.py
===========================
Trains CorridorNextEventGNN on the official CRIS corridor data.

Design notes:

  * PERSISTENCE IS REPORTED EVERY EPOCH. The previous model's headline
    val_mae looked good while it beat persistence by only 0.7-3.0 min, which
    was not visible during training. Here the baseline is computed on the same
    events, so "are we actually better than doing nothing" is answerable at a
    glance.

  * TBPTT_K covers network memory only. Journey propagation is handled
    explicitly by the history encoder, which back-propagates over the full
    run (<=20 halts) every step, so the GRU memories only have to carry
    station and section state and a short window suffices. This is also the
    dominant training memory cost — see the constant below.

  * Loss is HUBER in LINEAR minutes on the delta vs current delay, with spike
    upweighting. It was pinball (3 quantiles); pinball at q=0.5 is L1, which
    fits the median, and per-leg delay change is right-skewed — so the median
    ran below the mean and the autoregressive rollout flattened out (-20 min
    bias at 2-4 h). Huber fits the mean over the normal range while still
    bounding the influence of a 400-minute outlier.

    python -m cris_pipeline.train_cris --epochs 40
"""

from __future__ import annotations

import argparse
import gc
import os
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cris_pipeline.config import TOPOLOGY_PATH, DATASET_DIR, OUT_DIR   # noqa: E402
from cris_pipeline import build_dataset as BD
from cris_pipeline.dataset_cris import CrisDataset                     # noqa: E402
from cris_pipeline.model_cris import (                            # noqa: E402
    CorridorNextEventGNN, regression_loss, load_weights,
)

# Truncation window for the STATION/SECTION memory GRUs only. Journey
# propagation does not depend on it — the history encoder re-encodes the whole
# run (<=20 halts) with full gradient flow at every tick — so this can stay
# short. It is the dominant memory cost during training: K live forward graphs
# are held before each backward, and K=24 exhausted a 17 GB machine at epoch 3.
TBPTT_K = 8
SPIKE_THRESHOLD = 30.0
SPIKE_WEIGHT = 4.0
# Recovery is 5.2x RARER than a spike in the training split (563 events vs
# 2,916, i.e. 2.01% against 10.82%) and was weighted LOWER than one, so spikes
# received roughly 7x the effective gradient.
#
# That imbalance is not justified by the data. Fitting a gradient-boosted tree
# on the same features, grouped by journey, gives AUC 0.966 for detecting a
# fall against 0.864 for a rise -- recovery is the MORE separable of the two.
# We were leaving the easier signal on the floor.
#
# 10.0 offsets the 5.2x rarity without inverting the priority: a spike still
# carries more weight per unit of imbalance. Spike recall must be watched, not
# assumed -- this is a trade, and if it collapses the number goes back down.
#
# MEASURED RESULT (v4 vs v3, identical data, code, days and train types):
#   recovery journey gain  +13.31 -> +16.30  (n=22)
#   spike recall            73.5% -> 77.8%,  precision 43.3% -> 45.9%
#   overall rollout MAE      7.14 ->  7.38   (inside the ~+/-0.3 noise band)
# Spikes did NOT collapse; they improved. The cost is a small long-horizon
# MAE regression. Do not read more into the recovery figure than n=22 allows.
#
# REVERTED to 3.0 after v4. The trade did not pay: it bought +3.0 recovery
# journey gain on 22 events and cost +1.46 MAE at the 2-4 h horizon, which is
# the horizon this system exists to win. Mechanism: the heavier weight pulls
# every estimate down (bias at 2-4 h -4.08 -> -5.61), so on spike journeys v4
# predicts LESS rise than v3 at exactly the deep hops where reality rises most.
#
# Recovery is a MISSING-FEATURE problem, not a loss-weighting one. The
# timetable's own padding -- `traffic_allowance_seconds` and
# `engineering_allowance_seconds` -- predicts recovery cleanly (mean delay
# change +1.24 min at zero allowance, -17.47 min at 10+ min of allowance,
# monotonic, and it holds inside every incoming-delay band) and was never
# given to the model. Fix the inputs, not the weights.
RECOVERY_WEIGHT = 3.0
DEP_LOSS_WEIGHT = 0.5

# Predict the next stop's DWELL EXTENSION (dep_delay - arr_delay) rather than
# an absolute departure delta. Checkpoints record which was used, because the
# rollout has to compose the two heads differently for each.
DEP_AS_EXTENSION = True
HOLD_WEIGHT = 4.0

# `remaining_dwell` — minutes until this train leaves the platform it is on.
# Half weight like the departure head: it does not appear in the reported MAE,
# it decides how long a train blocks a platform inside the forecast. DWELL_WEIGHT
# lifts the stands that actually block: over 15 minutes still to wait is 10% of
# standing observations and is where a queue starts forming behind.
DWELL_LOSS_WEIGHT = 0.5
DWELL_WEIGHT = 4.0

# Hurdle head. `spike_logit` answers "is |delta| > SPIKE_THRESHOLD"; `spike_mag`
# answers "how big, signed" on those events only. See model_cris for why the
# single regression head cannot do both jobs.
#
# POS_WEIGHT offsets the class imbalance: large changes are ~1.4% of supervised
# events, so an unweighted classifier reaches 98.6% accuracy by always saying
# "no" and learns nothing. 20.0 is deliberately below the full 1/0.014 = 71 --
# fully balancing would push the classifier to fire constantly, and the
# threshold is a dial we tune AFTER training anyway, so the classifier only has
# to RANK well, not be calibrated at 0.5.
HURDLE_POS_WEIGHT = 20.0
HURDLE_CLS_WEIGHT = 1.0
HURDLE_MAG_WEIGHT = 1.0


def _weights(target: torch.Tensor) -> torch.Tensor:
    """Upweight the events that matter operationally and are rarest."""
    spike = (target > SPIKE_THRESHOLD).float()
    recov = (target < -SPIKE_THRESHOLD).float()
    return 1.0 + (SPIKE_WEIGHT - 1.0) * spike + (RECOVERY_WEIGHT - 1.0) * recov


def _group_by_day(snaps):
    days = defaultdict(list)
    for s in snaps:
        days[s.day_index].append(s)
    return [days[k] for k in sorted(days)]


def _apply_scheduled_sampling(d, sim: dict, p: float) -> None:
    """Replace some trains' OBSERVED delay with the model's own running
    prediction, in place, before the forward pass.

    Training feeds the model a true current delay at every step; the rollout
    feeds it its own output. The two distributions differ — predictions are
    smoother and less extreme — and the model never learns to recover from its
    own error. Measured consequence: as val_mae improved 2.87 -> 2.38 between
    epochs 2 and 9, rollout MAE got WORSE (7.62 -> 8.16) and the 2-4 h bucket
    degraded 11.73 -> 18.46. The objective we optimise and the one we ship were
    moving in opposite directions.

    Only the delay-derived node dims are substituted (dim 0 signed-log delay,
    dim 1 delay velocity) plus `current_delay`, which is what the target is
    measured against. The journey history is NOT substituted — it is rebuilt
    from the instance rather than carried on the node — so this is a partial
    treatment, and the residual is worth re-measuring rather than assuming gone.
    """
    if p <= 0.0 or not sim:
        return
    iids = getattr(d["train"], "iid", None)
    if not iids:
        return
    cur = d["train"].current_delay
    take = torch.rand(len(iids)) < p
    for i, iid in enumerate(iids):
        if not bool(take[i]) or iid not in sim:
            continue
        new = float(sim[iid])
        old = float(cur[i])
        cur[i] = new
        d["train"].x[i, 0] = math.copysign(
            math.log1p(abs(new)), new) / math.log1p(240.0)
        d["train"].x[i, 1] = math.tanh((new - old) / 5.0)


_L240 = math.log1p(240.0)


def _unslog(x: float) -> float:
    """Inverse of dataset_cris._slog — recover minutes from the encoded dim."""
    return math.copysign(math.expm1(abs(x) * _L240), x)


def _apply_history_sampling(d, sim_hist: dict, p: float) -> None:
    """Substitute the model's OWN earlier predictions into the journey history.

    `_apply_scheduled_sampling` above only replaces the delay-derived NODE dims.
    The history channel stayed teacher-forced, so the model has never once
    practised reading a history that is drifting because of its own errors --
    which is exactly the state it is in during a runaway.

    Measured motivation: on the 96-journey ship set the headline mean is owned
    by 2-3 journeys where the forecast climbs monotonically and never returns
    (train 22351 went +122 -> +192 across fifteen stops while reality sat at
    +95). Post-hoc guards do not fix it -- damping toward persistence at
    lambda=0.97 left the mean unchanged at 7.66 and collapsed spike recall from
    82.4% to 63.5%, and a drift cap cannot fire because +115 of drift is inside
    the 99.5th percentile of real drift at that horizon. Exposure bias has to
    be trained out, not clamped out.

    Rows are patched oldest-first so each row's per-leg delta is computed
    against the delay actually carried into it, substituted or not.
    """
    if p <= 0.0 or not sim_hist:
        return
    iids = getattr(d["train"], "iid", None)
    sidx = getattr(d["train"], "stop_idx", None)
    if not iids or sidx is None:
        return
    hist = d["train"].history                       # [N, MAX_HISTORY, HIST_DIM]
    max_hist = hist.shape[1]
    take = torch.rand(len(iids)) < p
    for i, iid in enumerate(iids):
        pred = sim_hist.get(iid)
        if not pred or not bool(take[i]):
            continue
        # Row j describes stop `offset + j` once the window has slid.
        offset = max(0, int(sidx[i]) + 1 - max_hist)
        prev = None
        for j in range(max_hist):
            row = hist[i, j]
            if float(row[7]) == 0.0 and float(row[0]) == 0.0:
                break                               # zero padding: past the end
            k = offset + j
            if k in pred:
                dly = float(pred[k])
                row[0] = math.copysign(math.log1p(abs(dly)), dly) / _L240
            else:
                dly = _unslog(float(row[0]))
            if prev is not None:
                row[1] = math.tanh((dly - prev) / 10.0)
            prev = dly


def run_epoch(model, day_iter, optimiser=None, device="cpu", on_day=None,
              ss_prob: float = 0.0, hist_ss_prob: float | None = None):
    """`day_iter` yields one day's snapshots at a time (see CrisDataset.iter_days)
    so only a single day is resident in memory.

    `on_day(index, mean_loss, n_events)` is called after each day so the caller
    can show progress. At 3-min ticks an epoch takes ~8 minutes, and printing
    only at the epoch boundary makes a healthy run indistinguishable from a
    hung one for the whole of that window.
    """
    train_mode = optimiser is not None
    model.train(train_mode)
    # History-channel sampling defaults to the node rate. Set it to 0 to
    # reproduce v3's PARTIAL treatment, where only the delay dims were
    # substituted and the journey history stayed teacher-forced.
    if hist_ss_prob is None:
        hist_ss_prob = ss_prob

    tot_loss = n_loss = 0.0
    abs_err = pers_err = n_ev = 0.0
    sp_hit = sp_tot = 0
    rc_hit = rc_tot = 0
    sp_err = 0.0
    h_tp = h_fp = h_fn = 0
    h_mag_err = 0.0

    for day_i, day in enumerate(day_iter):
        model.reset_memory()
        window = []
        sim: dict = {}          # instance_id -> our own running prediction
        sim_hist: dict = {}     # instance_id -> {stop_idx: our prediction}

        for snap in day:
            d = snap.data
            if d["train"].x.numel() == 0:
                continue

            if train_mode and ss_prob > 0.0:
                _apply_scheduled_sampling(d, sim, ss_prob)
            if train_mode and hist_ss_prob > 0.0:
                _apply_history_sampling(d, sim_hist, hist_ss_prob)

            out = model(d, update_memory=True)

            mask = d["train"].target_mask[:, 0]
            tgt = d["train"].target_delta[:, 0]

            if DEP_AS_EXTENSION:
                # DWELL EXTENSION: how much longer than booked the train stands
                # at the next stop. Delay on this corridor is CREATED by holds
                # (77% of large gains), and in rollout the forecast world never
                # holds a train — measured time_at_stop 0.49x and
                # delay_velocity 0.09x of reality — so the delay-generating
                # mechanism cannot recur after T0 and forecasts flatten out.
                # This head is the only channel that can put a train back on a
                # platform, so give it a target that means exactly that.
                dep_mask = d["train"].y_next_dep_mask * mask
                dep_tgt = d["train"].y_next_dep - d["train"].target_abs[:, 0]
            else:
                dep_mask = d["train"].y_next_dep_mask
                dep_tgt = d["train"].y_next_dep - d["train"].current_delay

            w = _weights(tgt)
            # Extensions are rare and are the whole point: >5 min at only 6.9%
            # of stops (p50 = 0.0, p90 = +2, p99 = +27).
            dw = (1.0 + (HOLD_WEIGHT - 1.0) * (dep_tgt.abs() > 5.0).float()
                  if DEP_AS_EXTENSION else None)
            loss = (regression_loss(out["next_arr"], tgt, mask, w)
                    + DEP_LOSS_WEIGHT
                    * regression_loss(out["next_dep"], dep_tgt, dep_mask, dw))

            # HOW MUCH LONGER WILL THIS TRAIN STAND HERE?
            #
            # Supervised on every standing train node, not only the ones whose
            # next stop is a scored target -- 70,353 examples against the 295
            # large delay events the spike work has been starved by. A spike IS
            # a long stand, so this is the same physical event asked as a
            # dense duration question instead of a rare-event regression.
            #
            # Weighted like the other heads at the point where a stand stops
            # being routine: over 15 minutes still to wait is where platform
            # blocking begins, and those are 10% of standing observations.
            if "remaining_dwell" in out:
                rd_tgt = d["train"].y_remaining_dwell
                rd_mask = d["train"].y_remaining_dwell_mask
                rd_w = 1.0 + (DWELL_WEIGHT - 1.0) * (rd_tgt > 15.0).float()
                loss = loss + DWELL_LOSS_WEIGHT * regression_loss(
                    out["remaining_dwell"], rd_tgt, rd_mask, rd_w)

            # ── HURDLE LOSSES ────────────────────────────────────────────────
            #
            # Two extra terms, both on the SAME supervised events as above.
            #
            # 1. Is this a large change at all? Trained on every event, as a
            #    yes/no question. Rare-event classification is a far kinder
            #    problem than rare-event regression -- a gradient-boosted tree
            #    on these features already reaches AUC 0.864 for rises and
            #    0.966 for falls, so the signal is there to be had.
            #    pos_weight offsets the 1.4% base rate; without it the
            #    classifier can score 98.6% by always answering "no".
            #
            # 2. How large, signed, GIVEN that it is large. Masked to those
            #    events only -- this is the whole point. Today's single head
            #    sees 24,000 near-zero targets alongside 231 real ones and is
            #    dragged to the quiet answer. This head never sees a quiet stop
            #    and so can learn what a big event actually looks like
            #    (mean around +80) instead of the unconditional mean (+2).
            big = (tgt.abs() > SPIKE_THRESHOLD).float()
            sup = mask > 0.5
            if "spike_logit" in out and sup.any():
                pw = torch.tensor(HURDLE_POS_WEIGHT, device=tgt.device)
                cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    out["spike_logit"][sup], big[sup], pos_weight=pw)
                mag_mask = mask * big
                loss = (loss
                        + HURDLE_CLS_WEIGHT * cls_loss
                        + HURDLE_MAG_WEIGHT
                        * regression_loss(out["spike_mag"], tgt, mag_mask, None))

            if train_mode:
                window.append(loss)
                if len(window) >= TBPTT_K:
                    (sum(window) / len(window)).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimiser.step()
                    optimiser.zero_grad(set_to_none=True)
                    window.clear()
                    model.detach_memory()
            else:
                # Metrics on the regression head, in absolute minutes.
                with torch.no_grad():
                    sel = mask > 0.5
                    if sel.any():
                        pred = out["next_arr"][sel]
                        t = tgt[sel]
                        abs_err += float((pred - t).abs().sum())
                        # Persistence = "delay does not change" => delta 0
                        pers_err += float(t.abs().sum())
                        n_ev += int(sel.sum())

                        is_sp = t > SPIKE_THRESHOLD
                        if is_sp.any():
                            sp_tot += int(is_sp.sum())
                            # Same bar as eval_cris.py. This used to count a
                            # hit at SPIKE_THRESHOLD*0.5, so the training log
                            # reported 97% where the honest test gave 81% —
                            # the log must not grade itself more kindly than
                            # the evaluation does.
                            sp_hit += int((pred[is_sp] > SPIKE_THRESHOLD).sum())
                            sp_err += float((pred[is_sp] - t[is_sp]).abs().sum())

                        # Recovery, tracked on the same bar as spikes. It was
                        # invisible in this log while spike_recall sat at 93%
                        # and the ROLLOUT recovery recall was 21.4% -- a gap
                        # nobody could see per-epoch. Measured on the training
                        # split, recovery is the MORE detectable of the two
                        # (a gradient-boosted tree on the same features gets
                        # AUC 0.966 for falls vs 0.864 for rises), so a low
                        # number here is lost signal, not an impossible task.
                        # Hurdle diagnostics. The classifier is the half
                        # we expect to work (detection AUC 0.864/0.966); the
                        # magnitude head is the half at risk, because it sees
                        # only ~231 events. Watch both per epoch rather than
                        # discovering at the end that one never trained.
                        if "spike_logit" in out:
                            pb = torch.sigmoid(out["spike_logit"][sel])
                            ab = t.abs() > SPIKE_THRESHOLD
                            pos = pb > 0.5
                            h_tp += int((pos & ab).sum())
                            h_fp += int((pos & ~ab).sum())
                            h_fn += int((~pos & ab).sum())
                            if ab.any():
                                h_mag_err += float(
                                    (out["spike_mag"][sel][ab] - t[ab]).abs().sum())

                        is_rc = t < -SPIKE_THRESHOLD
                        if is_rc.any():
                            rc_tot += int(is_rc.sum())
                            rc_hit += int((pred[is_rc] < -SPIKE_THRESHOLD).sum())

            if train_mode and ss_prob > 0.0:
                with torch.no_grad():
                    nxt = (d["train"].current_delay
                           + out["next_arr"].detach())
                sidx = getattr(d["train"], "stop_idx", None)
                for i, iid in enumerate(getattr(d["train"], "iid", [])):
                    sim[iid] = float(nxt[i])
                    if sidx is not None:
                        # the prediction is for the NEXT stop, which is
                        # the row the history will carry from then on
                        sim_hist.setdefault(iid, {})[int(sidx[i]) + 1] = float(nxt[i])

            tot_loss += float(loss.detach())
            n_loss += 1

        if train_mode and window:
            (sum(window) / len(window)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            window.clear()
            model.detach_memory()
        del day
        gc.collect()
        if on_day:
            on_day(day_i, tot_loss / max(n_loss, 1), int(n_ev))

    return {
        "loss": tot_loss / max(n_loss, 1),
        "mae": abs_err / max(n_ev, 1),
        "persistence_mae": pers_err / max(n_ev, 1),
        "n_events": int(n_ev),
        "spike_recall": sp_hit / max(sp_tot, 1),
        "spike_mae": sp_err / max(sp_tot, 1),
        "n_spikes": sp_tot,
        "recovery_recall": rc_hit / max(rc_tot, 1),
        "n_recoveries": rc_tot,
        "hurdle_recall": h_tp / max(h_tp + h_fn, 1),
        "hurdle_prec": h_tp / max(h_tp + h_fp, 1),
        "hurdle_mag_mae": h_mag_err / max(h_tp + h_fn, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--checkpoint", default=str(OUT_DIR / "corridor_nextevent.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--history-to-gnn", action="store_true",
                    help="also feed journey history into the node input "
                         "(ablation; default is head-only)")
    ap.add_argument("--history-sampling", type=float, default=None,
                    help="scheduled-sampling rate for the JOURNEY HISTORY "
                         "channel. Defaults to --scheduled-sampling. Set 0 to "
                         "reproduce v3's partial treatment (delay dims only), "
                         "which isolates what full history sampling is worth.")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch seed. Recorded in the checkpoint so two runs "
                         "can be told apart from a genuine config change.")
    ap.add_argument("--hurdle", action="store_true",
                    help="add the spike classifier/magnitude heads. OFF by "
                         "default: measured harmful in v7 even when the "
                         "heads are not used at inference.")
    ap.add_argument("--service-dropout", type=float, default=0.0,
                    help="probability of hiding a train's service identity "
                         "during training. Only 48 of 301 services run "
                         "near-daily and 42 embedding rows are never "
                         "trained at all, so a quarter of the test set "
                         "presents a condition training never contains.")
    ap.add_argument("--scheduled-sampling", type=float, default=0.0,
                    help="peak probability of feeding a train its OWN earlier "
                         "prediction instead of the observed delay, annealed "
                         "linearly from 0 over the run. 0 disables it.")
    ap.add_argument("--save-every", type=int, default=3,
                    help="also checkpoint every N epochs. val_mae is NOT a "
                         "reliable guide to forecast quality (epoch 2 rolled "
                         "out at MAE 7.62 vs epoch 9 at 8.16 while val_mae "
                         "improved 2.87 -> 2.38), so keep intermediate epochs "
                         "and score them by rollout afterwards.")
    ap.add_argument("--resume", default=None,
                    help="'auto' resumes <checkpoint>_last.pt if it exists and "
                         "starts fresh if not — safe to re-run the same command "
                         "after a sleep or a kill. A path resumes that file. "
                         "Restores optimiser, scheduler, best and history, not "
                         "just weights.")
    ap.add_argument("--log", default=None,
                    help="append each epoch line to this file, flushed "
                         "immediately (do NOT pipe through `tee` — its block "
                         "buffering leaves NUL-filled gaps in the file while "
                         "the run is in progress)")
    args = ap.parse_args()

    # ── Single-writer lock ────────────────────────────────────────────────────
    # Two trainers writing one checkpoint silently corrupts the run: each
    # tracks its own `best`, so they overwrite each other and the saved epoch
    # can go BACKWARDS (observed: epoch 6 replaced by epoch 4). Concurrent
    # torch.save to the same path can also truncate the file.
    lock = Path(args.checkpoint).with_suffix(".lock")
    if lock.exists():
        try:
            other = int(lock.read_text().strip())
        except Exception:
            other = -1
        alive = False
        if other > 0:
            try:
                import subprocess
                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {other}", "/NH"],
                    capture_output=True, text=True, timeout=10).stdout
                alive = str(other) in out
            except Exception:
                alive = True          # cannot verify -> assume alive, be safe
        if alive:
            sys.exit(f"ERROR: another trainer (PID {other}) is already writing "
                     f"{args.checkpoint}\n"
                     f"Stop it first, or pass a different --checkpoint. "
                     f"If you are sure it is dead, delete {lock}")
        print(f"  (clearing stale lock from dead PID {other})")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()), encoding="utf-8")

    import atexit
    atexit.register(lambda: lock.unlink(missing_ok=True))

    log_fh = open(args.log, "a", encoding="utf-8", buffering=1) if args.log else None

    def emit(msg: str) -> None:
        print(msg, flush=True)
        if log_fh:
            log_fh.write(msg + "\n")
            log_fh.flush()

    # The dataset records which delay convention it was built under. Read it
    # and apply it to build_dataset, so training and the rollout can never
    # silently disagree about what a standing train's delay means.
    _man = Path(args.dataset) / "meta.json"
    HONEST_STANDING = False
    if _man.exists():
        HONEST_STANDING = bool(json.loads(_man.read_text(encoding="utf-8"))
                               .get("honest_standing_delay", False))
    BD.HONEST_STANDING_DELAY = HONEST_STANDING
    print(f"  honest standing delay: {HONEST_STANDING}")

    torch.manual_seed(42)
    print(f"Loading {args.dataset} ...")
    ds = CrisDataset(args.dataset, args.topology, max_days=args.max_days,
                     lazy=True)
    print(f"{ds.n_days('train')} train days | {ds.n_days('validation')} val days")

    model = CorridorNextEventGNN(
        station_in_dim=ds.station_feat_dim,
        train_in_dim=ds.train_feat_dim,
        section_in_dim=ds.section_feat_dim,
        hist_in_dim=ds.hist_feat_dim,
        num_services=ds.num_services,
        num_stations=ds.num_stations,
        num_sections=ds.num_sections,
        hidden=args.hidden,
        history_to_gnn=args.history_to_gnn,
        service_dropout=args.service_dropout,
        hurdle=args.hurdle,
    )
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"history path: {'GNN + head' if args.history_to_gnn else 'head only'}")

    ck_config = {
        "station_in_dim": ds.station_feat_dim,
        "train_in_dim": ds.train_feat_dim,
        "section_in_dim": ds.section_feat_dim,
        "hist_in_dim": ds.hist_feat_dim,
        "num_services": ds.num_services,
        "num_stations": ds.num_stations,
        "num_sections": ds.num_sections,
        "hidden": args.hidden,
        "history_to_gnn": args.history_to_gnn,
        "service_dropout": args.service_dropout,
        "hurdle": args.hurdle,
    }
    torch.manual_seed(args.seed)
    hist_ss = (args.scheduled_sampling if args.history_sampling is None
               else args.history_sampling)
    print(f"scheduled sampling: node {args.scheduled_sampling} | "
          f"history {hist_ss} | seed {args.seed}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    # ── Resume ────────────────────────────────────────────────────────────────
    # `_last.pt` is written after EVERY epoch and carries optimiser, scheduler,
    # best-so-far and history as well as the weights. Restoring only the model
    # would restart Adam's moments and rewind the cosine schedule to its
    # starting learning rate, which is a different (and worse) run wearing the
    # same name. `--resume auto` picks it up if it exists and starts fresh if
    # not, so the same command is safe to re-run after a sleep or a kill.
    last_path = Path(args.checkpoint).with_name(
        Path(args.checkpoint).stem + "_last.pt")
    start_ep = 1
    best = float("inf")
    history: list = []

    resume_from = None
    if args.resume == "auto":
        resume_from = last_path if last_path.exists() else None
        if resume_from is None:
            print("  --resume auto: no _last.pt found, starting from scratch")
    elif args.resume:
        resume_from = Path(args.resume)

    if resume_from is not None:
        ck = torch.load(resume_from, map_location="cpu", weights_only=False)
        load_weights(model, ck["model"])
        if "optimiser" in ck:
            opt.load_state_dict(ck["optimiser"])
        if "scheduler" in ck:
            sched.load_state_dict(ck["scheduler"])
        start_ep = int(ck.get("epoch", 0)) + 1
        best = float(ck.get("best", float("inf")))
        history = list(ck.get("history", []))
        # The cosine schedule's T_max is args.epochs, so resuming with a
        # different --epochs makes the restored scheduler state meaningless and
        # the learning rate can jump back UP mid-run (observed: 1.57e-4 at
        # epoch 3 then 3.00e-4 at epoch 4). Resume with the same --epochs.
        prev_total = int(ck.get("total_epochs", args.epochs))
        if prev_total != args.epochs:
            print(f"  !! WARNING: this run was started with --epochs "
                  f"{prev_total} but you passed {args.epochs}. The cosine "
                  f"schedule will be wrong. Re-run with --epochs {prev_total}.")
        print(f"resumed {resume_from.name} -> starting at epoch {start_ep} "
              f"(best so far {best:.3f}, "
              f"{'optimiser+scheduler restored' if 'optimiser' in ck else 'WEIGHTS ONLY'})")
    # --epochs is the TARGET TOTAL, not "this many more". Resuming at 12 with
    # --epochs 30 runs 12..30, not 12..42 — otherwise every resume silently
    # extends the run and desynchronises the cosine schedule from it.
    if start_ep > args.epochs:
        print(f"already at epoch {start_ep - 1} of {args.epochs} — nothing to do")
        return
    for ep in range(start_ep, args.epochs + 1):
        t0 = time.time()
        n_tr_days = ds.n_days("train")

        def _prog(i, loss, _n):
            emit(f"    ep{ep:<3} training day {i + 1:2d}/{n_tr_days}  "
                 f"loss={loss:.3f}  {time.time() - t0:.0f}s")

        # Anneal from 0: the model must first learn the task on true inputs,
        # then learn to survive its own errors. Starting high just teaches it
        # to fit noise.
        ss_p = (args.scheduled_sampling
                * min(1.0, max(0.0, (ep - 1) / max(args.epochs - 1, 1))))
        tr = run_epoch(model, ds.iter_days("train"), opt, on_day=_prog,
                       ss_prob=ss_p, hist_ss_prob=hist_ss * (ss_p / max(args.scheduled_sampling, 1e-9)))
        with torch.no_grad():
            va = run_epoch(model, ds.iter_days("validation"), None)
        sched.step()

        gain = va["persistence_mae"] - va["mae"]
        line = (f"Epoch {ep:3d}  train_loss={tr['loss']:.3f}  "
                f"val_mae={va['mae']:.2f}  persist={va['persistence_mae']:.2f}  "
                f"gain={gain:+.2f}  spike_recall={va['spike_recall']:.0%} "
                f"({va['n_spikes']})  spike_mae={va['spike_mae']:.1f}  "
                f"recov_recall={va['recovery_recall']:.0%} "
                f"({va['n_recoveries']})  "
                f"hurdle R={va['hurdle_recall']:.0%}/"
                f"P={va['hurdle_prec']:.0%} "
                f"mag={va['hurdle_mag_mae']:.0f}  "
                f"lr={sched.get_last_lr()[0]:.2e}  ss={ss_p:.2f}  "
                f"t={time.time() - t0:.0f}s")
        emit(line)
        history.append({"epoch": ep, **{k: va[k] for k in va},
                        "train_loss": tr["loss"], "gain_vs_persistence": gain})

        def _payload(extra: dict | None = None) -> dict:
            d = {"model": model.state_dict(), "epoch": ep,
                 "val_mae": va["mae"], "persistence_mae": va["persistence_mae"],
                 "model_class": "CorridorNextEventGNN",
                 # How the rollout must compose the arrival and departure heads.
                 "dep_mode": "extension" if DEP_AS_EXTENSION else "absolute",
                 # Which delay convention this model was trained under, so
                 # the rollout scores it under the same one. Read from the
                 # dataset manifest, never assumed.
                 "honest_standing_delay": HONEST_STANDING,
                 "config": ck_config}
            d.update(extra or {})
            return d

        if va["mae"] < best:
            best = va["mae"]
            torch.save(_payload(), args.checkpoint)
            emit(f"  *** saved best (val_mae={best:.2f}) ***")

        # Periodic snapshot. val_mae is NOT a reliable proxy for forecast
        # quality — epoch 2 rolled out at MAE 7.62 and epoch 9 at 8.16 while
        # val_mae improved 2.87 -> 2.38 — so keep intermediate epochs and score
        # them by rollout afterwards instead of trusting best-by-val.
        if args.save_every and ep % args.save_every == 0:
            snap_path = Path(args.checkpoint).with_name(
                Path(args.checkpoint).stem + f"_ep{ep:02d}.pt")
            torch.save(_payload(), snap_path)
            emit(f"      (periodic checkpoint -> {snap_path.name})")

        # Full resume state, every epoch. Written last so a kill mid-write
        # cannot corrupt the best/periodic checkpoints.
        torch.save(_payload({"optimiser": opt.state_dict(),
                             "scheduler": sched.state_dict(),
                             "best": best, "history": history,
                             "total_epochs": args.epochs}), last_path)

    Path(args.checkpoint).with_suffix(".json").write_text(
        json.dumps(history, indent=1), encoding="utf-8")
    print(f"\nDone. best val_mae={best:.2f}")


if __name__ == "__main__":
    main()
