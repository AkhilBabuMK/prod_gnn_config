# -*- coding: utf-8 -*-
"""
cris_pipeline/rollout_cris.py
=============================
TRUE autoregressive rollout — the honest multi-hop metric.

`eval_cris.py` section 2 is NOT this. It scores the one-step p50 against hop-h
targets while every input still comes from ground truth, so a model that only
ever learned "delay stays roughly the same" scores well there. This module
instead hands the model its own predictions and never shows it the future.

METHOD
------
At a decision time T0 the model has seen the real day up to T0 and nothing
after it. From T0 onward the train's own predictions are written back into its
journey — arrival delay, departure delay, and therefore the wall-clock times
that decide where it sits in the graph — and the snapshot is rebuilt from that
forecast world. Feeding predictions back through `build_snapshot` rather than
patching feature tensors is deliberate: the rollout then exercises the exact
code path training used, so a divergence between the two is impossible by
construction rather than by review.

Errors compound the way they would in service: a hop-3 prediction rests on the
hop-1 and hop-2 predictions that placed the train there.

WHAT IS AND IS NOT ASSUMED KNOWN AFTER T0
-----------------------------------------
  known    : the working timetable, and planned exogenous events (maintenance
             blocks, TSRs, freight paths). A controller has these at T0.
  predicted: every corridor arrival and departure of every train present, and
             hence all occupancy, congestion and conflict state derived from
             them. No observed delay after T0 reaches the model.
  caveat   : a train that has not yet entered the corridor at T0 is taken over
             at its first corridor stop using its true entry delay. Corridor
             entry is set outside this corridor and this model does not claim
             to predict it. Those events are reported separately as
             `entered later` so the strict already-running subset can be read
             on its own — that subset is the number to quote.

    python -m cris_pipeline.rollout_cris --checkpoint data_cris/corridor_nextevent.pt
    python -m cris_pipeline.rollout_cris --checkpoint ... --show 6
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cris_pipeline.config import (                                   # noqa: E402
    JOURNEYS_DIR, TOPOLOGY_PATH, DATASET_DIR, OUT_DIR,
)
from cris_pipeline import build_dataset as BD                        # noqa: E402
from cris_pipeline.dataset_cris import CrisDataset                   # noqa: E402
from cris_pipeline.model_cris import CorridorNextEventGNN            # noqa: E402
from cris_pipeline.overlays import (                                 # noqa: E402
    FreightOverlay, DisruptionOverlay, DetentionOverlay,
)

import os
ZERO_DIMS = [int(x) for x in os.environ.get('ZERO_DIMS','').split(',') if x.strip()]
HORIZON_MIN = 240                       # forecast 4 h ahead, as the old system
LEAD_BUCKETS = [(0, 30), (30, 60), (60, 120), (120, 240)]
SPIKE = 30.0


# ── Forecast world ────────────────────────────────────────────────────────────

def _write_prediction(stops: list, k: int, arr_delay: float,
                      dep_delay: float) -> tuple[float, bool]:
    """Commit a predicted arrival/departure at stop k into the forecast world.

    Applies the same monotonicity clamp `_build_instance` applies to observed
    data, so a prediction can never place a train before the stop it just left
    (which would corrupt `_find_position` and put it in the wrong section).
    Returns the effective arrival delay and whether the clamp bound.
    """
    s = stops[k]
    raw_arr_abs = s["sched_arr_abs"] + arr_delay
    arr_abs = raw_arr_abs
    if k > 0:
        arr_abs = max(arr_abs, stops[k - 1]["actual_dep_abs"])
    clamped = arr_abs > raw_arr_abs + 1e-9

    dep_abs = max(s["sched_dep_abs"] + dep_delay, arr_abs)
    eff_arr = arr_abs - s["sched_arr_abs"]

    s["actual_arr_abs"] = arr_abs
    s["actual_dep_abs"] = dep_abs
    s["arr_delay"] = eff_arr
    s["dep_delay"] = dep_abs - s["sched_dep_abs"]
    s["delay_est"] = eff_arr
    # The rollout treats its own prediction as the observation it is standing
    # in for. Leaving this False would tell the model "this figure is inferred,
    # discount it" at every predicted stop — an input combination that never
    # occurs in training, so it would measure distribution shift, not skill.
    s["has_actual"] = True
    return eff_arr, clamped


def _state_copy(st: BD._StationState) -> BD._StationState:
    c = BD._StationState()
    c.cong = defaultdict(float, st.cong)
    c.crit = defaultdict(bool, st.crit)
    c.steps = defaultdict(int, st.steps)
    return c


class StateCheck:
    """Compares forecast-world node features against the ground-truth world at
    the same tick, so 'are the features actually being updated' is answerable
    with numbers instead of by reading the writer.

    Reported per feature: the mean in each world. A feature that is correct by
    construction (time of day) must match exactly; one that depends on
    predicted positions will drift, and HOW it drifts is the diagnosis.
    """

    def __init__(self, train_labels, station_labels):
        self.tl, self.sl = train_labels, station_labels
        self.f_sum, self.t_sum = None, None
        self.f_n = self.t_n = 0
        self.sf_sum, self.st_sum = None, None
        self.s_n = 0

    def add(self, fdata, tdata):
        fx, tx = fdata["train"].x, tdata["train"].x
        if self.f_sum is None:
            self.f_sum = torch.zeros(fx.shape[1])
            self.t_sum = torch.zeros(tx.shape[1])
        self.f_sum += fx.sum(0)
        self.t_sum += tx.sum(0)
        self.f_n += fx.shape[0]
        self.t_n += tx.shape[0]

        fs, ts = fdata["station"].x, tdata["station"].x
        if self.sf_sum is None:
            self.sf_sum = torch.zeros(fs.shape[1])
            self.st_sum = torch.zeros(ts.shape[1])
        self.sf_sum += fs.sum(0)
        self.st_sum += ts.sum(0)
        self.s_n += fs.shape[0]

    def report(self):
        print("\n\n" + "#" * 78)
        print("# STATE FIDELITY — forecast world vs ground truth at the same tick")
        print("#" * 78)
        print(f"  train nodes: forecast {self.f_n:,}  truth {self.t_n:,}"
              f"   ({100 * self.f_n / max(self.t_n, 1):.1f}% as many)")
        for name, fs, ts, n_f, n_t, labels in (
                ("TRAIN", self.f_sum, self.t_sum, self.f_n, self.t_n, self.tl),
                ("STATION", self.sf_sum, self.st_sum, self.s_n, self.s_n, self.sl)):
            if fs is None:
                continue
            print(f"\n  {name} features        {'forecast':>10} {'truth':>10} "
                  f"{'ratio':>8}")
            print("  " + "-" * 56)
            rows = []
            for j in range(len(fs)):
                a = float(fs[j]) / max(n_f, 1)
                b = float(ts[j]) / max(n_t, 1)
                nm = labels[j] if j < len(labels) else f"dim{j}"
                r = a / b if abs(b) > 1e-6 else (1.0 if abs(a) < 1e-6 else 99.0)
                rows.append((abs(r - 1.0), nm, a, b, r))
            for _, nm, a, b, r in sorted(rows, reverse=True)[:10]:
                flag = "  <-- DRIFT" if abs(r - 1.0) > 0.25 else ""
                print(f"  {nm:<24} {a:>10.4f} {b:>10.4f} {r:>8.2f}x{flag}")


class Rollout:
    """One forecast issued at T0, run forward to T0 + HORIZON_MIN."""

    def __init__(self, model, ds, day_idx, date, day_meta, freight, disrupt,
                 truth_by_iid, state_check=None, dep_mode='absolute',
                 hurdle_tau: float = 1.0):
        self.model, self.ds = model, ds
        self.day_idx, self.date, self.day_meta = day_idx, date, day_meta
        self.freight, self.disrupt = freight, disrupt
        self.truth = truth_by_iid
        self.state_check = state_check
        self.dep_mode = dep_mode
        self.hurdle_tau = hurdle_tau

    def run(self, t0, instances, st_state, prev_delay):
        """`instances` is already a deep copy owned by this rollout."""
        model = self.model
        by_iid = {i["instance_id"]: i for i in instances}
        routes = {i["instance_id"]: i["route"] for i in instances}

        # Stop each train occupies at T0 — everything after it is predicted.
        takeover, delay_at_t0 = {}, {}
        for inst in instances:
            pos = BD._find_position(inst, t0)
            if pos is not None:
                takeover[inst["instance_id"]] = pos["stop_idx"]

        # ERASE THE FUTURE from the forecast world.
        #
        # `instances` is a deep copy, but the stops the rollout has not reached
        # yet still hold their TRUE timestamps until a prediction overwrites
        # them. build_snapshot never reads past the current stop (proved by
        # audit_raw_causality), so features were safe — but `_find_position`
        # does read one thing beyond it:
        #
        #     if t > stops[-1]["actual_dep_abs"] + 30: return None
        #
        # In training that is causal: a last-departure lying in the future only
        # tells you "not finished yet", which is knowable at t. In a ROLLOUT it
        # is not, because it governs the forecast world with the TRUE finish
        # time instead of the predicted one — a train left the graph when
        # reality said it was done, not when the forecast said so. 83% of
        # journeys complete inside the 4 h horizon, so this fires often.
        #
        # Pushing unwritten stops to a sentinel far beyond the horizon means
        # the forecast world holds no future information at all: a train stays
        # in the graph until the model's own predictions carry it to its end,
        # and every value the rollout reads is either observed before T0 or
        # predicted by the model. Each prediction overwrites its sentinel as
        # the rollout advances.
        horizon_end = t0 + HORIZON_MIN
        sentinel = horizon_end + 10_000.0
        # A train that has NOT yet entered the corridor at T0 was previously
        # skipped here, so it kept its true timestamps for the whole journey
        # until predictions overwrote them one stop at a time. It is a node in
        # the graph, so its position fed congestion and conflict features to
        # the very trains we score. That is the same shape as the bug fixed
        # above, applied to the other population, and it contradicted the
        # documented assumption -- which grants only the true ENTRY delay, not
        # the whole journey.
        #
        # Careful: `_find_position` returns None for TWO different trains. One
        # has not entered yet; the other already FINISHED before T0. Only the
        # first has a future to erase. Erasing the second resurrects a
        # completed journey, because the "have we finished" test reads
        # stops[-1], which would then be a sentinel that never compares true --
        # measured at 153% of the true train count before this was separated.
        #
        # Measured cost of closing it: STRICT MAE 7.65 -> 7.60, spike recall
        # 82.4% -> 86.0%, train-node fidelity 100.2% of truth. The old numbers
        # were NOT materially optimistic; this is correctness, not a rescue.
        for inst in instances:
            k0 = takeover.get(inst["instance_id"])
            if k0 is None:
                if t0 >= inst["stops"][0]["actual_arr_abs"]:
                    continue                       # already finished: past
                k0 = 0                             # not entered: keep entry only
            for s in inst["stops"][k0 + 1:]:
                s["actual_arr_abs"] = sentinel
                s["actual_dep_abs"] = sentinel
                s["arr_delay"] = None
                s["dep_delay"] = None
                s["has_actual"] = False

        written: set[tuple[str, int]] = set()
        records, n_clamped = [], 0
        n_hurdle = [0]

        ticks = [m for m in BD.SNAPSHOT_SCHEDULE
                 if 0 <= (self.day_idx * 1440 + m) - t0 <= HORIZON_MIN]
        for m in ticks:
            t = self.day_idx * 1440 + m
            snap = BD.build_snapshot(self.day_idx, t, instances, st_state,
                                     self.day_meta, prev_delay, self.freight,
                                     self.disrupt, self.date)
            if not snap["train_nodes"]:
                continue
            data = self.ds.materialise(snap, by_iid, routes)
            if data is not None and ZERO_DIMS and data['train'].x.numel():
                for _dz in ZERO_DIMS:
                    data['train'].x[:, _dz] = 0.0
            if data is None or data["train"].x.numel() == 0:
                continue

            # STATE FIDELITY: build the SAME tick from ground truth and compare.
            # The forecast world is only as good as the state it reconstructs;
            # if predicted trains never stand at platforms, the very condition
            # that creates delay can never recur and the forecast must flatten.
            if self.state_check is not None:
                tsnap = BD.build_snapshot(
                    self.day_idx, t, list(self.truth.values()),
                    _state_copy(st_state), self.day_meta, dict(prev_delay),
                    self.freight, self.disrupt, self.date)
                if tsnap["train_nodes"]:
                    tdata = self.ds.materialise(
                        tsnap, self.truth,
                        {k: v["route"] for k, v in self.truth.items()})
                    if tdata is not None and tdata["train"].x.numel():
                        self.state_check.add(data, tdata)

            with torch.no_grad():
                out = model(data, update_memory=True)
            arr_q = out["next_arr"]
            dep_q = out["next_dep"]
            sp_logit = out.get("spike_logit")
            sp_mag = out.get("spike_mag")

            for j, n in enumerate(snap["train_nodes"]):
                iid = n["instance_id"]
                i = int(n["stop_idx"])
                k = i + 1
                inst = by_iid[iid]
                if k >= len(inst["stops"]) or (iid, k) in written:
                    continue
                # A train first seen after T0 is taken over where it appears.
                if iid not in takeover:
                    takeover[iid] = i
                if i < takeover[iid]:
                    continue

                cur = float(n["delay_min"])
                if iid not in delay_at_t0:
                    delay_at_t0[iid] = cur

                # HURDLE COMBINE. `next_arr` is the ordinary head and is what
                # every model up to v6 used alone. When the classifier clears
                # the threshold we take the magnitude head instead, which was
                # trained only on large changes and so is not dragged toward
                # the quiet answer.
                #
                # The threshold is the dial that used to be buried inside the
                # loss: raise it for fewer false alarms, lower it to catch more.
                # At 1.0 nothing ever fires and this reduces EXACTLY to the
                # pre-hurdle model, which is the degeneracy test.
                delta = float(arr_q[j])
                if sp_logit is not None and self.hurdle_tau < 1.0:
                    p_big = 1.0 / (1.0 + math.exp(-float(sp_logit[j])))
                    if p_big > self.hurdle_tau:
                        delta = float(sp_mag[j])
                        n_hurdle[0] += 1
                pred_arr = cur + delta
                # The dep head means different things in the two regimes;
                # composing them wrongly silently corrupts every rollout.
                pred_dep = (pred_arr + float(dep_q[j])
                            if self.dep_mode == 'extension'
                            else cur + float(dep_q[j]))

                eff, clamped = _write_prediction(inst["stops"], k,
                                                 pred_arr, pred_dep)
                n_clamped += int(clamped)
                written.add((iid, k))

                tstops = self.truth[iid]["stops"]
                if k >= len(tstops):
                    continue
                ts = tstops[k]
                if not (ts["is_target"] and ts["has_actual"]
                        and ts["arr_delay"] is not None
                        and self.truth[iid]["target_eligible"]):
                    continue

                records.append({
                    "iid": iid,
                    "train_no": self.truth[iid]["train_no"],
                    "train_type": self.truth[iid].get("train_type", ""),
                    "stop_k": k,
                    "station": ts["station_code"],
                    "hop": k - takeover[iid],
                    "lead_min": float(ts["actual_arr_abs"]) - t0,
                    "pred": eff,
                    "actual": float(ts["arr_delay"]),
                    # Persistence = the delay this train had at T0, held flat.
                    "persist": delay_at_t0[iid],
                    # Was this train inside the corridor at the decision time?
                    # Tested against TRUTH, since the forecast world has been
                    # rewritten by then. Only these are quoted as the headline.
                    "already_running": BD._find_position(
                        self.truth[iid], t0) is not None,
                    "t0": t0,
                })

        if n_hurdle[0]:
            print(f"    hurdle head fired on {n_hurdle[0]} predictions", flush=True)
        return records, n_clamped


# ── Reporting ─────────────────────────────────────────────────────────────────

def _table(title, rows, keyfn, keys, keyname):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"{keyname:>10} {'n':>7} {'MAE':>8} {'persist':>9} {'gain':>8} "
          f"{'bias':>8} {'P90err':>8}")
    for key in keys:
        sel = [r for r in rows if keyfn(r) == key]
        if not sel:
            continue
        n = len(sel)
        errs = sorted(abs(r["pred"] - r["actual"]) for r in sel)
        mae = sum(errs) / n
        per = sum(abs(r["persist"] - r["actual"]) for r in sel) / n
        bias = sum(r["pred"] - r["actual"] for r in sel) / n
        p90 = errs[min(int(0.9 * n), n - 1)]
        print(f"{str(key):>10} {n:>7} {mae:>8.2f} {per:>9.2f} "
              f"{per - mae:>+8.2f} {bias:>+8.2f} {p90:>8.2f}")


def _bucket(v):
    for lo, hi in LEAD_BUCKETS:
        if lo <= v < hi:
            return f"{lo}-{hi}m"
    return None


def report(rows, n_clamped, label):
    if not rows:
        print(f"\n[{label}] no scored events")
        return
    print("\n\n" + "#" * 78)
    print(f"# {label}   ({len(rows):,} forecast events)")
    print("#" * 78)

    _table("MAE BY LEAD TIME  (wall-clock minutes from decision to event)",
           rows, lambda r: _bucket(r["lead_min"]),
           [f"{lo}-{hi}m" for lo, hi in LEAD_BUCKETS], "lead")

    hops = sorted({r["hop"] for r in rows if r["hop"] <= 8})
    _table("MAE BY HOP  (stations ahead — errors compound autoregressively)",
           rows, lambda r: r["hop"], hops, "hop")

    # Spikes: the event the old system missed 70-87% of.
    sp = [r for r in rows if r["actual"] - r["persist"] > SPIKE]
    if sp:
        hit = sum(1 for r in sp if r["pred"] - r["persist"] > SPIKE * 0.5)
        pred_sp = [r for r in rows if r["pred"] - r["persist"] > SPIKE * 0.5]
        tp = sum(1 for r in pred_sp if r["actual"] - r["persist"] > SPIKE)
        print("\n" + "=" * 78)
        print("SPIKE DETECTION UNDER ROLLOUT  (actual rise > 30 min vs T0)")
        print("=" * 78)
        print(f"  spikes            : {len(sp)}")
        print(f"  recall            : {hit / len(sp):.1%}   "
              f"(old system: 13-30%)")
        print(f"  precision         : {tp / max(len(pred_sp), 1):.1%}")
        print(f"  spike MAE         : "
              f"{sum(abs(r['pred'] - r['actual']) for r in sp) / len(sp):.1f} min")
        under = sum(1 for r in sp if r["pred"] < r["actual"])
        print(f"  underpredicted    : {under}/{len(sp)} = "
              f"{under / len(sp):.0%}   (old system: 100%)")

    rec = [r for r in rows if r["persist"] - r["actual"] > SPIKE]
    if rec:
        hit = sum(1 for r in rec if r["persist"] - r["pred"] > SPIKE * 0.5)
        print("\n" + "=" * 78)
        print("RECOVERY DETECTION  (actual fall > 30 min vs T0)")
        print("=" * 78)
        print(f"  recoveries        : {len(rec)}")
        print(f"  recall            : {hit / len(rec):.1%}")
        print(f"  recovery MAE      : "
              f"{sum(abs(r['pred'] - r['actual']) for r in rec) / len(rec):.1f} min")

    # PER-JOURNEY ROBUST VIEW.
    #
    # The event-level mean below is dominated by a handful of runaway
    # journeys: measured on the 96-journey ship set, ONE journey moved the
    # v3-vs-v5 verdict from "v5 better" to "v5 worse", and three of them
    # flipped the sign. Every model comparison in this project has landed
    # inside the noise band, and this is why -- the headline statistic is not
    # sensitive enough to read the experiments we run on it.
    #
    # Median and trimmed mean say what happens on a TYPICAL journey; the mean
    # says what happens to the total minute count, which the tail owns. Report
    # both, and when they disagree, the disagreement IS the finding.
    by_j = defaultdict(list)
    for r in rows:
        by_j[(r["iid"], r["t0"])].append(r)
    jm, jp = [], []
    for v in by_j.values():
        jm.append(sum(abs(r["pred"] - r["actual"]) for r in v) / len(v))
        jp.append(sum(abs(r["persist"] - r["actual"]) for r in v) / len(v))

    def _q(x, f):
        x = sorted(x)
        return x[min(int(len(x) * f), len(x) - 1)]

    def _trim(x, f=0.05):
        x = sorted(x)
        k = int(len(x) * f)
        x = x[k:len(x) - k] or x
        return sum(x) / len(x)

    runaway = 0
    for v in by_j.values():
        if len(v) < 4:
            continue
        e = [abs(r["pred"] - r["actual"]) for r in sorted(v, key=lambda r: r["hop"])]
        if e[-1] > 40 and e[-1] > e[len(e) // 2] + 20:
            runaway += 1

    print("\n" + "=" * 78)
    print(f"PER-JOURNEY ERROR  ({len(jm):,} journeys) — the tail-robust view")
    print("=" * 78)
    print(f"  {'statistic':18} {'persistence':>12} {'model':>9} {'gain':>8}")
    for nm, f in (("mean", lambda x: sum(x) / len(x)),
                  ("MEDIAN", lambda x: _q(x, 0.5)),
                  ("5% trimmed mean", _trim),
                  ("75th percentile", lambda x: _q(x, 0.75)),
                  ("90th percentile", lambda x: _q(x, 0.90))):
        m, p = f(jm), f(jp)
        print(f"  {nm:18} {p:>12.2f} {m:>9.2f} {p - m:>+8.2f}")
    print(f"\n  runaway journeys (error grew >20 min in the second half "
          f"and ended >40): {runaway}")

    n = len(rows)
    mae = sum(abs(r["pred"] - r["actual"]) for r in rows) / n
    per = sum(abs(r["persist"] - r["actual"]) for r in rows) / n
    print("\n" + "-" * 78)
    print(f"OVERALL  MAE {mae:.2f}   persistence {per:.2f}   "
          f"gain {per - mae:+.2f}   (monotonicity clamp fired {n_clamped}x)")


def _hhmm(x) -> str:
    hh, mm = divmod(int(round(x)) % 1440, 60)
    return f"{hh:02d}:{mm:02d}"


def journey_view(rows, truth, n_show, want_train=None, want_at=None):
    """The whole run on one page: what was OBSERVED up to the decision time,
    then what was FORECAST after it, against what actually happened.

    This is the view that makes propagation legible — you can see the delay the
    train carried into the corridor, how it moved leg by leg while we were
    still watching, and then whether the model continued that trajectory
    correctly once it was flying blind.
    """
    by_run = defaultdict(list)
    for r in rows:
        if want_train and str(r["train_no"]) != str(want_train):
            continue
        if want_at is not None and int(r["t0"]) % 1440 != int(want_at):
            continue
        by_run[(r["t0"], r["iid"])].append(r)
    if not by_run:
        print("\nno runs match that filter")
        return

    # Show a SPREAD of scenarios, not just the biggest movers. Sorting purely
    # by magnitude returns four variations of the same runaway train and says
    # nothing about whether the model handles an ordinary run, or a train that
    # recovers time — which is the case a controller most needs to trust.
    def scenario(rs):
        rise = max(x["actual"] - x["persist"] for x in rs)
        fall = max(x["persist"] - x["actual"] for x in rs)
        if rise > SPIKE:
            return "SPIKE"
        if fall > SPIKE:
            return "RECOVERY"
        if max(abs(x["actual"] - x["persist"]) for x in rs) < 10:
            return "STEADY"
        return "MODERATE"

    buckets = defaultdict(list)
    for k, rs in by_run.items():
        buckets[scenario(rs)].append((k, rs))
    for b in buckets:
        buckets[b].sort(key=lambda kv: -len(kv[1]))       # longest journeys

    order = ["SPIKE", "RECOVERY", "MODERATE", "STEADY"]
    picked, i = [], 0
    while len(picked) < n_show and i < 50:                 # round-robin
        for b in order:
            if i < len(buckets.get(b, [])) and len(picked) < n_show:
                picked.append((b, *buckets[b][i]))
        i += 1

    print("\n\n" + "#" * 78)
    print("# PER-TRAIN JOURNEY — observed history, then forecast vs actual")
    print(f"# scenarios available: "
          + "  ".join(f"{b}={len(buckets.get(b, []))}" for b in order))
    print("#" * 78)

    for kind, (t0, iid), rs in picked:
        rs = sorted(rs, key=lambda r: r["stop_k"])
        inst = truth.get(iid)
        if inst is None:
            continue
        takeover = rs[0]["stop_k"] - rs[0]["hop"]
        stops = inst["stops"]

        print(f"\n{'=' * 74}")
        print(f"  [{kind}]  Train {inst['train_no']}  "
              f"({inst.get('train_class','?')}, "
              f"{inst.get('direction','?')})   {inst['corridor_date']}")
        print(f"  Forecast issued {_hhmm(t0)}   "
              f"delay at that moment {rs[0]['persist']:+.0f} min")
        entry = inst.get("entry") or {}
        if entry.get("has_upstream"):
            print(f"  Entered corridor {float(entry.get('entry_delay_min',0)):+.0f} "
                  f"min late from upstream "
                  f"({float(entry.get('km_before_corridor',0)):.0f} km before)")
        print(f"{'=' * 74}")
        print(f"  {'station':<8} {'sched':>6} {'actual':>7} {'delay':>7} "
              f"{'pred':>7} {'error':>7}   note")
        print(f"  {'-' * 68}")

        # ── Observed: everything up to and including the decision point ──
        for k in range(0, min(takeover + 1, len(stops))):
            s = stops[k]
            d = s.get("arr_delay")
            shown = f"{float(d):+7.0f}" if d is not None else "      ?"
            act = (_hhmm(s["actual_arr_abs"])
                   if s.get("has_actual") and d is not None else "      -")
            note = "" if s.get("has_actual") else "not reported"
            if k == takeover:
                note = (note + "  <== NOW").strip()
            print(f"  {str(s['station_code']):<8} "
                  f"{_hhmm(s['sched_arr_abs']):>6} {act:>7} {shown} "
                  f"{'':>7} {'':>7}   {note}")

        print(f"  {'- ' * 34}")
        print(f"  {'':<8} {'':>6} {'':>7} {'':>7} {'':>7} {'':>7}   "
              f"forecast from here — no observations used")
        print(f"  {'- ' * 34}")

        # ── Forecast: hop by hop, autoregressive ──
        for r in rs:
            s = stops[r["stop_k"]]
            err = r["pred"] - r["actual"]
            flag = ""
            if r["actual"] - r["persist"] > SPIKE:
                flag = "SPIKE" if err > -SPIKE * 0.5 else "SPIKE MISSED"
            elif r["persist"] - r["actual"] > SPIKE:
                flag = "recovery"
            print(f"  {str(r['station']):<8} "
                  f"{_hhmm(s['sched_arr_abs']):>6} "
                  f"{_hhmm(s['actual_arr_abs']):>7} "
                  f"{r['actual']:>+7.0f} {r['pred']:>+7.0f} {err:>+7.0f}   "
                  f"h{r['hop']} +{r['lead_min']:.0f}m {flag}")

        n = len(rs)
        mae = sum(abs(r["pred"] - r["actual"]) for r in rs) / n
        per = sum(abs(r["persist"] - r["actual"]) for r in rs) / n
        print(f"  {'-' * 68}")
        print(f"  {n} forecast stops   MAE {mae:.1f} min   "
              f"persistence {per:.1f} min   gain {per - mae:+.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUT_DIR / "corridor_nextevent.pt"))
    ap.add_argument("--journeys", default=str(JOURNEYS_DIR))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--split", default="test")
    ap.add_argument("--every", type=int, default=90,
                    help="minutes between decision times")
    ap.add_argument("--first", type=int, default=330)
    ap.add_argument("--last", type=int, default=1290)
    ap.add_argument("--show", type=int, default=0,
                    help="print N per-train journey tables")
    ap.add_argument("--train", default=None,
                    help="restrict --show to this train number")
    ap.add_argument("--at", type=int, default=None,
                    help="restrict --show to this decision time (minute of day)")
    ap.add_argument("--check-state", action="store_true",
                    help="rebuild each tick from ground truth too and compare "
                         "features, to verify the forecast world updates right")
    ap.add_argument("--out", default=None, help="write raw records as JSON")
    # Comparing two models is only meaningful on identical ground. Both of
    # these exist so a run can be pinned to the same days and the same train
    # population as an earlier run, even when the split moved underneath it
    # (adding supervised train types changed which boundary days survive
    # _drop_partial_days, which shifted every split by one day).
    ap.add_argument("--hurdle-tau", type=float, default=1.0,
                    help="P(large change) above which the magnitude head "
                         "replaces the ordinary head. 1.0 disables it, "
                         "which reproduces the pre-hurdle model exactly. "
                         "Tune on VALIDATION days, never on test.")
    ap.add_argument("--dates", default=None,
                    help="comma-separated dates to score, overriding --split")
    ap.add_argument("--only-types", default=None,
                    help="comma-separated train_type codes to SCORE; others "
                         "still run as actors but are not counted")
    args = ap.parse_args()

    topo = json.loads(Path(args.topology).read_text(encoding="utf-8"))
    BD._init_topology(topo)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CorridorNextEventGNN(**ck["config"])
    # Checkpoints from before the hurdle heads existed (v1..v6) carry no
    # spike_logit_head / spike_mag_head. Load them anyway so every historical
    # model stays scoreable on the current code -- that comparability is the
    # only reason the release directories are worth keeping. Their hurdle
    # weights stay at init and are never consulted, because a missing head
    # forces the threshold to 1.0 below.
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    hurdle_ok = not any(k.startswith(("spike_logit_head", "spike_mag_head"))
                        for k in missing)
    other = [k for k in missing
             if not k.startswith(("spike_logit_head", "spike_mag_head"))]
    if other or unexpected:
        raise SystemExit(f"checkpoint does not match the model: "
                         f"missing={other} unexpected={list(unexpected)}")
    if not hurdle_ok:
        if args.hurdle_tau < 1.0:
            print("  ! checkpoint predates the hurdle heads — "
                  "--hurdle-tau ignored")
        args.hurdle_tau = 1.0
    model.eval()
    dep_mode = ck.get("dep_mode", "absolute")
    print(f"checkpoint epoch {ck['epoch']}  val_mae {ck['val_mae']:.2f}"
          f"  dep_mode={dep_mode}")

    ds = CrisDataset(args.dataset, args.topology, lazy=True, verbose=False)

    freight = FreightOverlay.load()
    section_pairs = {(int(s["from_id"]), int(s["to_id"]))
                     for s in BD._SECTIONS.values()}
    disrupt = DisruptionOverlay.load(section_pairs)
    detent = DetentionOverlay.load()

    # Reproduce build_dataset's exact day indexing and split assignment so
    # `day_idx` (and therefore every absolute minute) matches the dataset.
    by_date = BD._load_journeys(Path(args.journeys))
    dates = BD._drop_partial_days(by_date, sorted(by_date))
    splits = BD._assign_splits(dates)
    if args.dates:
        want = [d.strip() for d in args.dates.split(",") if d.strip()]
        missing = [d for d in want if d not in dates]
        if missing:
            raise SystemExit(f"--dates not present in this build: {missing}\n"
                             f"available: {dates[0]} .. {dates[-1]}")
        target_dates = want
        print(f"pinned dates: {len(target_dates)} days  {target_dates}")
    else:
        target_dates = [d for d in dates if splits[d] == args.split]
        print(f"{args.split}: {len(target_dates)} days  {target_dates}")

    only_types = ({t.strip().upper() for t in args.only_types.split(",")}
                  if args.only_types else None)
    if only_types:
        print(f"scoring only train types: {sorted(only_types)}")

    decisions = list(range(args.first, args.last + 1, args.every))
    print(f"decision times/day: {len(decisions)}  "
          f"horizon {HORIZON_MIN} min\n")

    state_check = None
    if args.check_state:
        from cris_pipeline.audit_features import TRAIN_LABELS, STATION_LABELS
        state_check = StateCheck(TRAIN_LABELS, STATION_LABELS)

    all_rows, total_clamped = [], 0
    truth_all: dict[str, dict] = {}       # pristine journeys, for the viewer
    for date in target_dates:
        di = dates.index(date)
        d_obj = dt.date.fromisoformat(date)
        day_meta = {"day_of_week": d_obj.weekday()}

        instances = []
        for rec in by_date[date]:
            inst = BD._build_instance(rec, di)
            if inst:
                inst["detention"] = detent.get(
                    inst["train_no"], inst.get("journey_date") or date) or {}
                instances.append(inst)
        if not instances:
            continue
        truth_by_iid = {i["instance_id"]: i for i in instances}
        # Truth is never mutated — the rollout owns a deep copy — so these can
        # be kept as-is for the journey view.
        truth_all.update(truth_by_iid)
        by_iid = truth_by_iid
        routes = {i["instance_id"]: i["route"] for i in instances}

        # One chronological pass over the REAL day. At each decision time the
        # memory and station state are forked and a rollout is run from there,
        # so the model enters every forecast having seen exactly the history a
        # controller would have at that moment — no more, no less.
        st_state = BD._StationState()
        prev_delay: dict[str, float] = {}
        model.reset_memory()
        day_rows, day_clamped = [], 0
        pending = set(decisions)

        for m in BD.SNAPSHOT_SCHEDULE:
            t = di * 1440 + m
            if m in pending:
                pending.discard(m)
                saved = (model.station_memory.clone(),
                         model.service_memory.clone(),
                         model.section_memory.clone())
                ro = Rollout(model, ds, di, date, day_meta, freight, disrupt,
                             truth_by_iid, state_check, dep_mode,
                             hurdle_tau=args.hurdle_tau)
                rows, nc = ro.run(t, copy.deepcopy(instances),
                                  _state_copy(st_state), dict(prev_delay))
                day_rows += rows
                day_clamped += nc
                (model.station_memory, model.service_memory,
                 model.section_memory) = saved

            snap = BD.build_snapshot(di, t, instances, st_state, day_meta,
                                     prev_delay, freight, disrupt, date)
            if not snap["train_nodes"]:
                continue
            data = ds.materialise(snap, by_iid, routes)
            if data is None or data["train"].x.numel() == 0:
                continue
            with torch.no_grad():
                model(data, update_memory=True)

        if only_types:
            kept = [r for r in day_rows
                    if str(r.get("train_type", "")).upper() in only_types]
            dropped = len(day_rows) - len(kept)
            day_rows = kept
        else:
            dropped = 0
        all_rows += day_rows
        total_clamped += day_clamped
        print(f"  {date}: {len(day_rows):5d} forecast events"
              + (f"  ({dropped} not scored: type filter)" if dropped else ""),
              flush=True)

    strict = [r for r in all_rows if r["already_running"]]
    report(strict, total_clamped, "STRICT — trains already running at T0")
    later = [r for r in all_rows if not r["already_running"]]
    if later:
        report(later, 0, "ENTERED LATER — true entry delay assumed known")

    if state_check is not None:
        state_check.report()

    if args.show:
        journey_view(strict, truth_all, args.show,
                     want_train=args.train, want_at=args.at)

    if args.out:
        Path(args.out).write_text(json.dumps(all_rows, indent=1),
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
