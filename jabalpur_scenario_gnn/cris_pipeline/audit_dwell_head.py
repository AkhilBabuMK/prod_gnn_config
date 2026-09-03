# -*- coding: utf-8 -*-
"""
cris_pipeline/audit_dwell_head.py
=================================
Validate `remaining_dwell`, the v13 head — "how many more minutes will this
train stand on the platform it is on right now?"

WHY THIS EXISTS

The trainer adds a DWELL_LOSS_WEIGHT term for this head but reports nothing
about it: `val_mae` is computed on `next_arr` alone. Without this script v13
would be judged only by a rollout number, with no way to tell a head that
learned nothing from a head that learned something the rollout then wasted.

That is exactly how the departure head survived nine unmeasured runs before
audit_dep_head.py finally scored it at 7% recall on holds over 30 min — worse
than predicting zero. This head must clear that bar or v13's premise is wrong.

THE BASELINES IT MUST BEAT

  * predict 0            — "it leaves now". What the rollout effectively does
                           for a train already past its booked departure.
  * best constant        — the target's own median, the strongest predictor
                           available to something that has learned nothing
                           train-specific. Beating MAE-at-0 but not this means
                           the head learned the average stand, not the stand.

THE TEST THAT MATTERS

Recall on remaining stands over 15 and over 30 minutes. A head that never
commits to a long hold cannot extend a departure in rollout, and cannot move
the number, whatever its mean error looks like.

STAGED SUPERVISION CHECK

v13's other change was to stop deduplicating supervision to the first tick of a
stand, so a train standing 40 min is now taught at 4 stages instead of 1.
Accuracy is therefore broken out by how long the train has ALREADY been
standing. If the fix worked, the later stages are no longer blind.

Read-only. Touches no checkpoint and no dataset.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from cris_pipeline.config import DATASET_DIR, TOPOLOGY_PATH, OUT_DIR
from cris_pipeline.dataset_cris import CrisDataset
from cris_pipeline.model_cris import (
    CorridorNextEventGNN, load_weights, check_feature_dims,
)

# dataset_cris feature layout — see _train_features
F_TIME_AT_STOP = 12          # min(time_at_stop_min / 30, 1.5)

_STAGES = [(0.0, 5.0, "0-5 min standing"), (5.0, 15.0, "5-15"),
           (15.0, 30.0, "15-30"), (30.0, 1e9, "over 30")]


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def _mae(ps: list[float], ts: list[float]) -> float:
    return sum(abs(p - t) for p, t in zip(ps, ts)) / max(len(ts), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUT_DIR / "corridor_v13a.pt"))
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ds = CrisDataset(args.dataset, args.topology, lazy=True)
    check_feature_dims(ck["config"], ds)
    model = CorridorNextEventGNN(**ck["config"])
    _, dwell_ok = load_weights(model, ck["model"], verbose=False)
    model.eval()

    vm = ck.get("val_mae")
    print(f"checkpoint  epoch {ck.get('epoch')}"
          f"  val_mae {vm:.3f}" if vm is not None else "")
    print(f"honest_standing_delay = {ck.get('honest_standing_delay', False)}"
          f"   split = {args.split}")
    if not dwell_ok:
        print("\nthis checkpoint has NO trained remaining_dwell head "
              "(pre-v13). Nothing to audit.")
        return

    preds: list[float] = []
    tgts: list[float] = []
    stood: list[float] = []          # minutes already standing

    with torch.no_grad():
        for day in ds.iter_days(args.split):
            model.reset_memory()
            for snap in day:
                d = snap.data
                if d["train"].x.numel() == 0:
                    continue
                out = model(d, update_memory=True)
                m = d["train"].y_remaining_dwell_mask > 0.5
                if not bool(m.any()):
                    continue
                preds += out["remaining_dwell"][m].tolist()
                tgts += d["train"].y_remaining_dwell[m].tolist()
                stood += (d["train"].x[m, F_TIME_AT_STOP] * 30.0).tolist()

    n = len(tgts)
    if n == 0:
        print("\nno supervised remaining-dwell events in this split.")
        return

    med = _median(tgts)
    head = _mae(preds, tgts)
    zero = _mae([0.0] * n, tgts)
    const = _mae([med] * n, tgts)
    bias = sum(p - t for p, t in zip(preds, tgts)) / n

    print("\n" + "=" * 76)
    print(f"REMAINING-DWELL HEAD — {n:,} supervised standing observations")
    print("=" * 76)
    print(f"  head MAE                 {head:7.3f} min")
    print(f"  predict 0 ('leaves now') {zero:7.3f} min   gain {zero - head:+6.3f}")
    print(f"  best constant ({med:.1f} min)  {const:7.3f} min   gain {const - head:+6.3f}"
          f"   <- THE BAR")
    print(f"  bias (pred - actual)     {bias:+7.3f} min")

    print(f"\n  actual remaining : median {_pct(tgts, .50):6.2f}"
          f"   p90 {_pct(tgts, .90):6.2f}   p99 {_pct(tgts, .99):6.2f}"
          f"   max {max(tgts):6.1f}")
    print(f"  head prediction  : median {_pct(preds, .50):6.2f}"
          f"   p90 {_pct(preds, .90):6.2f}   p99 {_pct(preds, .99):6.2f}"
          f"   max {max(preds):6.1f}")

    tot_a, tot_p = sum(tgts), sum(preds)
    print(f"\n  TOTAL standing time")
    print(f"    actual  {tot_a:10.1f} min")
    print(f"    head    {tot_p:10.1f} min   "
          f"= {tot_p / max(tot_a, 1e-9) * 100:.1f}% of reality")

    print("\n" + "-" * 76)
    print("LONG-HOLD DETECTION  (the departure head managed 7% at >30 min)")
    print("-" * 76)
    for thr in (5.0, 15.0, 30.0):
        act = [t > thr for t in tgts]
        prd = [p > thr for p in preds]
        tp = sum(1 for a, b in zip(act, prd) if a and b)
        fp = sum(1 for a, b in zip(act, prd) if b and not a)
        fn = sum(1 for a, b in zip(act, prd) if a and not b)
        rec = tp / max(tp + fn, 1) * 100
        pre = tp / max(tp + fp, 1) * 100
        print(f"  remaining > {thr:4.0f} min : {tp + fn:5,} real   "
              f"recall {rec:5.1f}%   precision {pre:5.1f}%   fired {tp + fp:,}")

    print("\n" + "-" * 76)
    print("CALIBRATION BY ACTUAL REMAINING")
    print("-" * 76)
    buckets = [(0.0, 3.0, "0-3 (about to go)"), (3.0, 10.0, "3-10"),
               (10.0, 20.0, "10-20"), (20.0, 45.0, "20-45"),
               (45.0, 1e9, "over 45")]
    rows = defaultdict(list)
    for p, t in zip(preds, tgts):
        for lo, hi, nm in buckets:
            if lo <= t < hi:
                rows[nm].append((p, t))
                break
    print(f"  {'bucket':22}{'n':>7}{'mean actual':>13}{'mean pred':>11}{'captured':>10}")
    for _, _, nm in buckets:
        r = rows.get(nm, [])
        if not r:
            continue
        ma = sum(t for _, t in r) / len(r)
        mp = sum(p for p, _ in r) / len(r)
        cap = f"{mp / ma * 100:.0f}%" if abs(ma) > 1e-6 else "-"
        print(f"  {nm:22}{len(r):7,}{ma:13.2f}{mp:11.2f}{cap:>10}")

    print("\n" + "-" * 76)
    print("BY STAGE OF THE STAND  (did the staged-supervision fix take?)")
    print("-" * 76)
    print(f"  {'already standing':22}{'n':>7}{'head MAE':>11}"
          f"{'best const':>12}{'gain':>8}")
    for lo, hi, nm in _STAGES:
        idx = [i for i, s in enumerate(stood) if lo <= s < hi]
        if not idx:
            continue
        p = [preds[i] for i in idx]
        t = [tgts[i] for i in idx]
        c = _mae([_median(t)] * len(t), t)
        h = _mae(p, t)
        print(f"  {nm:22}{len(t):7,}{h:11.3f}{c:12.3f}{c - h:+8.3f}")

    print("\nnote: time-already-standing is read back from feature 12, which "
          "saturates at 45 min.")


if __name__ == "__main__":
    main()
