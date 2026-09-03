# -*- coding: utf-8 -*-
"""
cris_pipeline/audit_dep_head.py
===============================
Validate `next_dep`, the departure head — the one output that has never had a
number attached to it.

WHY THIS EXISTS

`val_mae` in train_cris.py is computed on `next_arr` alone. `next_dep` appears
at exactly one place at inference (composing the departure time in the rollout)
and nowhere in eval_cris.py. It is 30,977 parameters and a 0.5-weighted loss
term that has been trained across nine runs unmeasured.

WHAT IT PREDICTS

    target = dep_delay(next stop) - arr_delay(next stop)
           = (actual_dep - actual_arr) - (sched_dep - sched_arr)
           = ACTUAL DWELL - BOOKED DWELL

i.e. how much longer the train stands at the next station than its timetable
allows. That quantity is what physically blocks a platform, which is why it was
introduced: the forecast world was reconstructing `time_at_stop` at 0.49x of
reality, so the delay-generating mechanism could not recur after T0.

WHAT THIS MEASURES

Teacher-forced, on held-out days, so the head is judged on its own prediction
rather than through the compounding of a rollout:

  * MAE and bias against the persistence-equivalent baseline (always predict 0,
    i.e. "every train dwells exactly as booked")
  * calibration by target bucket — the question that matters is not average
    error but whether the head ever commits to a LONG hold
  * total predicted dwell extension against total actual, which is the direct
    explanation for a `time_at_stop` reconstruction ratio below 1.0
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


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUT_DIR / "corridor_v6.pt"))
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ds = CrisDataset(args.dataset, args.topology, lazy=True)
    check_feature_dims(ck["config"], ds)
    model = CorridorNextEventGNN(**ck["config"])
    load_weights(model, ck["model"])
    model.eval()
    print(f"checkpoint epoch {ck.get('epoch')}  val_mae {ck.get('val_mae'):.3f}"
          f"  dep_mode={ck.get('dep_mode')}  split={args.split}\n")

    dep_p: list[float] = []
    dep_t: list[float] = []
    arr_p: list[float] = []
    arr_t: list[float] = []

    with torch.no_grad():
        for day in ds.iter_days(args.split):
            model.reset_memory()
            for snap in day:
                d = snap.data
                if d["train"].x.numel() == 0:
                    continue
                out = model(d, update_memory=True)

                # Same composition the trainer supervises: the departure head
                # predicts the EXTENSION beyond the predicted arrival.
                tgt = d["train"].y_next_dep - d["train"].target_abs[:, 0]
                m = d["train"].y_next_dep_mask > 0.5
                if m.any():
                    dep_p += out["next_dep"][m].tolist()
                    dep_t += tgt[m].tolist()

                am = d["train"].target_mask[:, 0] > 0.5
                if am.any():
                    arr_p += out["next_arr"][am].tolist()
                    arr_t += (d["train"].target_delta[:, 0])[am].tolist()

    n = len(dep_t)
    print("=" * 76)
    print(f"DEPARTURE HEAD — dwell extension beyond booked, {n:,} supervised events")
    print("=" * 76)

    mae = sum(abs(p - t) for p, t in zip(dep_p, dep_t)) / n
    base = sum(abs(t) for t in dep_t) / n          # always predict 0
    bias = sum(p - t for p, t in zip(dep_p, dep_t)) / n
    print(f"  head MAE                   {mae:7.3f} min")
    print(f"  'dwell exactly as booked'  {base:7.3f} min   <- predict 0 always")
    print(f"  gain over that baseline    {base - mae:+7.3f} min")
    print(f"  bias (pred - actual)       {bias:+7.3f} min")

    print(f"\n  actual extension : median {_pct(dep_t, .50):6.2f}"
          f"   p90 {_pct(dep_t, .90):6.2f}   p99 {_pct(dep_t, .99):6.2f}"
          f"   max {max(dep_t):6.1f}")
    print(f"  head prediction  : median {_pct(dep_p, .50):6.2f}"
          f"   p90 {_pct(dep_p, .90):6.2f}   p99 {_pct(dep_p, .99):6.2f}"
          f"   max {max(dep_p):6.1f}")

    tot_a = sum(t for t in dep_t if t > 0)
    tot_p = sum(p for p in dep_p if p > 0)
    print(f"\n  TOTAL positive dwell extension")
    print(f"    actual  {tot_a:10.1f} min")
    print(f"    head    {tot_p:10.1f} min   = {tot_p / max(tot_a, 1e-9) * 100:.1f}% of reality")

    print("\n" + "-" * 76)
    print("CALIBRATION BY ACTUAL EXTENSION")
    print("-" * 76)
    buckets = [(-1e9, -0.5, "negative (left early)"), (-0.5, 0.5, "on time (0)"),
               (0.5, 5, "+0.5 to 5"), (5, 15, "+5 to 15"),
               (15, 30, "+15 to 30"), (30, 1e9, "over +30")]
    rows = defaultdict(list)
    for p, t in zip(dep_p, dep_t):
        for lo, hi, nm in buckets:
            if lo <= t < hi:
                rows[nm].append((p, t))
                break
    print(f"  {'bucket':24}{'n':>7}{'mean actual':>13}{'mean pred':>11}{'captured':>10}")
    for _, _, nm in buckets:
        r = rows.get(nm, [])
        if not r:
            continue
        ma = sum(t for _, t in r) / len(r)
        mp = sum(p for p, _ in r) / len(r)
        cap = f"{mp / ma * 100:.0f}%" if abs(ma) > 1e-6 else "—"
        print(f"  {nm:24}{len(r):7,}{ma:13.2f}{mp:11.2f}{cap:>10}")

    print("\n" + "-" * 76)
    print("HOLD DETECTION  (does it ever commit to a long stand?)")
    print("-" * 76)
    for thr in (5.0, 10.0, 30.0):
        act = [t > thr for t in dep_t]
        prd = [p > thr for p in dep_p]
        tp = sum(1 for a, b in zip(act, prd) if a and b)
        fp = sum(1 for a, b in zip(act, prd) if b and not a)
        fn = sum(1 for a, b in zip(act, prd) if a and not b)
        rec = tp / max(tp + fn, 1) * 100
        pre = tp / max(tp + fp, 1) * 100
        print(f"  extension > {thr:4.0f} min : {tp + fn:5,} real   "
              f"recall {rec:5.1f}%   precision {pre:5.1f}%   fired {tp + fp:,}")

    na = len(arr_t)
    amae = sum(abs(p - t) for p, t in zip(arr_p, arr_t)) / na
    abase = sum(abs(t) for t in arr_t) / na
    print("\n" + "=" * 76)
    print(f"ARRIVAL HEAD for reference — {na:,} events")
    print("=" * 76)
    print(f"  head MAE {amae:7.3f}   persistence {abase:7.3f}"
          f"   gain {abase - amae:+7.3f} min")


if __name__ == "__main__":
    main()
