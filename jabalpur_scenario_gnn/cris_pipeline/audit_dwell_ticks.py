# -*- coding: utf-8 -*-
"""
cris_pipeline/audit_dwell_ticks.py
==================================
What does the trainer actually see while a train STANDS at a station?

THE QUESTION

Snapshots are built every 3 minutes. A train held at a platform therefore
appears in many consecutive snapshots, and each one emits a supervised event.
Two things follow that nobody has checked:

  1. Is the TARGET the same at every one of those ticks? If a train arrives at
     stop k at +10 and eventually reaches k+1 at +45, then `current_delay` is
     pinned at +10 for the whole stand and the target is +35 every time. The
     same answer, repeated, with only `time_at_stop` and the neighbourhood
     changing underneath it.

  2. If so, a long detention contributes MANY examples and a clean pass-through
     contributes one or two. The training distribution is then weighted by
     dwell length, and the later ticks of a stand are far easier than the first
     -- by then the evidence that the train is stuck is overwhelming.

     If the model's error collapses on those late ticks, then a good val_mae
     may largely be the model getting easy repeats right, while the tick that
     actually matters operationally -- the FIRST one, where a controller could
     still act -- stays hard.

WHAT THIS REPORTS

  * events per train-stop, i.e. the duplication factor
  * whether the target is constant across a stand (it should be, by
    construction -- this verifies it rather than assuming it)
  * one-step error bucketed by position within the stand, which is the number
    that says whether the model is learning the mechanism or the repeat
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
    print(f"checkpoint epoch {ck.get('epoch')}  split={args.split}\n")

    # (iid, stop_idx) -> ordered list of (target, prediction, in_transit)
    runs: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)

    with torch.no_grad():
        for day in ds.iter_days(args.split):
            model.reset_memory()
            for snap in day:
                d = snap.data
                if d["train"].x.numel() == 0:
                    continue
                out = model(d, update_memory=True)
                m = d["train"].target_mask[:, 0] > 0.5
                if not m.any():
                    continue
                tgt = d["train"].target_delta[:, 0]
                pred = out["next_arr"]
                iids = d["train"].iid
                sidx = d["train"].stop_idx
                # dim 10 of the train feature block is `in_transit`
                intr = d["train"].x[:, 10]
                for i in range(len(iids)):
                    if not bool(m[i]):
                        continue
                    runs[(iids[i], int(sidx[i]))].append(
                        (float(tgt[i]), float(pred[i]), float(intr[i])))

    # ── duplication ──────────────────────────────────────────────────────────
    lens = [len(v) for v in runs.values()]
    lens.sort()
    n_ev = sum(lens)
    print("=" * 74)
    print(f"DUPLICATION — {n_ev:,} supervised events from {len(runs):,} train-stops")
    print("=" * 74)
    print(f"  events per train-stop: median {lens[len(lens)//2]}"
          f"   p90 {lens[int(len(lens)*.9)]}   max {lens[-1]}")
    print(f"  mean {n_ev/len(lens):.2f}  <- every train-stop is supervised this "
          f"many times, on the SAME target")

    hist = defaultdict(int)
    for L in lens:
        b = "1" if L == 1 else "2-3" if L <= 3 else "4-7" if L <= 7 else \
            "8-15" if L <= 15 else "16+"
        hist[b] += 1
    print(f"\n  {'ticks at this stop':22}{'train-stops':>13}{'share of events':>17}")
    for b in ("1", "2-3", "4-7", "8-15", "16+"):
        if not hist[b]:
            continue
        ev = sum(L for L in lens if
                 (L == 1 if b == "1" else
                  2 <= L <= 3 if b == "2-3" else
                  4 <= L <= 7 if b == "4-7" else
                  8 <= L <= 15 if b == "8-15" else L >= 16))
        print(f"  {b:22}{hist[b]:13,}{ev/n_ev*100:16.1f}%")

    # ── is the target constant across a stand? ───────────────────────────────
    spreads = [max(t for t, _, _ in v) - min(t for t, _, _ in v)
               for v in runs.values() if len(v) > 1]
    const = sum(1 for s in spreads if abs(s) < 1e-6)
    print(f"\n  multi-tick train-stops: {len(spreads):,}")
    print(f"  target identical across every tick: {const:,}"
          f"  ({const/max(len(spreads),1)*100:.1f}%)")
    if spreads:
        print(f"  max spread within one train-stop: {max(spreads):.2f} min")

    # ── error by position within the stand ───────────────────────────────────
    print("\n" + "=" * 74)
    print("ERROR BY POSITION WITHIN THE STAND  (only train-stops with 4+ ticks)")
    print("=" * 74)
    by_pos = defaultdict(list)
    by_pos_last = defaultdict(list)
    for v in runs.values():
        if len(v) < 4:
            continue
        for j, (t, p, _) in enumerate(v):
            by_pos[min(j, 9)].append(abs(p - t))
            by_pos_last[min(len(v) - 1 - j, 9)].append(abs(p - t))

    print(f"  {'tick since arriving':22}{'n':>8}{'MAE':>9}")
    for j in range(10):
        if not by_pos[j]:
            continue
        lbl = f"{j*3}-{j*3+3} min" if j < 9 else "27+ min"
        print(f"  {lbl:22}{len(by_pos[j]):8,}{sum(by_pos[j])/len(by_pos[j]):9.3f}")

    print(f"\n  {'ticks before leaving':22}{'n':>8}{'MAE':>9}")
    for j in range(10):
        if not by_pos_last[j]:
            continue
        lbl = "last tick" if j == 0 else f"{j} tick(s) before"
        print(f"  {lbl:22}{len(by_pos_last[j]):8,}"
              f"{sum(by_pos_last[j])/len(by_pos_last[j]):9.3f}")

    # ── the operationally important comparison ───────────────────────────────
    first = [abs(v[0][1] - v[0][0]) for v in runs.values() if len(v) >= 4]
    rest = [abs(p - t) for v in runs.values() if len(v) >= 4
            for t, p, _ in v[1:]]
    allev = [abs(p - t) for v in runs.values() for t, p, _ in v]
    print("\n" + "=" * 74)
    print("THE COMPARISON THAT MATTERS")
    print("=" * 74)
    print(f"  every supervised event        n={len(allev):6,}   MAE "
          f"{sum(allev)/len(allev):6.3f}   <- this is val_mae")
    print(f"  FIRST tick of a long stand    n={len(first):6,}   MAE "
          f"{sum(first)/max(len(first),1):6.3f}   <- when a controller can act")
    print(f"  every later tick of one       n={len(rest):6,}   MAE "
          f"{sum(rest)/max(len(rest),1):6.3f}   <- the repeats")


if __name__ == "__main__":
    main()
