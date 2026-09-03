# -*- coding: utf-8 -*-
"""
cris_pipeline/eval_cris.py
==========================
Honest evaluation on the held-out test split.

Every number is reported against PERSISTENCE on the identical set of events.
The previous generation's headline `val_mae 7.79` looked strong while it beat
persistence by 0.7-3.0 min; that gap is the only thing that matters.

Reports:
  1. Next-event MAE vs persistence, bucketed by lead time to the event
  2. One-step prediction scored against hop-h targets (NOT a rollout —
     teacher-forced throughout; rollout_cris.py is the honest multi-hop test)
  3. Spike detection: recall / precision / magnitude error
  4. Prediction bias: is the model systematically optimistic?
  5. Input-path ablation: how much comes from the network vs journey history

    python -m cris_pipeline.eval_cris --checkpoint data_cris/corridor_nextevent.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cris_pipeline.config import TOPOLOGY_PATH, DATASET_DIR       # noqa: E402
from cris_pipeline.dataset_cris import CrisDataset                # noqa: E402
from cris_pipeline.model_cris import (                            # noqa: E402
    CorridorNextEventGNN, load_weights, check_feature_dims,
)

SPIKE = 30.0
LEAD_BUCKETS = [(0, 15), (15, 30), (30, 60), (60, 120), (120, 1e9)]


def _bucket(v: float) -> str:
    for lo, hi in LEAD_BUCKETS:
        if lo <= v < hi:
            return f"{lo}-{hi if hi < 1e9 else '+'}m"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ds = CrisDataset(args.dataset, args.topology, verbose=False)
    snaps = ds.get_split(args.split)
    print(f"{len(snaps)} {args.split} snapshots")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    check_feature_dims(cfg, ds)
    model = CorridorNextEventGNN(**cfg)
    load_weights(model, ck["model"])
    model.eval()
    print(f"checkpoint epoch {ck['epoch']}  val_mae {ck['val_mae']:.2f}\n")

    lead = defaultdict(lambda: {"e": 0.0, "p": 0.0, "n": 0})
    hop = defaultdict(lambda: {"e": 0.0, "p": 0.0, "n": 0})
    tp = fp = fn = 0
    sp_err = 0.0
    tot = 0
    bias_sum = 0.0
    abl_h = abl_n = abl_n_ev = 0.0

    prev_day = None
    with torch.no_grad():
        for s in snaps:
            if s.day_index != prev_day:
                model.reset_memory()
                prev_day = s.day_index
            d = s.data
            if d["train"].x.numel() == 0:
                continue

            out = model(d, update_memory=True)
            p50 = out["next_arr"]        # single regression output, in minutes

            mask = d["train"].target_mask[:, 0] > 0.5
            tgt = d["train"].target_delta[:, 0]
            if mask.any():
                t = tgt[mask]
                pred = p50[mask]
                # eta_next is dim 2, log-normalised over a 90-min clamp
                eta_n = d["train"].x[mask, 2]
                eta = torch.expm1(eta_n * torch.log1p(torch.tensor(90.0)))

                err = (pred - t).abs()
                per = t.abs()
                for i in range(len(t)):
                    b = _bucket(float(eta[i]))
                    lead[b]["e"] += float(err[i])
                    lead[b]["p"] += float(per[i])
                    lead[b]["n"] += 1

                is_sp = t > SPIKE
                pr_sp = pred > SPIKE
                tp += int((is_sp & pr_sp).sum())
                fn += int((is_sp & ~pr_sp).sum())
                fp += int((~is_sp & pr_sp).sum())
                if is_sp.any():
                    sp_err += float((pred[is_sp] - t[is_sp]).abs().sum())

                tot += int(len(t))
                bias_sum += float((pred - t).sum())

                oh = model(d, update_memory=False,
                           ablate_history=True)["next_arr"][mask]
                on = model(d, update_memory=False,
                           ablate_network=True)["next_arr"][mask]
                abl_h += float((pred - oh).abs().sum())
                abl_n += float((pred - on).abs().sum())
                abl_n_ev += len(t)

            # multi-hop: how far does the delay chain stay accurate?
            for h in range(d["train"].target_delta.shape[1]):
                mh = d["train"].target_mask[:, h] > 0.5
                if not mh.any():
                    continue
                th = d["train"].target_delta[mh, h]
                # p50 is a next-event head; for hop h>0 the honest comparison
                # is against persistence on that same event.
                hop[h]["e"] += float((p50[mh] - th).abs().sum())
                hop[h]["p"] += float(th.abs().sum())
                hop[h]["n"] += int(mh.sum())

    print("=" * 78)
    print("1. NEXT-EVENT MAE BY LEAD TIME  (p50 vs persistence)")
    print("=" * 78)
    print(f"{'bucket':>10} {'n':>7} {'MAE':>8} {'persist':>9} {'gain':>8}")
    for lo, hi in LEAD_BUCKETS:
        b = f"{lo}-{hi if hi < 1e9 else '+'}m"
        v = lead.get(b)
        if not v or not v["n"]:
            continue
        m, p = v["e"] / v["n"], v["p"] / v["n"]
        print(f"{b:>10} {v['n']:>7} {m:>8.2f} {p:>9.2f} {p - m:>+8.2f}")

    print("\n" + "=" * 78)
    print("2. ONE-STEP PREDICTION SCORED AGAINST HOP-h TARGETS")
    print("   NOT a rollout and NOT evidence of propagation — every input here")
    print("   is still ground truth. Use rollout_cris.py for the real thing.")
    print("=" * 78)
    print(f"{'hop':>5} {'n':>7} {'MAE':>8} {'persist':>9} {'gain':>8}")
    for h in sorted(hop):
        v = hop[h]
        if not v["n"]:
            continue
        m, p = v["e"] / v["n"], v["p"] / v["n"]
        print(f"{h + 1:>5} {v['n']:>7} {m:>8.2f} {p:>9.2f} {p - m:>+8.2f}")

    print("\n" + "=" * 78)
    print("3. SPIKE DETECTION  (target delta > 30 min)")
    print("=" * 78)
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    print(f"  spikes in test  : {tp + fn}")
    print(f"  recall          : {rec:.1%}   (old system: 13-30%)")
    print(f"  precision       : {prec:.1%}")
    print(f"  spike MAE       : {sp_err / max(tp + fn, 1):.1f} min")

    print("\n" + "=" * 78)
    print("4. PREDICTION BIAS  (mean signed error — the rollout killer)")
    print("=" * 78)
    print(f"  mean (pred - actual): {bias_sum / max(tot, 1):+.2f} min over "
          f"{tot:,} events")
    print("  A negative value means the model is systematically optimistic. It "
          "compounds\n  in rollout, so this matters far more than MAE alone.")

    print("\n" + "=" * 78)
    print("5. INPUT-PATH ABLATION  (mean |shift| in predicted minutes)")
    print("=" * 78)
    print(f"  remove journey history : {abl_h / max(abl_n_ev, 1):.2f} min")
    print(f"  remove GNN network     : {abl_n / max(abl_n_ev, 1):.2f} min")
    print("  Both must be clearly non-zero: if the network term is ~0 the model")
    print("  has collapsed to per-train persistence, which is the failure mode")
    print("  that the previous generation exhibited.")


if __name__ == "__main__":
    main()
