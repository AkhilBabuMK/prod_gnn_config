# -*- coding: utf-8 -*-
"""
cris_pipeline/ablate_edges.py
=============================
Which parts of the graph actually earn their place?

Removes one edge type at a time on the held-out split and reports the change
in MAE.

    dMAE          how much worse the model gets without this edge type
    mean|shift|   how much the predictions move at all

A type with a large shift but ~zero dMAE is doing work that does not help.

HISTORY, because the earlier conclusion here was wrong and worth not
repeating. This file used to state that the conflict GAT "carries no signal",
citing three measurements putting its contribution near zero (dMAE +0.043).
Every one of those was taken while the conflict features were scoped to the 4
junction stations -- which happen to have the LOWEST precedence rates on the
corridor (JBP 47.3%, KTE 40.6%) against non-junctions like SBD at 83.0%. The
measurement was real; what it measured was not the thing being claimed.

After rescoping conflicts to all 22 stations, the same ablation on v2 gives
dMAE +0.292 for ('train','conflicts','train') -- roughly 10% of MAE, and 7x
the old figure. No edge type is currently a deletion candidate; the cheapest
still costs +7.8% and removing ('section','carries','train') doubles MAE.

The lesson generalises: an ablation only bounds the value of the STRUCTURE as
currently fed. If the features flowing along an edge are impoverished, the
ablation measures the features, not the edge.

    python -m cris_pipeline.ablate_edges --checkpoint data_cris/corridor_nextevent.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cris_pipeline.config import TOPOLOGY_PATH, DATASET_DIR, OUT_DIR   # noqa: E402
from cris_pipeline.dataset_cris import CrisDataset                     # noqa: E402
from cris_pipeline.model_cris import (                            # noqa: E402
    CorridorNextEventGNN, load_weights,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUT_DIR / "corridor_nextevent.pt"))
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CorridorNextEventGNN(**ck["config"])
    load_weights(model, ck["model"])
    model.eval()
    print(f"checkpoint epoch {ck['epoch']}  val_mae {ck['val_mae']:.2f}")

    ds = CrisDataset(args.dataset, args.topology, verbose=False)
    snaps = [s for s in ds.get_split(args.split) if s.data["train"].x.numel()]
    print(f"{len(snaps)} {args.split} snapshots\n")

    etypes = list(snaps[0].data.edge_types)
    acc = {et: {"shift": 0.0, "err": 0.0} for et in etypes}
    base_err = 0.0
    n_ev = 0
    prev_day = None

    with torch.no_grad():
        for s in snaps:
            if s.day_index != prev_day:
                model.reset_memory()
                prev_day = s.day_index
            d = s.data
            mk = d["train"].target_mask[:, 0] > 0.5
            tgt = d["train"].target_delta[:, 0]
            base = model(d, update_memory=True)["next_arr"]
            if not mk.any():
                continue
            base_err += float((base[mk] - tgt[mk]).abs().sum())
            n_ev += int(mk.sum())

            for et in etypes:
                if d[et].edge_index.shape[1] == 0:
                    continue
                pert = d.clone()
                pert[et].edge_index = torch.zeros((2, 0), dtype=torch.long)
                ea = getattr(pert[et], "edge_attr", None)
                if ea is not None:
                    pert[et].edge_attr = ea.new_zeros((0, ea.shape[1]))
                try:
                    out = model(pert, update_memory=False)["next_arr"]
                except Exception:
                    continue
                acc[et]["shift"] += float((out[mk] - base[mk]).abs().sum())
                acc[et]["err"] += float((out[mk] - tgt[mk]).abs().sum())

    base_mae = base_err / max(n_ev, 1)
    print(f"baseline {args.split} MAE (h1): {base_mae:.3f}   n={n_ev:,}\n")
    print(f"{'ablated edge type':<44} {'mean|shift|':>12} {'MAE':>8} {'dMAE':>8}")
    print("-" * 76)
    rows = []
    for et in etypes:
        a = acc[et]
        if a["shift"] == 0.0 and a["err"] == 0.0:
            continue
        mae = a["err"] / max(n_ev, 1)
        rows.append((mae - base_mae, a["shift"] / max(n_ev, 1), mae, et))
    for dm, sh, mae, et in sorted(rows, reverse=True):
        note = ""
        if dm < 0.02:
            note = "   <- earns nothing"
        elif dm < 0.05:
            note = "   <- marginal"
        print(f"{str(et):<44} {sh:>12.4f} {mae:>8.3f} {dm:>+8.3f}{note}")

    print("\nA type whose removal barely changes MAE is a candidate for "
          "deletion:\n  it costs parameters, compute and complexity for no "
          "measured accuracy.")


if __name__ == "__main__":
    main()
