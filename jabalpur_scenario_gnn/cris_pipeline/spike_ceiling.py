# -*- coding: utf-8 -*-
"""
cris_pipeline/spike_ceiling.py
==============================
Is the spike signal IN the data, or is the model failing to use it?

Every discussion of "why do we miss spikes" stalls on this fork, so measure it.

  unit    : one (snapshot, train) pair that has supervised targets
  label   : this train's delay rises more than SPIKE_MIN over its remaining
            horizon, relative to its delay right now
  features: the numeric fields of that train's own node -- the same values the
            GNN's train vector is built from

Then fit a gradient-boosted tree on those features.

IMPORTANT about what this bounds. The tree sees STRICTLY LESS than the GNN: no
neighbouring trains, no station or section nodes, no message passing, no
journey history, no memory. So it is a LOWER bound on extractable signal, not
an upper one. Read it accordingly:

  tree AUC ~ 0.5      -> not even the train's own state predicts a spike.
  tree AUC high, GNN  -> the signal is there and reachable WITHOUT the graph.
    not beating it       If the GNN with far more information only matches a
                         context-free tree, its extra structure is not paying
                         for itself on this task.
  GNN >> tree         -> the graph is doing real work.

For MAGNITUDE the logic is different and stronger. If the tree cannot beat
"always guess the median" (R^2 <= 0), and the GNN also lands at the median
baseline, then two models with very different information both fail. That is
good evidence the quantity is genuinely unpredictable rather than merely hard,
and it means the systematic under-prediction is CORRECT behaviour: with no
signal about size, minimising absolute error on a right-skewed target is
exactly what a well-fit model should do.

Splitting is by train instance, never at random: snapshots of one train three
minutes apart are near-duplicate rows sharing a label, so a random split
measures memorisation. Random splitting here reported AUC 0.952; grouped by
journey it is 0.864.

Usage:
    python -m cris_pipeline.spike_ceiling
    python -m cris_pipeline.spike_ceiling --split test --spike-min 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import DATASET_DIR, TOPOLOGY_PATH
from .dataset_cris import CrisDataset


def build_table(ds, split: str, spike_min: float):
    """Return (X, y, feature_names) over every supervised train node."""
    X, y = [], []
    names = None

    for day in ds.days(split) if hasattr(ds, "days") else []:
        pass  # placeholder; real iteration below

    return X, y, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--split", default="test")
    ap.add_argument("--spike-min", type=float, default=30.0)
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, precision_recall_curve
    from sklearn.model_selection import train_test_split

    from .audit_features import TRAIN_LABELS, STATION_LABELS

    ds = CrisDataset(args.dataset, args.topology, lazy=True, verbose=False)

    rows, labels, groups, rises = [], [], [], []
    meta = json.loads((Path(args.dataset) / "meta.json").read_text("utf-8"))

    n_days = 0
    for fp in sorted(Path(args.dataset).glob("2025-*.json")):
        blob = json.loads(fp.read_text(encoding="utf-8"))
        if args.split != "all" and blob.get("split") != args.split:
            continue
        n_days += 1
        for snap in blob["snapshots"]:
            for n in snap["train_nodes"]:
                mask = n.get("delay_target_mask") or []
                tgts = n.get("delay_targets") or []
                valid = [t for t, m in zip(tgts, mask) if m]
                if len(valid) < 3:
                    continue
                now = float(n.get("delay_min", 0.0))
                rise = max(valid) - now
                labels.append(1 if rise > args.spike_min else 0)
                rows.append(n)
                groups.append(n["instance_id"])
                rises.append(rise)

    if not rows:
        raise SystemExit("no supervised train nodes found for that split")

    # Feature block: the numeric fields of the train node, which is what the
    # GNN's train vector is built from, plus its immediate spatial context.
    # Everything named *target* is supervision, not input. Missing two of them
    # the first time this ran gave AUC 0.954 -- the tree was reading the answer
    # off `next_dep_target_min`. Same failure mode as the four feature leaks
    # found in build_dataset: a value that exists in the row but is not
    # knowable at decision time.
    LEAKS = {"delay_targets", "delay_target_mask", "next_delay_target_min",
             "next_dep_target_min", "next_dep_target_mask",
             # not an answer leak, but not a GNN input either -- it is the
             # supervision flag. Keeping it would let the tree use something
             # the model never sees.
             "target_eligible"}
    keys = [k for k, v in rows[0].items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k not in LEAKS]
    assert not any("target" in k for k in keys), \
        f"target field still in features: {[k for k in keys if 'target' in k]}"
    X = np.array([[float(r.get(k) or 0.0) for k in keys] for r in rows],
                 dtype=np.float32)
    y = np.array(labels, dtype=np.int8)

    print(f"days ({args.split})   : {n_days}")
    print(f"units            : {len(y):,}")
    print(f"spikes (>{args.spike_min:.0f} min): {int(y.sum()):,}  "
          f"({y.mean():.2%})")
    print(f"features         : {len(keys)}")
    print()

    # Single-feature discrimination first: which raw signals separate at all?
    aucs = []
    for j, k in enumerate(keys):
        col = X[:, j]
        if np.std(col) < 1e-9:
            continue
        try:
            a = roc_auc_score(y, col)
        except ValueError:
            continue
        aucs.append((max(a, 1 - a), k, a))
    aucs.sort(reverse=True)
    print("SINGLE FEATURE — how well does one number alone separate a spike?")
    print(f"  {'feature':<28} {'AUC':>6}")
    print("  " + "-" * 36)
    for adj, k, a in aucs[:12]:
        print(f"  {k:<28} {adj:6.3f}")
    print()

    # Joint model. Split by TRAIN INSTANCE, never randomly: snapshots of the
    # same train three minutes apart are near-identical rows carrying the same
    # label, so a random split puts near-duplicates on both sides and the AUC
    # measures memorisation. Grouping by instance_id forces the tree to
    # generalise to journeys it has never seen.
    from sklearn.model_selection import GroupShuffleSplit
    g = np.array(groups)
    tr_idx, te_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.4, random_state=0).split(X, y, groups=g))
    Xtr, Xte, ytr, yte = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
    print(f"grouped split    : {len(set(g[tr_idx]))} train journeys / "
          f"{len(set(g[te_idx]))} held-out journeys "
          f"(overlap {len(set(g[tr_idx]) & set(g[te_idx]))})")
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, random_state=0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)

    print("JOINT MODEL — gradient boosting on the same information")
    print(f"  AUC                    : {auc:.3f}")
    prec, rec, thr = precision_recall_curve(yte, p)
    for want in (0.50, 0.70, 0.80, 0.90):
        idx = np.where(rec >= want)[0]
        if len(idx):
            i = idx[-1]
            print(f"  precision @ {want:.0%} recall : {prec[i]:.3f}")
    # ---- Magnitude, which is a different question from detection ----------
    # Knowing a spike is coming and knowing how big it will be are separate
    # skills, and our model is much worse at the second (90% of its spike
    # errors are under-predictions). Restrict to the spikes and ask whether
    # the SIZE of the rise is predictable at all from the same features.
    from sklearn.ensemble import HistGradientBoostingRegressor
    rise_all = np.array(rises, dtype=np.float32)
    sp_tr = tr_idx[y[tr_idx] == 1]
    sp_te = te_idx[y[te_idx] == 1]
    print()
    print("MAGNITUDE — given a spike happens, how big will it be?")
    print(f"  spike units      : {len(sp_tr):,} train / {len(sp_te):,} held out")
    if len(sp_te) > 40:
        reg = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=6, random_state=0)
        reg.fit(X[sp_tr], rise_all[sp_tr])
        pm = reg.predict(X[sp_te])
        truth = rise_all[sp_te]
        mae = float(np.mean(np.abs(pm - truth)))
        base = float(np.mean(np.abs(truth - np.median(rise_all[sp_tr]))))
        ss = 1.0 - float(np.sum((truth - pm) ** 2)
                         / max(np.sum((truth - truth.mean()) ** 2), 1e-9))
        print(f"  mean actual rise : {truth.mean():.1f} min")
        print(f"  predict-the-median MAE : {base:.1f} min")
        print(f"  tree MAE               : {mae:.1f} min   "
              f"(R^2 {ss:+.3f})")
        print(f"  our GNN spike MAE      : 22.3 min  (rollout, original types)")

    print()
    print("READING IT")
    print("  ~0.50            signal absent; no architecture fixes this")
    print("  0.60-0.75        weak signal, partially learnable")
    print("  >0.80            strong signal — if the GNN is not matching this,")
    print("                   the gap is ours, not the data's")


if __name__ == "__main__":
    main()
