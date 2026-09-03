# -*- coding: utf-8 -*-
"""
cris_pipeline/diagnose_model.py
===============================
Wiring and message-passing diagnostics.

The previous model reached production with message passing that did not
propagate delay (injecting +40 min into one train moved its conflict partner by
+0.24 min) and station embeddings that were indistinguishable between congested
and free (cosine similarity 0.973). Neither was caught before training.

Checks, in order:
  1. shapes / dtypes / parameter count / gradient reachability
  2. weight-init scale per layer group
  3. gradient accumulation through the TBPTT window
  4. embedding drift across the 4 conv layers (is message passing doing work?)
  5. node distinguishability (do congested and free stations differ?)
  6. causal propagation (does injecting delay into one train move its
     conflict partners?)
  7. input-path attribution (network vs journey history)

    python -m cris_pipeline.diagnose_model [--checkpoint ckpt.pt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cris_pipeline.config import TOPOLOGY_PATH, DATASET_DIR      # noqa: E402
from cris_pipeline.dataset_cris import CrisDataset               # noqa: E402
from cris_pipeline.model_cris import (                            # noqa: E402
    CorridorNextEventGNN, regression_loss, load_weights,
)


def _pick_busy(snaps, min_conf=6):
    """A snapshot with real conflict structure — trivial ones prove nothing."""
    best, best_n = None, -1
    for s in snaps:
        n = int(s.data["train", "conflicts", "train"].edge_index.shape[1])
        if n > best_n:
            best, best_n = s, n
        if n >= min_conf and s.data["train"].x.shape[0] >= 8:
            return s, n
    return best, best_n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-days", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(0)
    ds = CrisDataset(args.dataset, args.topology, max_days=args.max_days,
                     verbose=False)
    snaps = ds.get_split("train")
    print(f"snapshots: {len(snaps)}")

    model = CorridorNextEventGNN(
        station_in_dim=ds.station_feat_dim,
        train_in_dim=ds.train_feat_dim,
        section_in_dim=ds.section_feat_dim,
        hist_in_dim=ds.hist_feat_dim,
        num_services=ds.num_services,
        num_stations=ds.num_stations,
        num_sections=ds.num_sections,
    )
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        load_weights(model, ck["model"])
        print(f"loaded checkpoint (epoch {ck.get('epoch')}, "
              f"val_mae {ck.get('val_mae')})")
    model.eval()

    fails: list[str] = []

    # ── 1. Shapes & parameters ────────────────────────────────────────────────
    print("\n" + "=" * 78 + "\n1. SHAPES / PARAMETERS\n" + "=" * 78)
    n_par = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {n_par:,} ({n_tr:,} trainable)")

    snap, n_conf = _pick_busy(snaps)
    d = snap.data
    print(f"  probe snapshot: {d['train'].x.shape[0]} trains, "
          f"{n_conf} conflict edges")
    out = model(d, update_memory=False)
    T = d["train"].x.shape[0]
    for k, exp in (("next_arr", (T,)), ("next_dep", (T,))):
        got = tuple(out[k].shape)
        ok = got == exp
        print(f"  {k:24} {got}  {'ok' if ok else f'FAIL expected {exp}'}")
        if not ok:
            fails.append(f"{k} shape {got} != {exp}")

    v = out["next_arr"]
    finite = bool(torch.isfinite(v).all())
    print(f"  predictions finite: {finite}   "
          f"range [{float(v.min()):+.2f}, {float(v.max()):+.2f}] min")
    if not finite:
        fails.append("next_arr contains non-finite values")

    # ── 2. Weight-init scale ──────────────────────────────────────────────────
    print("\n" + "=" * 78 + "\n2. WEIGHT INIT\n" + "=" * 78)
    groups: dict[str, list] = {}
    for name, p in model.named_parameters():
        if p.dim() < 2:
            continue
        g = name.split(".")[0]
        groups.setdefault(g, []).append((name, p))
    for g, items in sorted(groups.items()):
        stds = [float(p.std()) for _, p in items]
        zeros = sum(1 for _, p in items if float(p.std()) < 1e-9)
        flag = ""
        if g in ("next_arr_head", "next_dep_head") and zeros:
            flag = "  (final layer zeroed on purpose -> starts at persistence)"
        print(f"  {g:26} n={len(items):2d}  std {min(stds):.4f}-{max(stds):.4f}"
              f"{flag}")
        for nm, p in items:
            s = float(p.std())
            if s < 1e-9 and not nm.startswith(("next_arr_head", "next_dep_head")):
                fails.append(f"{nm} initialised to all-zero")
            if s > 1.0:
                fails.append(f"{nm} init std {s:.2f} > 1.0")

    # ── 3. Gradient reachability & accumulation ───────────────────────────────
    print("\n" + "=" * 78 + "\n3. GRADIENTS\n" + "=" * 78)
    model.train()
    model.reset_memory()
    model.zero_grad()

    # Must be snapshots that actually CARRY SUPERVISION. Taking the first N
    # snapshots of the day selects pre-dawn ticks, and since each journey
    # prefix is now supervised only once, those can hold zero targets — the
    # loss then correctly returns 0 and every parameter reports "zero
    # gradient", which reads exactly like a dead network. That false alarm
    # cost a debugging cycle; select on the mask instead.
    steps = [s for s in snaps
             if s.data["train"].x.numel()
             and float(s.data["train"].target_mask[:, 0].sum()) > 0][:12]
    if not steps:
        fails.append("no snapshot in the probe set carries an h1 target — "
                     "cannot test gradient reachability")
        print("  SKIPPED: no supervised snapshots in probe set")
        steps = []
    n_sup = sum(float(s.data["train"].target_mask[:, 0].sum()) for s in steps)
    print(f"  probe set: {len(steps)} snapshots carrying {int(n_sup)} targets")
    total = None
    for s in steps:
        o = model(s.data, update_memory=True)
        m = s.data["train"].target_mask[:, 0]
        tgt = s.data["train"].target_delta[:, 0]
        # Both heads, exactly as the trainer drives them — otherwise the unused
        # head shows up as a false "no gradient" failure.
        dm = s.data["train"].y_next_dep_mask
        dt = s.data["train"].y_next_dep - s.data["train"].current_delay
        l = (regression_loss(o["next_arr"], tgt, m)
             + 0.5 * regression_loss(o["next_dep"], dt, dm))
        total = l if total is None else total + l
    if total is not None:
        (total / len(steps)).backward()

    no_grad, zero_grad = [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            no_grad.append(name)
        elif float(p.grad.abs().sum()) == 0.0:
            zero_grad.append(name)
    print(f"  accumulated over {len(steps)} snapshots (TBPTT window)")
    print(f"  params with NO grad   : {len(no_grad)}")
    print(f"  params with ZERO grad : {len(zero_grad)}")
    for n in no_grad[:8]:
        print(f"      no-grad: {n}")
    for n in zero_grad[:8]:
        print(f"      zero-grad: {n}")
    # station_critical_head is only supervised by an auxiliary loss not used here
    unexpected = [n for n in no_grad + zero_grad
                  if not n.startswith("station_critical_head")]
    if unexpected:
        fails.append(f"{len(unexpected)} params receive no gradient: "
                     f"{unexpected[:5]}")

    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
    print(f"  global grad norm      : {float(gn):.4f}")
    if not (0 < float(gn) < 1e4):
        fails.append(f"grad norm {float(gn)} out of sane range")

    hist_grad = sum(float(p.grad.abs().sum())
                    for n, p in model.named_parameters()
                    if n.startswith("history_encoder") and p.grad is not None)
    gnn_grad = sum(float(p.grad.abs().sum())
                   for n, p in model.named_parameters()
                   if n.startswith("convs") and p.grad is not None)
    print(f"  grad mass history_encoder : {hist_grad:.4f}")
    print(f"  grad mass conv layers     : {gnn_grad:.4f}")
    if hist_grad == 0:
        fails.append("history encoder receives no gradient")
    if gnn_grad == 0:
        fails.append("conv layers receive no gradient")

    # ── 4. Embedding drift through the conv stack ─────────────────────────────
    print("\n" + "=" * 78 + "\n4. MESSAGE PASSING — EMBEDDING DRIFT\n" + "=" * 78)
    model.eval()
    with torch.no_grad():
        hist = model.history_encoder(d["train"].history, d["train"].history_len)
        h_s0, h_t0, h_sec0 = model._encode(d, hist)
        h_s4, h_t4, h_sec4 = model._gnn(h_s0, h_t0, h_sec0, d)
    for nm, a, b in (("train", h_t0, h_t4), ("station", h_s0, h_s4),
                     ("section", h_sec0, h_sec4)):
        ratio = float(b.norm() / a.norm().clamp(min=1e-9))
        cos = float(F.cosine_similarity(a, b, dim=-1).mean())
        print(f"  {nm:8} |h4|/|h0| = {ratio:5.2f}   mean cos(h0,h4) = {cos:5.3f}")
        if abs(ratio - 1.0) < 0.02 and cos > 0.999:
            fails.append(f"{nm}: conv stack is a no-op (residual dominated)")

    # ── 5. Node distinguishability ────────────────────────────────────────────
    print("\n" + "=" * 78 + "\n5. DISTINGUISHABILITY\n" + "=" * 78)
    occ = d["station"].x[:, 1]                       # capacity_occupancy
    busy = (occ > occ.median()).nonzero().flatten()
    free = (occ <= occ.median()).nonzero().flatten()
    if len(busy) and len(free):
        cb = h_s4[busy].mean(0)
        cf = h_s4[free].mean(0)
        sim = float(F.cosine_similarity(cb, cf, dim=0))
        print(f"  busy vs free station cos = {sim:.4f}   "
              f"(old model: 0.973 — congestion invisible)")
        if sim > 0.995:
            fails.append(f"stations indistinguishable by occupancy (cos {sim:.4f})")

    dl = d["train"].x[:, 0]
    late = (dl > dl.median()).nonzero().flatten()
    early = (dl <= dl.median()).nonzero().flatten()
    if len(late) and len(early):
        sim = float(F.cosine_similarity(h_t4[late].mean(0),
                                        h_t4[early].mean(0), dim=0))
        print(f"  late vs on-time train cos = {sim:.4f}")
        if sim > 0.999:
            fails.append("trains indistinguishable by delay")

    # ── 6. Causal propagation ─────────────────────────────────────────────────
    print("\n" + "=" * 78 + "\n6. CAUSAL PROPAGATION\n" + "=" * 78)
    ei = d["train", "conflicts", "train"].edge_index
    if ei.shape[1] == 0:
        print("  no conflict edges in probe snapshot — skipped")
    else:
        with torch.no_grad():
            base = model(d, update_memory=False)["next_arr"].clone()
        src = int(ei[0, 0])
        partners = ei[1][ei[0] == src].unique()
        others = torch.tensor([i for i in range(T)
                               if i != src and i not in set(partners.tolist())])

        pert = d.clone()
        # +40 min on the injected train, in the same signed-log scale as dim 0
        import math
        pert["train"].x[src, 0] = math.copysign(
            math.log1p(abs(40.0)), 1.0) / math.log1p(240.0)
        with torch.no_grad():
            new = model(pert, update_memory=False)["next_arr"]

        d_partner = float((new[partners] - base[partners]).abs().mean())
        d_other = (float((new[others] - base[others]).abs().mean())
                   if len(others) else 0.0)
        print(f"  injected +40 min into train {src} "
              f"({len(partners)} conflict partners)")
        print(f"  mean |delta| conflict partners : {d_partner:.3f} min")
        print(f"  mean |delta| unrelated trains  : {d_other:.3f} min")
        print(f"  (old model: 0.24 min on partners — propagation was broken)")
        if d_partner < 1e-6:
            fails.append("no causal propagation across conflict edges")

    # ── 7. Input-path attribution ─────────────────────────────────────────────
    print("\n" + "=" * 78 + "\n7. INPUT-PATH ATTRIBUTION\n" + "=" * 78)
    with torch.no_grad():
        full = model(d, update_memory=False)["next_arr"]
        no_h = model(d, update_memory=False, ablate_history=True)["next_arr"]
        no_n = model(d, update_memory=False, ablate_network=True)["next_arr"]
    print(f"  removing journey history shifts p50 by {float((full - no_h).abs().mean()):.3f} min")
    print(f"  removing GNN network state shifts p50 by {float((full - no_n).abs().mean()):.3f} min")
    print("  (both should be non-zero after training; on an untrained model the")
    print("   heads start at zero so shifts are small by construction)")

    print("\n" + "=" * 78)
    if fails:
        print(f"DIAGNOSTICS FAILED — {len(fails)} problem(s):")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("DIAGNOSTICS PASSED")


if __name__ == "__main__":
    main()
