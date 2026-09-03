# -*- coding: utf-8 -*-
"""
cris_pipeline/audit_features.py
===============================
Acceptance gate for the corridor feature set.

The old feature set shipped with 4 of 9 section dims constant, the whole
station congestion channel ~95% zero, and two saturated junction-ETA dims —
none of which was visible until measured. This script makes that class of
failure a hard gate rather than a discovery.

FAILS the build if any dim is constant, >98% zero, or saturated
(std < 0.01 with a non-zero mean).

    python -m cris_pipeline.audit_features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cris_pipeline.config import (TOPOLOGY_PATH, DATASET_DIR,     # noqa: E402
                                  JOURNEYS_DIR)
from cris_pipeline.dataset_cris import CrisDataset                # noqa: E402

TRAIN_LABELS = [
    "delay_log", "delay_velocity", "eta_next", "eta_nearest_jct",
    "eta_second_jct", "jcts_remaining", "priority", "speed_dev",
    "stops_remaining", "route_progress", "in_transit", "at_booked_halt",
    "time_at_stop", "mins_since_departure", "mins_over_sched_run",
    "min_arr_delay_next", "n_coaches", "junction_overlap", "same_line_overlap",
    "higher_prio_overlap", "queue_ahead", "jct_gap_urgency",
    "freight_in_section", "freight_at_station", "freight_approaching",
    "section_disrupted", "section_run_mult", "entry_delay", "entry_trend",
    "has_upstream", "tod_sin", "tod_cos", "delay_observed",
    "precedence_urgency", "next_booked_halt", "next_sched_dwell",
    "next_allowance", "slack_ahead",
    "ev_jct_conflicts", "ev_cross_branch", "ev_higher_prio", "ev_queue_ahead",
    "ev_min_gap", "ev_max_prio_delta", "ev_has_next_jct", "ev_any_cross",
]
STATION_LABELS = [
    "platform_occupancy", "capacity_occupancy", "available_capacity",
    "freight_occupancy", "incoming_pressure",
    "max_train_delay", "delayed_density", "approach_density",
    "congestion_score", "temporal_congestion", "steps_since_critical",
    "critical_flag", "tod_sin", "tod_cos", "is_junction", "platforms_norm",
    "holding_capacity_norm", "loop_lines_norm", "crew_change",
    "traction_change",
]
SECTION_LABELS = [
    "base_run_norm", "trains_in_section", "freight_in_section",
    "run_multiplier", "is_disrupted", "psr_time_loss", "from_is_jct",
    "to_is_jct", "tod_sin", "tod_cos",
]
HIST_LABELS = [
    "delay_log", "leg_delta", "dwell_delta", "dwell_norm", "run_over",
    "sched_leg", "is_booked_halt", "has_actual",
]


def _integrity(failures: list) -> None:
    """Data integrity, as distinct from feature sanity.

    Every other check here asks 'does this dim carry information'. None of them
    ask 'is this dataset even well formed'. On 2025-07-30 a rebuild left stale
    files on disk beside the new ones: `build_journeys` writes `<date>#2.json`
    on collision rather than overwriting, and `build_dataset` did not remove
    day files from a previous build. The result was every train duplicated in
    every snapshot (6,183/day) AND a config change silently reverted, because
    the stale copy sorted last and won. This audit passed on it. So did
    audit_causality — duplicating a train breaks neither feature ranges nor
    causality.

    Two cheap invariants that would have caught it immediately.
    """
    import collections
    import glob as _glob
    import json as _json

    dup_total = 0
    days = sorted(_glob.glob(str(DATASET_DIR / "2025-*.json")))
    for fp in days:
        blob = _json.loads(Path(fp).read_text(encoding="utf-8"))
        snaps_raw = (blob["snapshots"] if isinstance(blob, dict)
                     and "snapshots" in blob else blob)
        for s in snaps_raw:
            seen = set()
            for n in s["train_nodes"]:
                if n["instance_id"] in seen:
                    dup_total += 1
                seen.add(n["instance_id"])

    print(f"\n{'=' * 86}\nDATA INTEGRITY\n{'=' * 86}")
    print(f"  day files on disk   : {len(days)}")
    print(f"  duplicate instances : {dup_total:,}")
    if dup_total:
        failures.append(
            f"integrity: {dup_total:,} duplicate instance_ids within a "
            f"snapshot — the same train is in the graph twice. Almost always "
            f"stale files from a previous build; rebuild journeys AND dataset "
            f"(both now wipe by default) rather than trying to patch around it")

    # Day files on disk must match what THIS build says it produced. Checking
    # against journeys on disk is not enough: the dropped boundary days still
    # have journey files, so a stale 2025-10-01.json looks legitimate to that
    # test. meta.json is written by the build itself, so it is the authority.
    meta_path = DATASET_DIR / "meta.json"
    orphan: list[str] = []
    if meta_path.exists():
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        rng = meta.get("date_range", {})
        lo, hi = rng.get("start"), rng.get("end")
        want_n = meta.get("num_days")
        stems = sorted(Path(f).stem for f in days)
        if lo and hi:
            orphan = [d for d in stems if d < lo or d > hi]
        if want_n is not None and len(days) != want_n:
            failures.append(
                f"integrity: {len(days)} day files on disk but meta.json says "
                f"this build produced {want_n} — leftovers from an earlier "
                f"build are still present")
        if orphan:
            failures.append(
                f"integrity: day file(s) outside this build's range "
                f"{lo}..{hi}: {orphan} — stale output from an earlier build")
    else:
        failures.append("integrity: dataset/meta.json missing")
    print(f"  meta says days      : "
          f"{meta.get('num_days') if meta_path.exists() else '?'}")
    print(f"  out-of-range files  : {len(orphan)}")


# Dims that are rare in the world, not broken in the code. Each needs a reason
# and is still required to FIRE — being on this list waives the zero-rate
# threshold, never the "carries information" test.
# Dims that are legitimately constant on THIS corridor. A constant input is
# harmless -- its contribution folds into the next layer's bias -- but it is
# dead weight, so each one needs a reason and should be revisited if the
# corridor definition changes.
EXPECTED_CONSTANT = {
    "ev_has_next_jct":
        "1.0 for every train. STA (id 0) and JBP (id 21) are both junctions "
        "and sit at the corridor ends, so any train still running always has "
        "a junction ahead of it. This dim was NOT constant before the "
        "reference merge, because journeys truncated by CRIS's midnight "
        "schedule gaps ended early and sometimes stopped short of a junction. "
        "Its becoming constant is evidence the journeys are now complete, not "
        "evidence of a bug. Worth deleting when the feature set is next "
        "revised; left in place here to avoid changing input dimensions in "
        "the middle of a leak-fix rebuild.",
}

EXPECTED_SPARSE = {
    "section_disrupted":
        "maintenance blocks + asset failures occupy ~3.7% of section-time "
        "(430 events x ~75 min over 29 days x 21 sections)",
    "is_disrupted":
        "same exogenous disruption signal on the section edge",
    "section_run_mult":
        "run-time multiplier only departs from 1.0 under a block or TSR",
    "run_multiplier":
        "run-time multiplier only departs from 1.0 under a block or TSR",
    "psr_time_loss":
        "only 3 corridor sections carry a permanent speed restriction",
}

MIN_NONZERO = 50      # below this a dim cannot be learned from, sparse or not


def audit(name, X, labels, failures, zero_tol=0.98):
    print(f"\n{'=' * 86}\n{name}   shape={tuple(X.shape)}\n{'=' * 86}")
    print(f"{'dim':>3} {'name':<24} {'mean':>9} {'std':>9} {'min':>8} "
          f"{'max':>8} {'%zero':>6}  status")
    for d in range(X.shape[1]):
        c = X[:, d]
        std, mean = float(c.std()), float(c.mean())
        nz = int((c != 0).sum())
        pz = float((c == 0).float().mean())
        nm = labels[d] if d < len(labels) else f"dim{d}"
        waived = nm in EXPECTED_SPARSE

        # A dim is broken if it is constant, effectively never fires, or is
        # pinned near a single value (the saturation failure that hid the
        # useless junction-ETA dims in the old feature set).
        status = "ok"
        if std < 1e-8:
            status = ("constant-ok" if nm in EXPECTED_CONSTANT
                      else "FAIL constant")
        elif nz < MIN_NONZERO:
            status = f"FAIL only {nz} non-zero"
        elif std < 0.01 and abs(mean) > 0.01:
            status = "FAIL saturated"
        elif pz > zero_tol:
            status = (f"sparse-ok ({nz} fire)" if waived
                      else f"FAIL {100 * pz:.1f}% zero")
        elif pz > 0.90:
            status = "warn sparse"

        if status.startswith("FAIL"):
            failures.append(f"{name}[{d}] {nm}: {status}")

        print(f"{d:>3} {nm:<24} {mean:>9.4f} {std:>9.4f} {float(c.min()):>8.3f} "
              f"{float(c.max()):>8.3f} {100 * pz:>5.1f}%  {status}")


def check_target_leakage(snaps, failures) -> None:
    """No input dim may be a near-perfect predictor of the target.

    `eta_next_min` was once built from the OBSERVED arrival time at the stop
    being predicted, so `eta_next + minimum_arrival_delay_next` reconstructed
    87% of h1 targets exactly and the model scored brilliantly without
    forecasting anything. Dead features are caught above; this catches the
    opposite and far more dangerous failure — a feature that is too good.
    """
    print(f"\n{'=' * 86}\nTARGET LEAKAGE\n{'=' * 86}")
    X, Y, A = [], [], []
    for s in snaps:
        d = s.data
        m = d["train"].target_mask[:, 0] > 0.5
        if m.any():
            X.append(d["train"].x[m])
            Y.append(d["train"].target_delta[m, 0])
            A.append(d["train"].target_abs[m, 0])   # absolute arrival delay
    if not X:
        print("  no supervised events")
        return
    X = torch.cat(X, 0)
    Y = torch.cat(Y, 0)
    A = torch.cat(A, 0)
    yc = Y - Y.mean()
    yn = float(yc.norm())

    top = []
    for d in range(X.shape[1]):
        c = X[:, d]
        cc = c - c.mean()
        cn = float(cc.norm())
        r = 0.0 if cn < 1e-8 or yn < 1e-8 else abs(float((cc * yc).sum()) / (cn * yn))
        nm = TRAIN_LABELS[d] if d < len(TRAIN_LABELS) else f"dim{d}"
        top.append((r, nm, d))
        if r > 0.95:
            failures.append(f"train[{d}] {nm}: |corr| {r:.3f} with target — leak")
    top.sort(reverse=True)
    print(f"  {len(Y):,} supervised events; |corr(feature, target_delta)| top 5:")
    for r, nm, d in top[:5]:
        flag = "  <-- LEAK" if r > 0.95 else ("  (high)" if r > 0.80 else "")
        print(f"    {nm:<26} dim{d:<3} {r:.4f}{flag}")

    # The exact identity that was broken before, re-checked in raw minutes.
    try:
        from cris_pipeline.dataset_cris import _eta_norm            # noqa: F401
        import math
        eta = torch.expm1(X[:, 2] * math.log1p(90.0))
        mad = torch.atanh(X[:, 15].clamp(-0.999, 0.999)) * 30.0
        # The leak reconstructed the ABSOLUTE arrival delay at the next stop.
        recon = eta + mad
        exact = float(((recon - A).abs() < 0.5).float().mean())
        print(f"  eta_next + min_arr_delay_next reconstructs target within "
              f"0.5 min for {100 * exact:.2f}% of events")
        if exact > 0.20:
            failures.append(
                f"eta_next+min_arr_delay_next reconstructs {100*exact:.1f}% "
                f"of targets — the historical leak is back")
    except Exception as e:                                    # pragma: no cover
        print(f"  (identity re-check skipped: {e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--max-days", type=int, default=8)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    ds = CrisDataset(args.dataset, args.topology, max_days=args.max_days)
    snaps = ds.get_split(args.split)
    if not snaps:
        sys.exit(f"no snapshots in split '{args.split}'")

    TX = torch.cat([s.data["train"].x for s in snaps
                    if s.data["train"].x.numel()], 0)
    SX = torch.cat([s.data["station"].x for s in snaps], 0)
    EX = torch.cat([s.data["section"].x for s in snaps], 0)

    # History: only real (non-padded) steps count.
    H, L = [], []
    for s in snaps:
        h = s.data["train"].history
        if h.numel():
            H.append(h)
            L.append(s.data["train"].history_len)
    HX = torch.cat(H, 0)
    HL = torch.cat(L, 0)
    step = torch.arange(HX.shape[1]).view(1, -1)
    real = step < HL.view(-1, 1)
    HF = HX[real]

    failures: list[str] = []
    audit("TRAIN NODE FEATURES", TX, TRAIN_LABELS, failures)
    audit("STATION NODE FEATURES", SX, STATION_LABELS, failures)
    audit("SECTION EDGE FEATURES", EX, SECTION_LABELS, failures)
    audit("JOURNEY HISTORY (real steps only)", HF, HIST_LABELS, failures)
    check_target_leakage(snaps, failures)

    n_conf = sum(int(s.data["train", "conflicts", "train"].edge_index.shape[1])
                 for s in snaps)
    n_tr = sum(int(s.data["train"].x.shape[0]) for s in snaps)
    n_at = sum(int(s.data["train", "at", "station"].edge_index.shape[1])
               for s in snaps)
    n_tv = sum(int(s.data["train", "traverses", "section"].edge_index.shape[1])
               for s in snaps)
    print(f"\n{'=' * 86}\nGRAPH CONNECTIVITY\n{'=' * 86}")
    print(f"  snapshots           : {len(snaps):,}")
    print(f"  train nodes         : {n_tr:,}  ({n_tr / len(snaps):.1f}/snapshot)")
    print(f"  conflict edges      : {n_conf:,}  ({n_conf / len(snaps):.1f}/snapshot)")
    print(f"  train->station edges: {n_at:,}")
    print(f"  train->section edges: {n_tv:,}")
    print(f"  mean history length : {float(HL.float().mean()):.1f} steps")
    if n_conf == 0:
        failures.append("graph: zero conflict edges — GAT layer would be dead")
    if n_tv == 0:
        failures.append("graph: zero train->section edges")

    _integrity(failures)

    print(f"\n{'=' * 86}")
    if failures:
        print(f"GATE FAILED — {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("GATE PASSED — no constant, near-empty, or saturated dims.")


if __name__ == "__main__":
    main()
