# -*- coding: utf-8 -*-
"""
cris_pipeline/audit_causality.py
================================
Proves that no feature reads the future — by experiment, not by reading code.

The `eta_next_min` leak (87% of h1 targets reconstructable to the exact minute)
survived a full feature audit, a wiring diagnostic and a training run because
every one of those inspects features that all come from ground truth. Nothing
in that toolchain can distinguish "informative" from "is the answer".

METHOD
------
Fix a snapshot time t. Take each train's position at t as given — that is
observed. Then rewrite every stop the train has NOT yet reached, shifting its
actual times and delays by a constant, and rebuild the snapshot. Do it twice
with two different shifts.

A feature computed only from the past and present cannot notice. Any dim whose
value differs between the two worlds is, by construction, reading information
that does not exist at time t. There is no judgement call and nothing to argue
about: same inputs up to t, different answer.

The position itself is held fixed (the shifts are positive, so a train that had
not arrived still has not arrived) because "has this train reached the next
station yet" IS observable in real time and must stay available.

    python -m cris_pipeline.audit_causality
    python -m cris_pipeline.audit_causality --days 3
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cris_pipeline.config import (                                   # noqa: E402
    JOURNEYS_DIR, TOPOLOGY_PATH, DATASET_DIR,
)
from cris_pipeline import build_dataset as BD                        # noqa: E402
from cris_pipeline.dataset_cris import CrisDataset                   # noqa: E402
from cris_pipeline.audit_features import (                           # noqa: E402
    TRAIN_LABELS, STATION_LABELS, SECTION_LABELS, HIST_LABELS,
)

# Several pairs, spanning small to large. A single large pair is NOT enough:
# clamped/normalised dims saturate, so two big shifts both land on the ceiling
# and the leak cancels out. The first run of this audit used (137, 991) and
# gave a FALSE PASS on a deliberately re-introduced `eta_next` leak because
# `_eta_norm` clamps at 90 min. Small pairs catch clamped features; large pairs
# catch features with wide dynamic range. Every pair must agree.
SHIFT_PAIRS = [(1.0, 4.0), (3.0, 11.0), (17.0, 53.0), (137.0, 991.0)]


def shift_future(instances: list, t: float, shift: float) -> list:
    """Move every not-yet-reached stop later by `shift` minutes.

    Only stops strictly after the train's current position are touched, so the
    observable present — where each train is, and that it has not yet arrived
    at the next station — is preserved exactly.
    """
    out = copy.deepcopy(instances)
    for inst in out:
        pos = BD._find_position(inst, t)
        if pos is None:
            continue
        i = pos["stop_idx"]
        for k in range(i + 1, len(inst["stops"])):
            s = inst["stops"][k]
            s["actual_arr_abs"] = float(s["actual_arr_abs"]) + shift
            s["actual_dep_abs"] = float(s["actual_dep_abs"]) + shift
            if s["arr_delay"] is not None:
                s["arr_delay"] = float(s["arr_delay"]) + shift
            if s["dep_delay"] is not None:
                s["dep_delay"] = float(s["dep_delay"]) + shift
            if s.get("delay_est") is not None:
                s["delay_est"] = float(s["delay_est"]) + shift
    return out


def shift_pending_departure(instances: list, t: float, shift: float) -> list:
    """For trains STANDING at a station right now, push their (not yet taken)
    departure later.

    `shift_future` only perturbs stops AFTER the current position, so it cannot
    see a feature that reads the CURRENT stop's `dep_delay` — which is the
    future for a train that has not departed. Two leaks hid in exactly that
    gap: `_path_etas` and `_junction_etas` both built their ETAs on the
    eventual departure delay, so a train three minutes into a fourteen-minute
    hold already "knew" it would leave 36 late.

    A train still standing at t is still standing after the shift (the shift is
    positive), so its observable present is unchanged and every feature must be
    identical.
    """
    out = copy.deepcopy(instances)
    for inst in out:
        pos = BD._find_position(inst, t)
        if pos is None or pos["mode"] != "at_station":
            continue
        s = inst["stops"][pos["stop_idx"]]
        if s.get("actual_dep_abs") is None:
            continue
        s["actual_dep_abs"] = float(s["actual_dep_abs"]) + shift
        if s.get("dep_delay") is not None:
            s["dep_delay"] = float(s["dep_delay"]) + shift
    return out


def build(ds, instances, di, t, date, day_meta, freight, disrupt):
    st_state = BD._StationState()
    prev: dict[str, float] = {}
    snap = BD.build_snapshot(di, t, instances, st_state, day_meta, prev,
                             freight, disrupt, date)
    if not snap["train_nodes"]:
        return None, None
    by_iid = {i["instance_id"]: i for i in instances}
    routes = {i["instance_id"]: i["route"] for i in instances}
    data = ds.materialise(snap, by_iid, routes)
    return snap, data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journeys", default=str(JOURNEYS_DIR))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--ticks", type=int, default=24)
    args = ap.parse_args()

    topo = json.loads(Path(args.topology).read_text(encoding="utf-8"))
    BD._init_topology(topo)
    ds = CrisDataset(args.dataset, args.topology, lazy=True, verbose=False)

    from cris_pipeline.overlays import (FreightOverlay, DisruptionOverlay,
                                        DetentionOverlay)
    freight = FreightOverlay.load()
    pairs = {(int(s["from_id"]), int(s["to_id"])) for s in BD._SECTIONS.values()}
    disrupt = DisruptionOverlay.load(pairs)
    detent = DetentionOverlay.load()

    by_date = BD._load_journeys(Path(args.journeys))
    dates = BD._drop_partial_days(by_date, sorted(by_date))[:args.days]

    groups = {
        "train":   (TRAIN_LABELS,   "train"),
        "station": (STATION_LABELS, "station"),
        "section": (SECTION_LABELS, "section"),
    }
    maxdiff = {g: {} for g in groups}
    hist_diff: dict[str, float] = {}
    n_cmp = 0

    for date in dates:
        di = dates.index(date)
        day_meta = {"day_of_week": dt.date.fromisoformat(date).weekday()}
        instances = []
        for rec in by_date[date]:
            inst = BD._build_instance(rec, di)
            if inst:
                inst["detention"] = detent.get(
                    inst["train_no"], inst.get("journey_date") or date) or {}
                instances.append(inst)
        if not instances:
            continue

        step = max(len(BD.SNAPSHOT_SCHEDULE) // args.ticks, 1)
        for m in BD.SNAPSHOT_SCHEDULE[::step]:
          t = di * 1440 + m
          for sh_a, sh_b in SHIFT_PAIRS:
           for perturb in (shift_future, shift_pending_departure):
             wa = perturb(instances, t, sh_a)
             wb = perturb(instances, t, sh_b)
             sa, da = build(ds, wa, di, t, date, day_meta, freight, disrupt)
             sb, db = build(ds, wb, di, t, date, day_meta, freight, disrupt)
             if da is None or db is None:
                 continue
             # Same trains, same order — position was deliberately preserved.
             if da["train"].x.shape != db["train"].x.shape:
                 print(f"  !! node count changed at t={m} "
                       f"({da['train'].x.shape} vs {db['train'].x.shape}) — "
                       f"position itself depends on the future")
                 continue
             n_cmp += 1

             for g, (labels, key) in groups.items():
                 A, B = da[key].x, db[key].x
                 d = (A - B).abs().max(dim=0).values
                 for j in range(A.shape[1]):
                     nm = labels[j] if j < len(labels) else f"dim{j}"
                     maxdiff[g][nm] = max(maxdiff[g].get(nm, 0.0), float(d[j]))

             HA, HB = da["train"].history, db["train"].history
             if HA.shape == HB.shape and HA.numel():
                 dh = (HA - HB).abs().amax(dim=(0, 1))
                 for j in range(HA.shape[-1]):
                     nm = HIST_LABELS[j] if j < len(HIST_LABELS) else f"dim{j}"
                     hist_diff[nm] = max(hist_diff.get(nm, 0.0), float(dh[j]))

             # Targets MUST move — they are the future. If they do not, the
             # perturbation never landed and this whole audit proves nothing.
             ta = da["train"].target_abs[da["train"].target_mask > 0.5]
             tb = db["train"].target_abs[db["train"].target_mask > 0.5]
             if ta.numel() and float((ta - tb).abs().max()) < 1e-6:
                 print(f"  !! targets identical at t={m} — perturbation "
                       f"did not take effect; audit is invalid")

    print(f"\ncompared {n_cmp} snapshot pairs across {len(SHIFT_PAIRS)} shift "
          f"pairs {[(int(a), int(b)) for a, b in SHIFT_PAIRS]} min\n")

    bad = []
    for g in list(groups) + ["history"]:
        src = hist_diff if g == "history" else maxdiff[g]
        print("=" * 74)
        print(f"{g.upper()} FEATURES — max |difference| between the two futures")
        print("=" * 74)
        for nm, v in sorted(src.items(), key=lambda kv: -kv[1]):
            status = "ok" if v < 1e-6 else "READS THE FUTURE"
            if v >= 1e-6:
                bad.append(f"{g}.{nm}  max diff {v:.4f}")
            print(f"  {nm:<28} {v:>12.6f}   {status}")
        print()

    print("=" * 74)
    if bad:
        print(f"CAUSALITY AUDIT FAILED — {len(bad)} feature(s) read the future:")
        for b in bad:
            print(f"  - {b}")
        sys.exit(1)
    print("CAUSALITY AUDIT PASSED — every feature is computable at time t.")


if __name__ == "__main__":
    main()
