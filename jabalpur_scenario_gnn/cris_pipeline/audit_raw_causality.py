# -*- coding: utf-8 -*-
"""
cris_pipeline/audit_raw_causality.py
====================================
Causality audit that perturbs the RAW JOURNEY RECORD, before _build_instance.

audit_causality.py perturbs the already-built instance:

    inst = BD._build_instance(rec, di)     # <- fills computed here
    world = perturb(instances, t, shift)   # <- perturbation applied here

so any leak INSIDE _build_instance is invisible to it. That is not a
hypothetical gap. _build_instance is where the carried-forward delay estimate
is computed and where missing arrival/departure timestamps are back-filled,
and on 2025-07-30 a leak lived in exactly that code: a train standing at a
station had its missing arrival back-filled from a `carried` value that had
already absorbed that stop's own (not yet taken) departure delay.

This audit closes the gap. It shifts the future in the RAW record and rebuilds
everything downstream from scratch, so `_build_instance` runs on perturbed
input and its internal derivations are covered too.

Invariant: for a train whose observable present is unchanged, every feature at
time t must be bit-identical between the two futures.

    python -m cris_pipeline.audit_raw_causality
    python -m cris_pipeline.audit_raw_causality --days 4 --ticks 32
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path

import cris_pipeline.build_dataset as BD
from .audit_features import (TRAIN_LABELS, STATION_LABELS, SECTION_LABELS,
                             HIST_LABELS)
from .config import JOURNEYS_DIR, TOPOLOGY_PATH, DATASET_DIR
from .dataset_cris import CrisDataset

SHIFT_PAIRS = [(1.0, 4.0), (3.0, 11.0), (17.0, 53.0), (137.0, 991.0)]


def shift_raw_future(records: list, day_idx: int, t: float,
                     shift: float) -> list:
    """Delay every stop the train has not yet REACHED, in the raw record.

    'Not yet reached' is decided from the unperturbed instance, so both futures
    agree on where the train is. Only stops strictly after the current position
    move, and for a train standing at a station its own pending departure moves
    too -- it has not happened yet, so it is future.
    """
    out = []
    off = day_idx * 1440
    for rec in records:
        rec = copy.deepcopy(rec)
        inst = BD._build_instance(rec, day_idx)
        if inst is None:
            out.append(rec)
            continue
        pos = BD._find_position(inst, t)
        if pos is None:
            out.append(rec)
            continue
        i, mode = pos["stop_idx"], pos["mode"]

        # Map instance stop index back onto the raw stop list.
        #
        # This MUST preserve order, not match on station code. Corridor
        # journeys can visit the same station twice -- train 00150 has 44 raw
        # stops over 22 stations, running out and back -- so a code lookup
        # returns the wrong occurrence and the audit then shifts stops the
        # train has ALREADY passed. That made the audit report 50 leaking
        # features that were artefacts of its own perturbation.
        #
        # _build_instance only ever drops stops (unknown station id), never
        # reorders, so walking both lists forward gives the true alignment.
        inst_codes = [s["station"] for s in inst["stops"]]
        code_of = {v: k for k, v in BD._CODE_TO_ID.items()}
        imap: list[int | None] = []
        j = 0
        for sid in inst_codes:
            want = code_of.get(sid)
            while j < len(rec["stops"]) and \
                    rec["stops"][j].get("station_code") != want:
                j += 1
            imap.append(j if j < len(rec["stops"]) else None)
            j += 1

        def raw_index(ii):
            return imap[ii] if ii < len(imap) else None

        for ii in range(i + 1, len(inst["stops"])):
            k = raw_index(ii)
            if k is None:
                continue
            s = rec["stops"][k]
            for f in ("actual_arr_min", "actual_dep_min",
                      "arr_delay_min", "dep_delay_min"):
                if s.get(f) is not None:
                    s[f] = float(s[f]) + shift

        if mode == "at_station":
            k = raw_index(i)
            if k is not None:
                s = rec["stops"][k]
                # departure not yet taken -> it is the future
                for f in ("actual_dep_min", "dep_delay_min"):
                    if s.get(f) is not None:
                        s[f] = float(s[f]) + shift
        out.append(rec)
    return out


def build(ds, records, di, t, date, day_meta, freight, disrupt, detent):
    instances = []
    for rec in records:
        inst = BD._build_instance(rec, di)
        if inst:
            inst["detention"] = detent.get(
                inst["train_no"], inst.get("journey_date") or date) or {}
            instances.append(inst)
    if not instances:
        return None
    st_state = BD._StationState()
    prev: dict[str, float] = {}
    snap = BD.build_snapshot(di, t, instances, st_state, day_meta, prev,
                             freight, disrupt, date)
    if not snap["train_nodes"]:
        return None
    by_iid = {i["instance_id"]: i for i in instances}
    routes = {i["instance_id"]: i["route"] for i in instances}
    return snap, ds.materialise(snap, by_iid, routes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journeys", default=str(JOURNEYS_DIR))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--ticks", type=int, default=20)
    args = ap.parse_args()

    topo = json.loads(Path(args.topology).read_text(encoding="utf-8"))
    BD._init_topology(topo)
    ds = CrisDataset(args.dataset, args.topology, lazy=True, verbose=False)

    from .overlays import FreightOverlay, DisruptionOverlay, DetentionOverlay
    freight = FreightOverlay.load()
    pairs = {(int(s["from_id"]), int(s["to_id"])) for s in BD._SECTIONS.values()}
    disrupt = DisruptionOverlay.load(pairs)
    detent = DetentionOverlay.load()

    by_date = BD._load_journeys(Path(args.journeys))
    dates = BD._drop_partial_days(by_date, sorted(by_date))[:args.days]

    groups = {"train": (TRAIN_LABELS, "train"),
              "station": (STATION_LABELS, "station"),
              "section": (SECTION_LABELS, "section")}
    maxdiff = {g: {} for g in groups}
    hist_diff: dict[str, float] = {}
    n_cmp = n_skip = 0

    for date in dates:
        di = dates.index(date)
        day_meta = {"day_of_week": dt.date.fromisoformat(date).weekday()}
        base = by_date[date]
        step = max(len(BD.SNAPSHOT_SCHEDULE) // args.ticks, 1)
        for m in BD.SNAPSHOT_SCHEDULE[::step]:
            t = di * 1440 + m
            for sa, sb in SHIFT_PAIRS:
                ra = shift_raw_future(base, di, t, sa)
                rb = shift_raw_future(base, di, t, sb)
                A = build(ds, ra, di, t, date, day_meta, freight, disrupt, detent)
                B = build(ds, rb, di, t, date, day_meta, freight, disrupt, detent)
                if A is None or B is None:
                    continue
                (_, da), (_, db) = A, B
                if da is None or db is None:
                    continue
                if da["train"].x.shape != db["train"].x.shape:
                    n_skip += 1
                    continue
                n_cmp += 1
                for g, (labels, key) in groups.items():
                    X, Y = da[key].x, db[key].x
                    d = (X - Y).abs().max(dim=0).values
                    for j in range(X.shape[1]):
                        nm = labels[j] if j < len(labels) else f"dim{j}"
                        maxdiff[g][nm] = max(maxdiff[g].get(nm, 0.0), float(d[j]))
                ha, hb = da["train"].history, db["train"].history
                if ha.shape == hb.shape:
                    d = (ha - hb).abs().amax(dim=(0, 1))
                    for j in range(ha.shape[2]):
                        nm = (HIST_LABELS[j] if j < len(HIST_LABELS)
                              else f"hdim{j}")
                        hist_diff[nm] = max(hist_diff.get(nm, 0.0), float(d[j]))

    print(f"\ncompared {n_cmp} snapshot pairs across {len(SHIFT_PAIRS)} shift "
          f"pairs, perturbing the RAW record ({n_skip} skipped: node count "
          f"changed)")

    fails = []
    for g in ("train", "station", "section"):
        print(f"\n{'=' * 74}\n{g.upper()} FEATURES — max |difference| between "
              f"the two futures\n{'=' * 74}")
        for nm, v in sorted(maxdiff[g].items(), key=lambda kv: -kv[1]):
            tag = "  READS THE FUTURE" if v > 1e-6 else "  ok"
            print(f"  {nm:<32} {v:.6f}{tag}")
            if v > 1e-6:
                fails.append(f"{g}.{nm}  max diff {v:.4f}")
    print(f"\n{'=' * 74}\nHISTORY FEATURES\n{'=' * 74}")
    for nm, v in sorted(hist_diff.items(), key=lambda kv: -kv[1]):
        tag = "  READS THE FUTURE" if v > 1e-6 else "  ok"
        print(f"  {nm:<32} {v:.6f}{tag}")
        if v > 1e-6:
            fails.append(f"history.{nm}  max diff {v:.4f}")

    print(f"\n{'=' * 74}")
    if fails:
        print(f"RAW CAUSALITY AUDIT FAILED — {len(fails)} feature(s):")
        for f in fails:
            print(f"  - {f}")
        raise SystemExit(1)
    print("RAW CAUSALITY AUDIT PASSED — no feature depends on the future, "
          "including everything derived inside _build_instance.")


if __name__ == "__main__":
    main()
