# -*- coding: utf-8 -*-
"""
cris_pipeline/audit_trace_reads.py
==================================
Whitebox leak audit: record every field the feature code actually READS, and
flag any read of information the train has not reached yet.

Why this exists. The perturbation audits (audit_causality, audit_raw_causality)
are blackbox: they shift the future and check whether outputs move. That only
catches a leak if the chosen shift happens to change the number. A field can be
read and still not move the output under one particular perturbation -- clamped
dims, min/max reductions, values that saturate. Six leaks were found by
perturbation, each after the previous "it passes now"; that is not a method you
can certify a production system with.

This audit does not guess. Every stop dict is replaced by a proxy that logs
(stop_index, field_name) on access. After building a snapshot at time t we know
exactly what was touched, and can check it against what was knowable:

  KNOWABLE at time t, for a train at stop i:
    - anything about stops[0 .. i-1]            (already happened)
    - stops[i] arrival fields                    (it got there)
    - stops[i] departure fields ONLY if the train has already left, i.e. it is
      in_transit, not standing at i
    - schedule fields of ANY stop                (timetable, known in advance)
    - static identity of any stop                (station code, is_booked_halt)

  NOT KNOWABLE:
    - actual/observed fields of stops[i+1 ..]    (hasn't got there)
    - stops[i] departure fields while standing at i (hasn't left)

A read that is flagged is not automatically a bug -- code may read a field and
discard it. But every flagged read is a place a human must look, and an empty
report is a much stronger statement than "my perturbation didn't move it".

    python -m cris_pipeline.audit_trace_reads
    python -m cris_pipeline.audit_trace_reads --days 5 --ticks 30 --verbose
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import cris_pipeline.build_dataset as BD
from .config import JOURNEYS_DIR, TOPOLOGY_PATH, DATASET_DIR
from .dataset_cris import CrisDataset

# Fields that come from the published timetable or are static identity. These
# are knowable for every stop, arbitrarily far ahead, and reading them is fine.
SCHEDULE_FIELDS = {
    "sched_arr_abs", "sched_dep_abs", "sched_arr_min", "sched_dep_min",
    "station", "station_code", "station_id", "is_booked_halt", "is_ptt_halt",
    "is_division_stop", "is_target", "distance_km", "platform",
}
# Fields that describe what actually happened. Reading these about a stop the
# train has not reached is a leak.
ACTUAL_FIELDS = {
    "actual_arr_abs", "actual_dep_abs", "actual_arr_min", "actual_dep_min",
    "arr_delay", "dep_delay", "arr_delay_min", "dep_delay_min",
    "delay_est", "delay_est_arr", "has_actual",
}
ARRIVAL_FIELDS = {"actual_arr_abs", "actual_arr_min", "arr_delay",
                  "arr_delay_min", "delay_est_arr", "has_actual"}


class TracedStop(dict):
    """A stop dict that records which fields are read."""

    __slots__ = ("_idx", "_log")

    def __init__(self, base: dict, idx: int, log: list):
        super().__init__(base)
        self._idx = idx
        self._log = log

    def __getitem__(self, k):
        self._log.append((self._idx, k))
        return super().__getitem__(k)

    def get(self, k, default=None):
        self._log.append((self._idx, k))
        return super().get(k, default)


def classify(reads, pos_idx: int, at_station: bool):
    """Split logged reads into ok / leaked."""
    leaks = []
    for idx, field in reads:
        if field in SCHEDULE_FIELDS or field not in ACTUAL_FIELDS:
            continue
        if idx < pos_idx:
            continue                      # already happened
        if idx == pos_idx:
            if field in ARRIVAL_FIELDS:
                continue                  # it arrived; arrival is observable
            if not at_station:
                continue                  # already departed, so departure is past
            leaks.append((idx - pos_idx, field, "pending departure"))
            continue
        leaks.append((idx - pos_idx, field, "future stop"))
    return leaks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journeys", default=str(JOURNEYS_DIR))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=24)
    ap.add_argument("--verbose", action="store_true")
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

    violations = defaultdict(int)
    examples = {}
    n_trains = n_snaps = 0

    for date in dates:
        di = dates.index(date)
        day_meta = {"day_of_week": dt.date.fromisoformat(date).weekday()}
        step = max(len(BD.SNAPSHOT_SCHEDULE) // args.ticks, 1)

        for m in BD.SNAPSHOT_SCHEDULE[::step]:
            t = di * 1440 + m

            instances, logs = [], {}
            for rec in by_date[date]:
                inst = BD._build_instance(rec, di)
                if not inst:
                    continue
                inst["detention"] = detent.get(
                    inst["train_no"], inst.get("journey_date") or date) or {}
                log = []
                inst["stops"] = [TracedStop(s, i, log)
                                 for i, s in enumerate(inst["stops"])]
                logs[inst["instance_id"]] = (inst, log)
                instances.append(inst)
            if not instances:
                continue

            st_state = BD._StationState()
            prev: dict[str, float] = {}
            snap = BD.build_snapshot(di, t, instances, st_state, day_meta,
                                     prev, freight, disrupt, date)
            if not snap["train_nodes"]:
                continue
            by_iid = {i["instance_id"]: i for i in instances}
            routes = {i["instance_id"]: i["route"] for i in instances}
            ds.materialise(snap, by_iid, routes)
            n_snaps += 1

            for iid, (inst, log) in logs.items():
                pos = BD._find_position(inst, t)
                if pos is None:
                    continue
                n_trains += 1
                bad = classify(log, pos["stop_idx"],
                               pos["mode"] == "at_station")
                for offset, field, kind in bad:
                    key = (field, kind)
                    violations[key] += 1
                    if key not in examples:
                        examples[key] = (inst["train_no"], date, offset)

    print(f"\ntraced {n_snaps:,} snapshots, {n_trains:,} positioned trains")
    print(f"{'=' * 74}")
    if not violations:
        print("READ TRACE PASSED — no code path read an actual-outcome field "
              "for a stop the train had not reached, nor a pending departure.")
        return

    print(f"READ TRACE — {len(violations)} distinct (field, kind) violations")
    print(f"{'=' * 74}")
    print(f"  {'field':<20} {'kind':<20} {'reads':>9}  example")
    print("  " + "-" * 68)
    for (field, kind), n in sorted(violations.items(), key=lambda kv: -kv[1]):
        tn, date, off = examples[(field, kind)]
        print(f"  {field:<20} {kind:<20} {n:>9,}  train {tn} {date} "
              f"(+{off} stops)")
    print()
    print("A flagged read is a place to LOOK, not proof of a bug: the value may")
    print("be discarded. Cross-check each against audit_raw_causality, which")
    print("says whether the output actually depends on it.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
