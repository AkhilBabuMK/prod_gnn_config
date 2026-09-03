# -*- coding: utf-8 -*-
"""
cris_pipeline/analyse_causes.py
===============================
Cause autopsy for delay creation on the corridor, at EXPLICIT thresholds.

Answers two questions with numbers you can re-check:

  1. Where is delay created -- standing at stations, or running between them?
  2. When a large amount is created at a station, what else was happening?

Everything is reported as a sweep over the "large event" threshold, because the
headline share of any single cause depends on where that line is drawn. A
single number quoted without its threshold is not a fact.

Control column: the same precedence test applied to stops that did NOT produce
a large gain. Without it, "58% of holds had a higher-priority train nearby"
means nothing -- it could just be a busy railway.

Usage:
    python -m cris_pipeline.analyse_causes
    python -m cris_pipeline.analyse_causes --split all --window 20
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .config import JOURNEYS_DIR, N_TRAIN_DAYS, N_VAL_DAYS
from .build_dataset import PRECEDENCE_WINDOW_MIN, PRIORITY_MARGIN

THRESHOLDS = [5.0, 10.0, 15.0, 20.0, 30.0, 45.0]


def _f(v, d=0.0):
    return d if v is None else float(v)


def load_days(split: str):
    """Return {date: [journey, ...]} for the requested split."""
    by_date = defaultdict(list)
    for train_dir in sorted(Path(JOURNEYS_DIR).iterdir()):
        if not train_dir.is_dir():
            continue
        for f in sorted(train_dir.glob("*.json")):
            j = json.loads(f.read_text(encoding="utf-8"))
            by_date[j["corridor_date"]].append(j)

    dates = sorted(by_date)
    # build_dataset drops partial boundary days; mirror that here so the split
    # boundaries line up with what the model actually trained on.
    dates = [d for d in dates if len(by_date[d]) > 80]
    if split == "train":
        keep = dates[:N_TRAIN_DAYS]
    elif split == "validation":
        keep = dates[N_TRAIN_DAYS:N_TRAIN_DAYS + N_VAL_DAYS]
    elif split == "test":
        keep = dates[N_TRAIN_DAYS + N_VAL_DAYS:]
    else:
        keep = dates
    return {d: by_date[d] for d in keep}


def station_occupancy(journeys):
    """{station_code: [(mid_time, priority, train_no), ...]} over one day.

    A train "occupies" a station across its arrival..departure window; we index
    it by the midpoint, which is enough resolution for a 20-minute window test.
    """
    occ = defaultdict(list)
    for j in journeys:
        pri = _f(j.get("priority"))
        for s in j["stops"]:
            if not s.get("has_actual"):
                continue
            a = s.get("actual_arr_min")
            d = s.get("actual_dep_min")
            if a is None and d is None:
                continue
            t = _f(a if a is not None else d)
            occ[s["station_code"]].append((t, pri, j["train_no"]))
    for k in occ:
        occ[k].sort()
    return occ


def higher_priority_near(occ, station, t, pri, train_no, window):
    """Was a strictly higher-priority train at this station within `window`?"""
    for ot, opri, otn in occ.get(station, ()):
        if otn == train_no:
            continue
        if abs(ot - t) <= window and opri > pri + PRIORITY_MARGIN:
            return True
    return False


def analyse(split: str, window: float):
    days = load_days(split)
    print(f"split={split}  days={len(days)}  window={window:.0f} min  "
          f"priority margin={PRIORITY_MARGIN}")

    hold_total = run_total = 0.0
    events = []          # (gain_min, is_hold, higher_prio_near)

    for date, journeys in days.items():
        occ = station_occupancy(journeys)
        for j in journeys:
            if not j.get("is_target_eligible"):
                continue
            pri = _f(j.get("priority"))
            stops = [s for s in j["stops"] if s.get("has_actual")]
            for i, s in enumerate(stops):
                arr = s.get("arr_delay_min")
                dep = s.get("dep_delay_min")

                # (1) delay created while STANDING at this station
                if arr is not None and dep is not None:
                    hold = _f(dep) - _f(arr)
                    if hold > 0:
                        hold_total += hold
                        t = _f(s.get("actual_arr_min"), _f(s.get("actual_dep_min")))
                        events.append((
                            hold, True,
                            higher_priority_near(occ, s["station_code"], t,
                                                 pri, j["train_no"], window),
                        ))

                # (2) delay created while RUNNING to the next station
                if i + 1 < len(stops) and dep is not None:
                    nxt = stops[i + 1].get("arr_delay_min")
                    if nxt is not None:
                        run = _f(nxt) - _f(dep)
                        if run > 0:
                            run_total += run
                            t = _f(stops[i + 1].get("actual_arr_min"))
                            events.append((
                                run, False,
                                higher_priority_near(
                                    occ, stops[i + 1]["station_code"], t,
                                    pri, j["train_no"], window),
                            ))

    total = hold_total + run_total
    print(f"\nDELAY CREATION  (all positive increments, {len(events):,} events)")
    print(f"  standing at stations : {hold_total:9,.0f} min  "
          f"{hold_total / total:5.1%}")
    print(f"  running between      : {run_total:9,.0f} min  "
          f"{run_total / total:5.1%}")

    print(f"\nPRECEDENCE SHARE BY THRESHOLD")
    print(f"  'large event' = a single increment greater than the threshold\n")
    print(f"  {'thresh':>7} {'events':>8} {'%of all':>8} {'higher-prio near':>17} "
          f"{'control':>9} {'lift':>7}")
    print("  " + "-" * 62)

    for th in THRESHOLDS:
        big = [e for e in events if e[0] > th]
        small = [e for e in events if e[0] <= th]
        if not big:
            continue
        p_big = sum(1 for e in big if e[2]) / len(big)
        p_small = (sum(1 for e in small if e[2]) / len(small)) if small else 0.0
        lift = (p_big / p_small) if p_small else float("inf")
        print(f"  {th:7.0f} {len(big):8,} {len(big)/len(events):7.2%} "
              f"{p_big:16.1%} {p_small:8.1%} {lift:6.2f}x")

    print(f"\nSAME, HOLDS ONLY  (delay created while standing)")
    print(f"  {'thresh':>7} {'events':>8} {'higher-prio near':>17} "
          f"{'control':>9} {'lift':>7}")
    print("  " + "-" * 54)
    holds = [e for e in events if e[1]]
    for th in THRESHOLDS:
        big = [e for e in holds if e[0] > th]
        small = [e for e in holds if e[0] <= th]
        if not big:
            continue
        p_big = sum(1 for e in big if e[2]) / len(big)
        p_small = (sum(1 for e in small if e[2]) / len(small)) if small else 0.0
        lift = (p_big / p_small) if p_small else float("inf")
        print(f"  {th:7.0f} {len(big):8,} {p_big:16.1%} {p_small:8.1%} "
              f"{lift:6.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train",
                    choices=["train", "validation", "test", "all"])
    ap.add_argument("--window", type=float, default=PRECEDENCE_WINDOW_MIN)
    args = ap.parse_args()
    analyse(args.split, args.window)


if __name__ == "__main__":
    main()
