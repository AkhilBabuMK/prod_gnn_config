# -*- coding: utf-8 -*-
"""
cris_pipeline/audit_journeys.py
===============================
END-TO-END DATA AUDIT — every journey file, every stop, against the raw source.

`audit_features` checks the TENSORS (nothing dead, nothing constant).
`audit_causality` checks that no feature reads the future.
Neither checks whether the numbers are TRUE.

This does. For every stop of every journey it re-derives the value from the
raw CRIS export and compares. Anything the pipeline invented, mis-dated, or
mis-offset shows up here as a mismatch, because the comparison never goes
through pipeline code -- it re-reads the CSV.

Checks, in order of how badly a failure would hurt:

  A  TIMESTAMP FIDELITY   corridor_date + stored minutes must reproduce the raw
                          timestamp exactly. Catches every day-offset, midnight
                          and month-boundary bug in one test.
  B  DELAY ARITHMETIC     arr_delay == actual_arr - sched_arr, exactly.
  C  ORDERING             arrival <= departure at a stop, and departure <=
                          the next stop's arrival. A violation means the train
                          is placed in the wrong section by _find_position.
  D  PLAUSIBILITY         implied speed between consecutive stops.
  E  PROVENANCE           which stops came from CRIS and which were filled from
                          the reference export, reported separately so a fill
                          can never hide inside the CRIS pass rate.
  F  EXTREMES             journeys with implausible delay, and -- the part that
                          actually matters -- whether they are allowed to be
                          TARGETS or are actor-only.

    python -m cris_pipeline.audit_journeys
    python -m cris_pipeline.audit_journeys --train 00902 --verbose
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cris_pipeline.config import JOURNEYS_DIR, TOPOLOGY_PATH, RUNNING_CSV  # noqa: E402

TOL_MIN = 0.5            # timestamp reproduction must be exact to the minute
MAX_KMH = 160.0          # above this the leg is physically impossible
MIN_KMH = 3.0            # below this the train is effectively stopped mid-leg


def _load_raw() -> pd.DataFrame:
    df = pd.read_csv(RUNNING_CSV, low_memory=False, dtype={"train_number": str})
    df["train_number"] = df["train_number"].astype(str).str.zfill(5)
    for c in ("scheduled_arrival_time", "actual_arrival_time",
              "scheduled_departure_time", "actual_departure_time"):
        # NOT dayfirst. RUNNING_CSV is ISO ("2025-09-08 01:35:00"); passing
        # dayfirst=True makes dateutil read "2025-09-08" as day 09 month 08 and
        # silently shifts the row 30 days, which is exactly the false failure
        # this audit reported on its first run. `build_journeys.load_running`
        # is already correct (no dayfirst) -- only MAINT_CSV/TSR_CSV are
        # genuinely day-first ("05-09-25 16:05:00"), and only overlays.py
        # parses those.
        df[c] = pd.to_datetime(df[c], errors="coerce", format="ISO8601")
    return df.sort_values(["train_number", "train_date", "serial_number"])


def _mins(ts, base: dt.date):
    """Minutes from midnight of `base`. None-safe."""
    if ts is None or pd.isna(ts):
        return None
    return (ts.to_pydatetime()
            - dt.datetime(base.year, base.month, base.day)).total_seconds() / 60.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journeys", default=str(JOURNEYS_DIR))
    ap.add_argument("--train", default=None, help="audit one train number only")
    ap.add_argument("--verbose", action="store_true",
                    help="print every mismatch, not just the summary")
    ap.add_argument("--max-print", type=int, default=15)
    args = ap.parse_args()

    topo = json.loads(Path(TOPOLOGY_PATH).read_text(encoding="utf-8"))
    corridor = set(topo["code_to_id"])

    print("reading raw export ...", flush=True)
    raw = _load_raw()
    raw = raw[raw.station_code.isin(corridor)]
    # Order-preserving index: the k-th corridor row of a run, NOT keyed by
    # station code -- a train can visit the same station twice (00150 has 44
    # corridor stops) and code-keying silently mismatches every stop after it.
    by_run: dict[tuple, list] = defaultdict(list)
    for r in raw.itertuples():
        by_run[(r.train_number, str(r.train_date))].append(r)

    files = sorted(Path(args.journeys).glob("*/*.json"))
    if args.train:
        want = args.train.zfill(5)
        files = [f for f in files if f.parent.name.zfill(5) == want]
    print(f"auditing {len(files):,} journey files\n", flush=True)

    n_stops = n_matched = 0
    fail = Counter()
    prov = Counter()
    problems: list[str] = []
    extreme: list[tuple] = []
    speeds: list[float] = []

    for fp in files:
        j = json.loads(fp.read_text(encoding="utf-8"))
        tno = str(j["train_no"]).zfill(5)
        cdate = dt.date.fromisoformat(j["corridor_date"])
        jdate = str(j.get("journey_date") or j["corridor_date"])
        stops = j["stops"]
        rows = by_run.get((tno, jdate), [])
        # Journeys are built by filtering the run to corridor stations in
        # order, so stop k corresponds to corridor row k of the same run.
        aligned = len(rows) == len(stops)

        for k, s in enumerate(stops):
            n_stops += 1
            code = s["station_code"]
            r = rows[k] if aligned and k < len(rows) else None
            if r is not None and r.station_code != code:
                fail["align"] += 1
                problems.append(f"{tno} {jdate} stop{k}: journey says {code}, "
                                f"raw says {r.station_code}")
                r = None

            # ---- A/E  timestamp fidelity + provenance -----------------------
            for fld, rawcol in (("actual_arr_min", "actual_arrival_time"),
                                ("actual_dep_min", "actual_departure_time"),
                                ("sched_arr_min", "scheduled_arrival_time"),
                                ("sched_dep_min", "scheduled_departure_time")):
                v = s.get(fld)
                if v is None:
                    continue
                if r is None:
                    prov[f"{fld}:no-raw-row"] += 1
                    continue
                exp = _mins(getattr(r, rawcol), cdate)
                if exp is None:
                    # Present in the journey, absent in CRIS -> reference fill.
                    prov[f"{fld}:reference-fill"] += 1
                    continue
                prov[f"{fld}:cris"] += 1
                n_matched += 1
                if abs(float(v) - exp) > TOL_MIN:
                    fail[f"timestamp:{fld}"] += 1
                    problems.append(
                        f"{tno} {jdate} {code} {fld}: stored {v}, "
                        f"raw implies {exp:.0f} (delta {float(v)-exp:+.0f} min)")

            # ---- B  delay arithmetic ---------------------------------------
            for dfld, afld, sfld in (("arr_delay_min", "actual_arr_min", "sched_arr_min"),
                                     ("dep_delay_min", "actual_dep_min", "sched_dep_min")):
                d, a, sc = s.get(dfld), s.get(afld), s.get(sfld)
                if d is None or a is None or sc is None:
                    continue
                if abs(float(d) - (float(a) - float(sc))) > 1e-6:
                    fail[f"arithmetic:{dfld}"] += 1
                    problems.append(f"{tno} {jdate} {code} {dfld}: stored {d}, "
                                    f"actual-sched = {float(a)-float(sc)}")

            # ---- C  ordering within a stop ---------------------------------
            aa, ad = s.get("actual_arr_min"), s.get("actual_dep_min")
            if aa is not None and ad is not None and float(ad) < float(aa) - 1e-6:
                fail["order:dep-before-arr"] += 1
                problems.append(f"{tno} {jdate} {code}: departs {ad} before it "
                                f"arrives {aa}")

            # ---- C/D  ordering and speed across the leg --------------------
            if k + 1 < len(stops):
                nx = stops[k + 1]
                if ad is not None and nx.get("actual_arr_min") is not None:
                    if float(nx["actual_arr_min"]) < float(ad) - 1e-6:
                        fail["order:next-arr-before-dep"] += 1
                        problems.append(
                            f"{tno} {jdate} {code}->{nx['station_code']}: next "
                            f"arrival {nx['actual_arr_min']} precedes departure {ad}")
                    else:
                        km = None
                        if s.get("distance_km") is not None and nx.get("distance_km") is not None:
                            km = abs(float(nx["distance_km"]) - float(s["distance_km"]))
                        mins = float(nx["actual_arr_min"]) - float(ad)
                        if km and mins > 0:
                            kmh = km / (mins / 60.0)
                            speeds.append(kmh)
                            if kmh > MAX_KMH:
                                fail["speed:too-fast"] += 1
                                problems.append(
                                    f"{tno} {jdate} {code}->{nx['station_code']}: "
                                    f"{km:.1f} km in {mins:.0f} min = {kmh:.0f} km/h")

        # ---- F  extreme journeys ------------------------------------------
        ds = [abs(float(s["arr_delay_min"])) for s in stops
              if s.get("arr_delay_min") is not None]
        if ds and max(ds) > 720:
            extreme.append((tno, j["corridor_date"], max(ds),
                            bool(j.get("is_target_eligible"))))

    # ── report ───────────────────────────────────────────────────────────────
    print("=" * 78)
    print("A/B/C/D  CORRECTNESS")
    print("=" * 78)
    print(f"  stops audited            : {n_stops:,}")
    print(f"  values checked vs raw    : {n_matched:,}")
    if not fail:
        print("  result                   : PASS — no mismatch of any kind")
    else:
        print("  result                   : FAIL")
        for k, v in fail.most_common():
            print(f"     {k:32s} {v:,}")

    if speeds:
        speeds.sort()
        print(f"\n  implied leg speed  p50 {speeds[len(speeds)//2]:.0f}"
              f"  p95 {speeds[int(len(speeds)*.95)]:.0f}"
              f"  max {speeds[-1]:.0f} km/h")

    print("\n" + "=" * 78)
    print("E  PROVENANCE — where each stored value actually came from")
    print("=" * 78)
    for k in sorted(prov):
        print(f"  {k:38s} {prov[k]:,}")
    cris = sum(v for k, v in prov.items() if k.endswith(":cris"))
    ref = sum(v for k, v in prov.items() if k.endswith(":reference-fill"))
    if cris + ref:
        print(f"\n  reference-filled share of populated values: "
              f"{100*ref/(cris+ref):.2f}%")

    print("\n" + "=" * 78)
    print("F  EXTREME JOURNEYS (|delay| > 12 h) — are they allowed to be targets?")
    print("=" * 78)
    if not extreme:
        print("  none")
    else:
        elig = [e for e in extreme if e[3]]
        print(f"  journeys with a stop over 12 h late : {len(extreme)}")
        print(f"     target-eligible (SUPERVISED)     : {len(elig)}")
        print(f"     actor-only (not supervised)      : {len(extreme)-len(elig)}")
        for tno, d, mx, el in sorted(extreme, key=lambda x: -x[2])[:12]:
            print(f"     {tno}  {d}  max {mx:7.0f} min"
                  f"  {'SUPERVISED <-- check this' if el else 'actor-only'}")

    if problems and (args.verbose or fail):
        print("\n" + "=" * 78)
        print(f"SAMPLE PROBLEMS ({len(problems):,} total)")
        print("=" * 78)
        for p in problems[:(None if args.verbose else args.max_print)]:
            print("  " + p)

    print()
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
