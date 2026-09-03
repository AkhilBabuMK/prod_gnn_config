# -*- coding: utf-8 -*-
"""
cris_pipeline/crosscheck_reference.py
=====================================
Cross-check the official CRIS running data against the older reference export,
to decide whether the reference can safely fill CRIS's gaps.

CRIS stays authoritative. The only question this answers is:

    where CRIS has NO value, does the reference have one, and can we trust it?

The test that decides that is NOT coverage -- it is AGREEMENT. Where both
sources have a value for the same (train, date, station), do they say the same
thing? If they disagree often, the reference cannot be trusted where CRIS is
silent either, and we take nothing from it.

Matching is on (train_number, train_start_date, station_code, serial_number).
Both sources carry the train's own start date and the raw serial number, so no
inference about midnight crossing is needed here -- timestamps are absolute in
both files and are compared as absolute instants.

    python -m cris_pipeline.crosscheck_reference
    python -m cris_pipeline.crosscheck_reference --tolerance 0
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import RUNNING_CSV, CORRIDOR

REF_DIR = (Path(__file__).resolve().parent.parent / "old_data_reference_for_missing"
           / "20260116_073928_20260116_080059 (2)" / "20260116_073928")

FIELDS = [("scheduled_arrival_time", "scheduled_arrival_time"),
          ("scheduled_departure_time", "scheduled_departure_time"),
          ("actual_arrival_time", "actual_arrival_time"),
          ("actual_departure_time", "actual_departure_time")]


def _ts(v):
    """Parse a timestamp from either source; 'nan'/None/NaT -> None."""
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "nan", "NaT", "None"):
        return None
    try:
        t = pd.to_datetime(s, errors="coerce")
    except Exception:
        return None
    return None if pd.isna(t) else t


def load_reference():
    """{(train_no_str, date_str): {serial: record}} for corridor stops only."""
    out = {}
    n_files = n_rows = 0
    for fp in glob.glob(str(REF_DIR / "train_*.json")):
        try:
            d = json.loads(Path(fp).read_text(encoding="utf-8"))
        except Exception:
            continue
        n_files += 1
        tn = str(d.get("train_number", "")).strip()
        sd = str(d.get("train_start_date", "")).strip()      # 'DD/MM/YY'
        try:
            day, mon, yr = sd.split("/")
            date = f"20{yr}-{mon}-{day}"
        except ValueError:
            continue
        recs = {}
        for s in d.get("stations", []):
            if s.get("station_code") not in CORRIDOR:
                continue
            recs[int(s["serial_number"])] = s
            n_rows += 1
        if recs:
            out[(tn, date)] = recs
    print(f"reference : {n_files:,} files, {len(out):,} train-days on corridor, "
          f"{n_rows:,} corridor stop-rows")
    return out


def load_cris():
    d = pd.read_csv(RUNNING_CSV, low_memory=False)
    d = d[d.station_code.isin(CORRIDOR)].copy()
    d["tn"] = d.train_number.apply(
        lambda v: "" if pd.isna(v) else str(int(v)).zfill(5))
    d["dt"] = d.train_date.astype(str).str[:10]
    out = defaultdict(dict)
    for r in d.itertuples():
        out[(r.tn, r.dt)][int(r.serial_number)] = r
    print(f"CRIS      : {len(d):,} corridor stop-rows, "
          f"{len(out):,} train-days")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=int, default=0,
                    help="minutes of difference still counted as agreement")
    args = ap.parse_args()

    ref = load_reference()
    cris = load_cris()

    keys = set(ref) & set(cris)
    print(f"matched   : {len(keys):,} train-days present in BOTH "
          f"({len(set(cris) - set(ref)):,} CRIS-only, "
          f"{len(set(ref) - set(cris)):,} reference-only)")
    print()

    agree = defaultdict(int)
    disagree = defaultdict(int)
    big_diff = defaultdict(list)
    cris_missing_ref_has = defaultdict(int)
    ref_missing_cris_has = defaultdict(int)
    both_missing = defaultdict(int)
    n_pairs = 0

    for k in sorted(keys):
        rrecs, crecs = ref[k], cris[k]
        for serial, crow in crecs.items():
            rrec = rrecs.get(serial)
            if rrec is None:
                continue
            # Guard the join: same serial must mean same station.
            if str(rrec.get("station_code")) != str(crow.station_code):
                continue
            n_pairs += 1
            for cfield, rfield in FIELDS:
                cv = _ts(getattr(crow, cfield))
                rv = _ts(rrec.get(rfield))
                if cv is None and rv is None:
                    both_missing[cfield] += 1
                elif cv is None:
                    cris_missing_ref_has[cfield] += 1
                elif rv is None:
                    ref_missing_cris_has[cfield] += 1
                else:
                    diff = abs((cv - rv).total_seconds()) / 60.0
                    if diff <= args.tolerance:
                        agree[cfield] += 1
                    else:
                        disagree[cfield] += 1
                        if len(big_diff[cfield]) < 4:
                            big_diff[cfield].append(
                                (k[0], k[1], crow.station_code, cv, rv, diff))

    print(f"compared  : {n_pairs:,} station-rows on the same "
          f"(train, date, serial, station)")
    print()
    print("AGREEMENT where BOTH sources have a value "
          f"(tolerance {args.tolerance} min)")
    print(f"  {'field':<28} {'agree':>9} {'differ':>8} {'agree%':>8}")
    print("  " + "-" * 56)
    for cfield, _ in FIELDS:
        a, dgr = agree[cfield], disagree[cfield]
        tot = a + dgr
        pct = f"{a / tot:.2%}" if tot else "n/a"
        print(f"  {cfield:<28} {a:>9,} {dgr:>8,} {pct:>8}")

    print()
    print("GAP FILLING potential")
    print(f"  {'field':<28} {'CRIS empty,':>13} {'ref empty,':>12} "
          f"{'both':>8}")
    print(f"  {'':<28} {'ref HAS':>13} {'CRIS has':>12} {'empty':>8}")
    print("  " + "-" * 64)
    for cfield, _ in FIELDS:
        print(f"  {cfield:<28} {cris_missing_ref_has[cfield]:>13,} "
              f"{ref_missing_cris_has[cfield]:>12,} {both_missing[cfield]:>8,}")

    shown = False
    for cfield, _ in FIELDS:
        for ex in big_diff[cfield]:
            if not shown:
                print()
                print("EXAMPLES OF DISAGREEMENT (these decide trust)")
                shown = True
            tn, dt, stn, cv, rv, diff = ex
            print(f"  {cfield:<26} train {tn} {dt} {stn:>5}  "
                  f"CRIS {cv}  ref {rv}  ({diff:.0f} min)")
    if not shown:
        print()
        print("No disagreements at all where both sources have a value.")


if __name__ == "__main__":
    main()
