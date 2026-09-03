# -*- coding: utf-8 -*-
"""
cris_pipeline/reference_merge.py
================================
Fill gaps in the official CRIS running data from the older reference export.

CRIS is authoritative and is NEVER overwritten. A reference value is used only
where CRIS has nothing, and only if it survives every validation check.

WHY THIS EXISTS
CRIS drops the scheduled time whenever a journey crosses midnight: 98.4% of
midnight-crossing journeys are missing at least one schedule, against 10.1% of
same-day journeys. Because a stop with no schedule has no computable delay, the
pipeline dropped it entirely -- so 65 trains vanished from the graph across
contiguous runs of stations, every night. Train 12792, a daily superfast, was
absent from MYR..PKRD on all 30 days. Trains like that are precisely the ones
that cause precedence holds for everyone else.

WHY IT IS SAFE
  * agreement where both sources have a value: 100.00% over 50,730 rows,
    zero disagreements at 0-minute tolerance
  * the reference is never empty where CRIS has a value (strict superset)
  * every fill is bracketed against CRIS's own timeline, and against the
    reference's internal ordering, and against a pace bound

THE REFERENCE'S OWN BUG
It transposes day and month at midnight: true 2025-09-02T00:00 is written
2025-02-09T00:00. All 531 such values are at exactly 00:00, all have day 9 or
10 (the true month), and all land back in range once swapped. Repair is applied
ONLY as a candidate: the swapped value must then pass every check that an
untouched value must pass. A repair is never trusted because it is a repair.

    from .reference_merge import ReferenceIndex
    idx = ReferenceIndex.load()
    df  = idx.fill(df)
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import CORRIDOR

REF_DIR = (Path(__file__).resolve().parent.parent / "old_data_reference_for_missing"
           / "20260116_073928_20260116_080059 (2)" / "20260116_073928")

TIME_COLS = ["scheduled_arrival_time", "actual_arrival_time",
             "scheduled_departure_time", "actual_departure_time"]

DATA_LO = pd.Timestamp("2025-08-31")
DATA_HI = pd.Timestamp("2025-10-03")
MAX_MIN_PER_STOP = 120.0


def _ts(v):
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "nan", "NaT", "None"):
        return None
    t = pd.to_datetime(s, errors="coerce")
    return None if pd.isna(t) else t


def swap_day_month(v: pd.Timestamp):
    """Candidate repair for the transposition. None when impossible."""
    try:
        return v.replace(month=v.day, day=v.month)
    except ValueError:
        return None


class ReferenceIndex:
    def __init__(self):
        # (train_no, start_date) -> serial -> {field: Timestamp}
        self.rows: dict[tuple, dict] = {}
        self.stats = defaultdict(int)

    @classmethod
    def load(cls) -> "ReferenceIndex":
        idx = cls()
        if not REF_DIR.exists():
            print(f"  ! reference not found at {REF_DIR} — no gap filling")
            return idx
        for fp in glob.glob(str(REF_DIR / "train_*.json")):
            try:
                d = json.loads(Path(fp).read_text(encoding="utf-8"))
            except Exception:
                continue
            tn = str(d.get("train_number", "")).strip().zfill(5)
            sd = str(d.get("train_start_date", "")).strip()
            try:
                day, mon, yr = sd.split("/")
                date = f"20{yr}-{mon}-{day}"
            except ValueError:
                continue
            recs = {}
            for s in d.get("stations", []):
                if s.get("station_code") not in CORRIDOR:
                    continue
                recs[int(s["serial_number"])] = {
                    c: _ts(s.get(c)) for c in TIME_COLS}
            if recs:
                idx.rows[(tn, date)] = recs
        print(f"  reference index: {len(idx.rows):,} corridor train-days")
        return idx

    # ── validation ────────────────────────────────────────────────────────────

    def _acceptable(self, v, col, serial, serials, cris_vals, ref_vals,
                    span_lo, span_hi):
        if v is None or not (DATA_LO <= v <= DATA_HI):
            return False
        if not (span_lo <= v <= span_hi):
            return False
        i = serials.index(serial)

        prev_c = next((cris_vals[serials[j]] for j in range(i - 1, -1, -1)
                       if cris_vals[serials[j]] is not None), None)
        next_c = next((cris_vals[serials[j]] for j in range(i + 1, len(serials))
                       if cris_vals[serials[j]] is not None), None)
        if prev_c is not None and v < prev_c:
            return False
        if next_c is not None and v > next_c:
            return False

        for anchor, rng in ((prev_c, range(i - 1, -1, -1)),
                            (next_c, range(i + 1, len(serials)))):
            if anchor is None:
                continue
            hops = 1
            for j in rng:
                if cris_vals[serials[j]] is not None:
                    break
                hops += 1
            if abs((v - anchor).total_seconds()) / 60.0 > MAX_MIN_PER_STOP * hops:
                return False

        prev_r = next((ref_vals[serials[j]] for j in range(i - 1, -1, -1)
                       if ref_vals[serials[j]] is not None), None)
        next_r = next((ref_vals[serials[j]] for j in range(i + 1, len(serials))
                       if ref_vals[serials[j]] is not None), None)
        if prev_r is not None and v < prev_r:
            return False
        if next_r is not None and v > next_r:
            return False
        return True

    # ── the merge ─────────────────────────────────────────────────────────────

    def fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing corridor timestamps in `df`, in place, and return it."""
        if not self.rows:
            return df
        df["ref_filled"] = 0
        corr = df[df.station_code.isin(CORRIDOR)]

        for (tn, dt), sub in corr.groupby(["train_no", "train_date"]):
            key = (str(tn), str(dt)[:10])
            recs = self.rows.get(key)
            if not recs:
                continue
            sub = sub.sort_values("serial_number")
            serials = [int(s) for s in sub.serial_number if int(s) in recs]
            if not serials:
                continue
            idx_by_serial = {int(r.serial_number): r.Index
                             for r in sub.itertuples()}

            cris_all = {c: {s: (None if pd.isna(df.at[idx_by_serial[s], c])
                                else df.at[idx_by_serial[s], c])
                            for s in serials} for c in TIME_COLS}
            span = [v for c in TIME_COLS for v in cris_all[c].values()
                    if v is not None]
            if not span:
                continue
            span_lo = min(span) - pd.Timedelta(days=1)
            span_hi = max(span) + pd.Timedelta(days=1)

            for col in TIME_COLS:
                cris_vals = cris_all[col]
                ref_vals = {s: recs[s].get(col) for s in serials}
                for s in serials:
                    if cris_vals[s] is not None:
                        continue
                    v = ref_vals[s]
                    if v is None:
                        continue
                    tag = "reference"
                    if not self._acceptable(v, col, s, serials, cris_vals,
                                            ref_vals, span_lo, span_hi):
                        # try the transposition, then re-validate identically
                        sw = swap_day_month(v)
                        if sw is None:
                            self.stats[f"{col}:unrepairable"] += 1
                            continue
                        ref_try = dict(ref_vals)
                        ref_try[s] = sw
                        if not self._acceptable(sw, col, s, serials, cris_vals,
                                                ref_try, span_lo, span_hi):
                            self.stats[f"{col}:rejected"] += 1
                            continue
                        v, tag = sw, "reference_repaired"
                    df.at[idx_by_serial[s], col] = v
                    cris_vals[s] = v
                    df.at[idx_by_serial[s], "ref_filled"] = 1
                    self.stats[f"{col}:{tag}"] += 1
        return df

    def report(self) -> None:
        if not self.stats:
            print("  reference merge: nothing filled")
            return
        print("  reference merge:")
        for col in TIME_COLS:
            a = self.stats.get(f"{col}:reference", 0)
            r = self.stats.get(f"{col}:reference_repaired", 0)
            x = (self.stats.get(f"{col}:rejected", 0)
                 + self.stats.get(f"{col}:unrepairable", 0))
            print(f"    {col:<28} +{a:>5,} direct  +{r:>4,} repaired  "
                  f"{x:>4,} rejected")
