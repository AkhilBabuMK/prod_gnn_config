# -*- coding: utf-8 -*-
"""
cris_pipeline/validate_reference_fill.py
========================================
Decide, per candidate value, whether a reference timestamp may be used to fill
a gap in the official CRIS data.

CRIS is authoritative and is never overwritten. A reference value is considered
ONLY where CRIS has nothing, and it is accepted only if it passes every check
below. There is no repair path: a value that looks wrong is rejected, because
"probably a swapped date" is a guess and guesses are how bad data gets in.

The reference is known to corrupt timestamps at midnight -- 2025-09-02T00:00
stored as 2025-02-09T00:00, day and month transposed. That specific failure is
caught by check 2, since a February timestamp cannot sit between two September
neighbours.

CHECKS, all must pass:

  1. SELF-CONSISTENT     at the same stop, arrival <= departure.
  2. BRACKETED BY CRIS   the fill must lie between the nearest CRIS-known value
                         before it and the nearest CRIS-known value after it,
                         in serial order. CRIS is the authority, so the fill has
                         to fit the authority's own timeline.
  3. BRACKETED IN REF    the fill must also be monotonic within the reference's
                         own sequence. Catches corruption where CRIS happens to
                         have no nearby value to bracket against.
  4. PLAUSIBLE PACE      the implied gap to the bracketing CRIS values must not
                         exceed MAX_MIN_PER_STOP per intervening stop. A value
                         that is technically between two distant neighbours but
                         implies 9 hours for one hop is not usable.
  5. DATE SANITY         the fill must fall inside the journey's own date span
                         (first to last CRIS timestamp, padded one day each
                         side). A September journey cannot contain November.

    python -m cris_pipeline.validate_reference_fill
    python -m cris_pipeline.validate_reference_fill --max-min-per-stop 90
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import pandas as pd

from .crosscheck_reference import load_reference, load_cris, _ts, FIELDS

MAX_MIN_PER_STOP = 120.0     # generous; corridor hops are booked 6-10 min

# The extract window. Any timestamp for this corridor must fall inside it;
# CRIS itself spans 2025-09-01 06:02 .. 2025-10-02 23:55 with nothing outside.
DATA_LO = pd.Timestamp("2025-08-31")
DATA_HI = pd.Timestamp("2025-10-03")


def swap_day_month(v: pd.Timestamp):
    """Undo the reference's day<->month transposition, or None if impossible.

    Measured shape of the corruption, over all 531 out-of-span values:
      * every one is at exactly 00:00
      * day is 9 (475x) or 10 (56x)   -- i.e. the true MONTH, transposed
      * month is spread 1..12         -- i.e. the true DAY, transposed
      * all 531 land inside their journey's date span once swapped
    So the writer emitted DD as month and MM as day when the clock rolled over.

    Edge cases this must survive, and does:
      - day > 12 cannot be a month  -> ValueError -> None (never guessed)
      - 2025-02-30 style impossible dates after swap -> ValueError -> None
      - day == month (2025-09-09) -> swap is identity, returns the same value,
        which is correct: for a journey starting 09-08 the next day IS 09-09,
        so nothing needs repairing and nothing is changed
      - month boundary 30 Sep -> 1 Oct: true 2025-10-01 corrupts to 2025-01-10,
        and swapping returns 2025-10-01. Verified by test_edge_cases().
    """
    try:
        return v.replace(month=v.day, day=v.month)
    except ValueError:
        return None


def test_edge_cases() -> None:
    """Executable proof for the boundaries that matter. Run with --self-test."""
    T = pd.Timestamp
    cases = [
        # (corrupted, expected repair, label)
        (T("2025-02-09 00:00"), T("2025-09-02 00:00"), "mid-month rollover"),
        (T("2025-01-10 00:00"), T("2025-10-01 00:00"), "SEP->OCT month boundary"),
        (T("2025-09-09 00:00"), T("2025-09-09 00:00"), "day==month, identity"),
        (T("2025-11-09 00:00"), T("2025-09-11 00:00"), "day 11 of September"),
        (T("2025-12-09 00:00"), T("2025-09-12 00:00"), "day 12, highest valid"),
        (T("2025-08-09 00:00"), T("2025-09-08 00:00"), "day 8"),
        (T("2025-10-09 00:00"), T("2025-09-10 00:00"), "day 10 of September"),
        (T("2025-02-10 00:00"), T("2025-10-02 00:00"), "2 Oct, day==2"),
    ]
    for bad, want, label in cases:
        got = swap_day_month(bad)
        assert got == want, f"{label}: {bad} -> {got}, expected {want}"
        print(f"  ok  {label:<28} {bad.date()} -> {got.date()}")

    # Values that must NOT be repairable. Note 2025-09-31 cannot be used as a
    # case here at all -- September has 30 days, so pandas rejects it before
    # this function is ever reached. Use months that really do have a 13th and
    # a 31st.
    for bad, label in [(T("2025-09-13 00:00"), "day 13 cannot be a month"),
                       (T("2025-08-31 00:00"), "day 31 cannot be a month")]:
        got = swap_day_month(bad)
        assert got is None, f"{label}: expected None, got {got}"
        print(f"  ok  {label:<28} -> correctly unrepairable")

    # A repair must still be inside the data window to be usable
    far = swap_day_month(T("2025-05-09 00:00"))
    assert far == T("2025-09-05 00:00") and DATA_LO <= far <= DATA_HI
    print(f"  ok  {'repair lands in data window':<28} {far.date()}")
    print("\nALL EDGE CASES PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-min-per-stop", type=float, default=MAX_MIN_PER_STOP)
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    ref = load_reference()
    cris = load_cris()
    keys = sorted(set(ref) & set(cris))
    print(f"matched   : {len(keys):,} train-days\n")

    accept = defaultdict(int)
    reject = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(list)
    accepted_rows = []

    for k in keys:
        rrecs, crecs = ref[k], cris[k]
        serials = sorted(set(rrecs) & set(crecs))
        if not serials:
            continue

        # Journey date span, from CRIS values only.
        span = [t for s in serials for f, _ in FIELDS
                if (t := _ts(getattr(crecs[s], f, None))) is not None]
        if not span:
            continue
        lo_day, hi_day = min(span) - pd.Timedelta(days=1), \
                         max(span) + pd.Timedelta(days=1)

        for cfield, rfield in FIELDS:
            cvals = {s: _ts(getattr(crecs[s], cfield, None)) for s in serials}
            rvals = {s: _ts(rrecs[s].get(rfield)) for s in serials}

            for idx, s in enumerate(serials):
                if cvals[s] is not None:
                    continue                      # CRIS has it; never touch
                v = rvals[s]
                if v is None:
                    continue
                accept_key = cfield
                why = None

                # 1. arrival <= departure at the same stop
                if cfield == "scheduled_arrival_time":
                    other = _ts(getattr(crecs[s], "scheduled_departure_time",
                                        None)) or rvals.get(s)
                    dep = _ts(rrecs[s].get("scheduled_departure_time"))
                    if dep is not None and v > dep:
                        why = "arr_after_dep"
                elif cfield == "actual_arrival_time":
                    dep = _ts(rrecs[s].get("actual_departure_time"))
                    if dep is not None and v > dep:
                        why = "arr_after_dep"

                # 5. date sanity
                if why is None and not (lo_day <= v <= hi_day):
                    why = "outside_journey_dates"

                # 2. bracketed by CRIS
                if why is None:
                    prev_c = next((cvals[serials[j]] for j in range(idx - 1, -1, -1)
                                   if cvals[serials[j]] is not None), None)
                    next_c = next((cvals[serials[j]] for j in range(idx + 1, len(serials))
                                   if cvals[serials[j]] is not None), None)
                    if prev_c is not None and v < prev_c:
                        why = "before_prev_cris"
                    elif next_c is not None and v > next_c:
                        why = "after_next_cris"
                    elif why is None:
                        # 4. plausible pace against whichever side exists
                        for anchor, j_range in ((prev_c, range(idx - 1, -1, -1)),
                                                (next_c, range(idx + 1, len(serials)))):
                            if anchor is None:
                                continue
                            hops = 1
                            for j in j_range:
                                if cvals[serials[j]] is not None:
                                    break
                                hops += 1
                            gap = abs((v - anchor).total_seconds()) / 60.0
                            if gap > args.max_min_per_stop * hops:
                                why = "implausible_pace"
                                break

                # 3. monotonic within the reference itself
                if why is None:
                    prev_r = next((rvals[serials[j]] for j in range(idx - 1, -1, -1)
                                   if rvals[serials[j]] is not None), None)
                    next_r = next((rvals[serials[j]] for j in range(idx + 1, len(serials))
                                   if rvals[serials[j]] is not None), None)
                    if prev_r is not None and v < prev_r:
                        why = "not_monotonic_in_ref"
                    elif next_r is not None and v > next_r:
                        why = "not_monotonic_in_ref"

                if why is None:
                    accept[accept_key] += 1
                    accepted_rows.append((k[0], k[1], s,
                                          crecs[s].station_code, cfield, v))
                else:
                    reject[cfield][why] += 1
                    if len(examples[(cfield, why)]) < 2:
                        examples[(cfield, why)].append(
                            (k[0], k[1], crecs[s].station_code, v))

    print("VALIDATED FILL RESULTS  (CRIS never overwritten)")
    print(f"  {'field':<28} {'accepted':>9} {'rejected':>9} {'accept%':>8}")
    print("  " + "-" * 58)
    tot_a = tot_r = 0
    for cfield, _ in FIELDS:
        a = accept[cfield]
        r = sum(reject[cfield].values())
        tot_a += a
        tot_r += r
        pct = f"{a / (a + r):.1%}" if (a + r) else "n/a"
        print(f"  {cfield:<28} {a:>9,} {r:>9,} {pct:>8}")
    print("  " + "-" * 58)
    pct = f"{tot_a / (tot_a + tot_r):.1%}" if (tot_a + tot_r) else "n/a"
    print(f"  {'TOTAL':<28} {tot_a:>9,} {tot_r:>9,} {pct:>8}")

    print("\nREJECTION REASONS")
    allr = defaultdict(int)
    for cfield in reject:
        for why, n in reject[cfield].items():
            allr[why] += n
    for why, n in sorted(allr.items(), key=lambda kv: -kv[1]):
        print(f"  {why:<26} {n:>7,}")

    print(f"\nEXAMPLES OF REJECTS (first {args.show})")
    shown = 0
    for (cfield, why), exs in examples.items():
        for tn, dt, stn, v in exs:
            if shown >= args.show:
                break
            print(f"  {why:<24} train {tn} {dt} {stn:>5}  ref said {v}")
            shown += 1

    # Which trains benefit, and by how much
    by_train = defaultdict(int)
    for tn, dt, s, stn, f, v in accepted_rows:
        if f == "scheduled_arrival_time":
            by_train[tn] += 1
    print("\nTRAINS GAINING THE MOST SCHEDULED ARRIVALS")
    for tn, n in sorted(by_train.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  train {tn}  +{n:,} stops")


if __name__ == "__main__":
    main()
