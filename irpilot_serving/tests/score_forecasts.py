# -*- coding: utf-8 -*-
"""
Score the forecasts sitting in the database against what actually happened.

Answers one question: is the model still as good as it was, now that it runs
through the trigger-maintained main table with the causal overlays?

WHAT IS BEING COMPARED
----------------------
Each forecast row says "train T reaches station S at lead L, N minutes late".
Ground truth is that train's real arrival at that station, against its booked
time. Both come from main_table; the forecast used only what was observable at
its issue time, the truth uses everything.

WHAT THIS IS NOT
----------------
Not the published evaluation. That ran over held-out days with the full rollout
harness. This scores whatever happens to be in `forecast` — one window, one
day, from a live-style run. Useful as a health check and for spotting a
regression; not a number to quote.

    python irpilot_serving/tests/score_forecasts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2                                                  # noqa: E402
import config                                                    # noqa: E402

# The buckets the pipeline's own rollout reports, so the numbers line up.
BUCKETS = [(0, 30), (30, 60), (60, 120), (120, 240)]
SPIKE = 30.0


def main() -> int:
    conn = psycopg2.connect(**config.DB)
    cur = conn.cursor()

    # Forecast joined to truth. The join is on the train, the station and the
    # corridor date carried in instance_id — the date matters, because the same
    # train number runs every day and would otherwise score against another
    # day's arrival.
    cur.execute("""
        SELECT f.lead_min,
               f.pred_delay_min,
               EXTRACT(EPOCH FROM (m.actual_arrival_time
                                 - m.scheduled_arrival_time))/60.0 AS truth
          FROM forecast f
          JOIN station_ref r ON r.station_id = f.station_id
          JOIN main_table  m
            ON  m.train_number = split_part(f.instance_id, '_', 1)
            AND m.station_code = r.station_code
            AND m.scheduled_arrival_time::date
                = split_part(f.instance_id, '_', 2)::date
         WHERE m.actual_arrival_time    IS NOT NULL
           AND m.scheduled_arrival_time IS NOT NULL
    """)
    rows = [(float(a), float(b), float(t)) for a, b, t in cur.fetchall()]

    cur.execute("SELECT count(*), count(DISTINCT issued_at) FROM forecast")
    n_fc, n_iss = cur.fetchone()
    conn.close()

    if not rows:
        print("no forecasts could be matched to ground truth")
        return 1

    print("=" * 74)
    print("FORECAST ACCURACY — scored against what actually happened")
    print("=" * 74)
    print(f"\n  {n_fc:,} forecasts across {n_iss} issue times")
    print(f"  {len(rows):,} of them have a real arrival to score against\n")

    def mae(sub):
        return sum(abs(p - t) for _, p, t in sub) / len(sub) if sub else None

    print(f"  {'lead time':16}{'n':>7}{'MAE':>9}{'median':>9}   what it means")
    print("  " + "-" * 68)
    for lo, hi in BUCKETS:
        sub = [r for r in rows if lo <= r[0] < hi]
        if not sub:
            print(f"  {f'{lo}-{hi} min':16}{0:>7}        --        --")
            continue
        errs = sorted(abs(p - t) for _, p, t in sub)
        med = errs[len(errs) // 2]
        print(f"  {f'{lo}-{hi} min':16}{len(sub):>7}{mae(sub):>9.2f}{med:>9.2f}"
              f"   how far out, in minutes")
    allerr = sorted(abs(p - t) for _, p, t in rows)
    print("  " + "-" * 68)
    print(f"  {'ALL':16}{len(rows):>7}{mae(rows):>9.2f}"
          f"{allerr[len(allerr)//2]:>9.2f}")

    # Spikes are what the corridor actually cares about: does the model see a
    # train that is badly late, or does it smooth them away?
    spikes = [r for r in rows if r[2] >= SPIKE]
    caught = [r for r in spikes if r[1] >= SPIKE]
    print(f"\n  BIG DELAYS (30+ min late in reality)")
    print(f"    happened          {len(spikes):>6}")
    if spikes:
        print(f"    we called them    {len(caught):>6}   "
              f"({100*len(caught)/len(spikes):.0f}% recall)")
        print(f"    MAE on them       {mae(spikes):>6.2f} min")
    flat = [r for r in rows if r[2] < SPIKE]
    if flat:
        print(f"\n  NORMAL RUNNING (under 30 min late)")
        print(f"    n {len(flat):,}   MAE {mae(flat):.2f} min")

    print("\n" + "=" * 74)
    print("  Compare with care: the published 7.04 min came from the full")
    print("  rollout harness over held-out days. This is one window of one day")
    print("  from a live-style run, and the trains in it are not the same set.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
