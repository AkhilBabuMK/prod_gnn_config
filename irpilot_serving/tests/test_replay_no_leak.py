# -*- coding: utf-8 -*-
"""
A replay must not be able to see the future.

This is the one failure mode that makes a run look BETTER, which is why it
needs a standing test rather than a careful reader. `run_prod.py --replay`
queries the same database production queries, and that database already holds
the whole month — so a plain read at 09:00 hands the model arrival times from
that afternoon and the accuracy figures come out flattering and worthless.

The fix reads observed values through `feed_chunk`, which carries the moment
CRIS reported each one. These tests pin that behaviour from both directions:
the mask must hide everything later than the replay clock, and it must hide
nothing earlier.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import psycopg2

import config
from sim.build_journeys_db import read_running_from_db, changed_trains

T = dt.datetime(2025, 9, 27, 9, 0)          # the replay instant under test
OBS = ["actual_arrival_time", "actual_departure_time"]

_passed = _failed = 0


def check(name, ok, detail=""):
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  [OK]   {name:52} {detail}")
    else:
        _failed += 1
        print(f"  [FAIL] {name:52} {detail}")


def n_future(df, col, when=T):
    return int((df[col] > pd.Timestamp(when)).sum())


def main() -> int:
    conn = psycopg2.connect(**config.DB)

    print("=" * 78)
    print(f"REPLAY LEAK GUARD   masking at {T}")
    print("=" * 78)

    masked = read_running_from_db(conn, as_of=T)
    plain = read_running_from_db(conn)

    # 1. the leak is real — this is what the mask is protecting against
    leaked = sum(n_future(plain, c) for c in OBS)
    check("an unmasked read really does see the future", leaked > 0,
          f"{leaked:,} observed values later than {T:%H:%M} in a plain read")

    # 2. and the mask removes all of it
    for c in OBS:
        check(f"masked read hides future {c}", n_future(masked, c) == 0,
              f"{n_future(masked, c)} future values")

    # 3. the mask must not be over-eager: it should return exactly what
    #    feed_chunk says had been reported, no fewer.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT count(actual_arrival_time), count(actual_departure_time)
              FROM (SELECT train_number, train_date, serial_number,
                           max(actual_arrival_time)   AS actual_arrival_time,
                           max(actual_departure_time) AS actual_departure_time
                      FROM feed_chunk WHERE reveal_time <= %s
                     GROUP BY 1, 2, 3) q""", (T,))
        want_a, want_d = cur.fetchone()
    got_a = int(masked.actual_arrival_time.notna().sum())
    got_d = int(masked.actual_departure_time.notna().sum())
    check("masked arrivals match the reveal log exactly", got_a == want_a,
          f"{got_a:,} vs {want_a:,}")
    check("masked departures match the reveal log exactly", got_d == want_d,
          f"{got_d:,} vs {want_d:,}")

    # 4. the same must hold on the incremental path, which is what actually
    #    runs every window — a filter that bypassed the mask would be invisible
    #    in the full-read tests above.
    changed = changed_trains(conn, T - dt.timedelta(minutes=5), until=T,
                             by_event_time=True)
    inc = read_running_from_db(conn, changed, as_of=T)
    check("incremental read still masks the future",
          sum(n_future(inc, c) for c in OBS) == 0,
          f"{len(changed)} trains, {len(inc):,} rows")

    # 5. the mask must grow monotonically with the clock — a later instant can
    #    only ever know more, never less.
    later = read_running_from_db(conn, as_of=T + dt.timedelta(hours=2))
    a_now = int(masked[OBS].notna().sum().sum())
    a_later = int(later[OBS].notna().sum().sum())
    check("two hours later, strictly more is known", a_later > a_now,
          f"{a_now:,} -> {a_later:,} observed values")

    # 6. production is NOT masked. Passing no as_of must return the table as it
    #    stands, because live the database IS the present.
    check("live path is untouched by the mask",
          int(plain[OBS].notna().sum().sum()) > a_later,
          "unmasked read still sees everything")

    conn.close()
    print("=" * 78)
    print(f"  {_passed} passed, {_failed} failed")
    if not _failed:
        print("  A REPLAY CANNOT SEE THE FUTURE.")
    print("=" * 78)
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
