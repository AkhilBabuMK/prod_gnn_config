# -*- coding: utf-8 -*-
"""
WHY setup_prod.py SAID NO.

    source env.sh
    python diagnose_prod.py

Reads only. Prints the actual rows behind each failing check, so the fix can be
decided from evidence instead of guesswork. Three things it looks at:

  DUPLICATE STOP KEYS   we need one row per (train, date, serial). Two rows for
                        one stop means a journey is built from both, and which
                        one wins is whatever order the planner returned.

  THE CHANGED-AT COLUMN which answers "what is new since I last looked?". If it
                        is NULL the incremental read returns nothing, forever,
                        and the forecaster silently stops seeing new data.

  TRAIN NUMBER SHAPE    every lookup key in the pipeline is 5 characters. We
                        pad shorter ones on read. Longer ones and NULLs cannot
                        be repaired that way and need a decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import psycopg2
import config

MAIN = config.NAMES["main"]
DATE = config.NAMES["train_date"]
CHANGED = config.NAMES["changed_at"]


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    conn = psycopg2.connect(**config.DB)
    # Read-only, and autocommit so we never hold an ACCESS SHARE lock on their
    # tables. Without it a long diagnostic would block a TRUNCATE or an ALTER
    # for as long as it runs.
    conn.autocommit = True
    cur = conn.cursor()
    print(f"reading {MAIN}  (nothing is written)")

    # ── 1. duplicates ───────────────────────────────────────────────────────
    section("1. DUPLICATE STOP KEYS")
    cur.execute(f"""
        SELECT count(*) AS dup_keys, sum(n) AS rows_involved
          FROM (SELECT count(*) AS n FROM {MAIN}
                 GROUP BY train_number, {DATE}, serial_number
                HAVING count(*) > 1) q""")
    keys, rows = cur.fetchone()
    cur.execute(f"SELECT count(*) FROM {MAIN}")
    total = cur.fetchone()[0]
    print(f"  stop keys appearing more than once : {keys or 0:,}")
    print(f"  rows involved                      : {rows or 0:,} of {total:,}"
          f"  ({100*(rows or 0)/total:.2f}%)")

    if keys:
        cur.execute(f"""
            SELECT train_number, {DATE} AS train_date, serial_number, count(*)
              FROM {MAIN}
             GROUP BY 1,2,3 HAVING count(*) > 1
             ORDER BY count(*) DESC, 1 LIMIT 5""")
        worst = cur.fetchall()
        print("\n  worst offenders:")
        for tn, td, sn, n in worst:
            print(f"     train {tn!r} date {td} serial {sn} -> {n} rows")

        # Do the duplicates actually DISAGREE, or are they identical copies?
        # That decides the fix: identical copies can simply be de-duplicated,
        # disagreeing rows need someone to say which is right.
        tn, td, sn, _ = worst[0]
        cur.execute(f"""
            SELECT station_code, scheduled_arrival_time, actual_arrival_time,
                   scheduled_departure_time, actual_departure_time
              FROM {MAIN}
             WHERE train_number = %s AND {DATE} = %s AND serial_number = %s""",
                    (tn, td, sn))
        print(f"\n  the {len(worst[0])>0 and worst[0][3]} rows for that one stop:")
        for r in cur.fetchall():
            print(f"     {r}")

        cur.execute(f"""
            SELECT count(*) FROM (
              SELECT train_number, {DATE} AS d, serial_number
                FROM {MAIN}
               GROUP BY 1,2,3
              HAVING count(*) > 1
                 AND (count(DISTINCT station_code) > 1
                   OR count(DISTINCT actual_arrival_time) > 1
                   OR count(DISTINCT scheduled_arrival_time) > 1)) q""")
        disagree = cur.fetchone()[0]
        print(f"\n  of those {keys:,} duplicated keys, {disagree:,} have rows that"
              f" DISAGREE on station or times.")
        print(f"  the other {keys - disagree:,} are identical copies.")

    # ── 2. the changed-at column ────────────────────────────────────────────
    section(f"2. THE CHANGED-AT COLUMN  ({CHANGED})")
    cur.execute(f"""SELECT count(*), count({CHANGED}),
                           min({CHANGED}), max({CHANGED}) FROM {MAIN}""")
    n, nn, lo, hi = cur.fetchone()
    print(f"  rows                  : {n:,}")
    print(f"  with a value          : {nn:,}  ({100*nn/n:.1f}%)")
    print(f"  range                 : {lo}  ->  {hi}")
    if nn == 0:
        print("\n  THIS IS FATAL FOR THE LIVE LOOP.")
        print("  Every 5 minutes we ask 'which trains changed since I last")
        print("  looked?' by comparing this column against the clock. All NULL")
        print("  means the answer is always 'none', so the forecaster would")
        print("  keep running on the journeys it built at boot and never see a")
        print("  single new report.")
        print("\n  Candidates to use instead:")
        for c in ("record_update_timestamp", "record_insert_time",
                  "arrival_time", "batch_id"):
            cur.execute("""SELECT 1 FROM information_schema.columns
                            WHERE table_schema = %s AND table_name = %s
                              AND column_name = %s""",
                        (*MAIN.split(".", 1), c) if "." in MAIN
                        else ("public", MAIN, c))
            if cur.fetchone():
                cur.execute(f"SELECT count({c}), min({c}), max({c}) FROM {MAIN}")
                cnt, mn, mx = cur.fetchone()
                print(f"     {c:26} {cnt:>9,} non-null   {mn} -> {mx}")

    # ── 3. train number shape ───────────────────────────────────────────────
    section("3. TRAIN NUMBER SHAPE")
    cur.execute(f"""SELECT length(train_number) AS len, count(*)
                      FROM {MAIN} GROUP BY 1 ORDER BY 1 NULLS LAST""")
    print("  length   rows        note")
    for ln, cnt in cur.fetchall():
        if ln is None:
            note = "NULL — cannot be keyed at all"
        elif ln < 5:
            note = "padded to 5 on read, fine"
        elif ln == 5:
            note = "fine"
        else:
            note = "LONGER than 5 — padding cannot fix this"
        print(f"    {str(ln):>5}  {cnt:>9,}   {note}")

    cur.execute(f"""SELECT DISTINCT train_number FROM {MAIN}
                     WHERE length(train_number) > 5 LIMIT 8""")
    long = [r[0] for r in cur.fetchall()]
    if long:
        print(f"\n  examples longer than 5: {long}")
    cur.execute(f"SELECT count(*) FROM {MAIN} WHERE train_number IS NULL")
    nulls = cur.fetchone()[0]
    if nulls:
        print(f"  rows with NO train number: {nulls:,}")

    conn.close()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
