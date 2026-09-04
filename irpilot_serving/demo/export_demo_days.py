# -*- coding: utf-8 -*-
"""
EXPORT DEMO DAYS  —  runs HERE, on the machine with the good data.

    python demo/export_demo_days.py 2025-09-01
    python demo/export_demo_days.py 2025-09-01 2025-09-27 -o demo/days

Writes one plain CSV per corridor date. That file is the ONLY thing that
travels: the loader on their side reads it and needs nothing else from us.

Deliberately NOT compressed. A day is only a couple of megabytes, and a plain
text file survives git, email and copy-paste without a line-ending setting
quietly corrupting it — which a .gz does not.

WHY A FILE AND NOT THEIR TABLES
Their `test.train_schedule` and `test.train_running` cannot be trusted for a
demo — the history a forecast depends on is not reliably in them, and a model
that cannot see which trains are already late will confidently call a train
that is thirty minutes down as running on time. Everything else they hold
(freight, maintenance, asset failures, speed restrictions, coaches, platforms)
is sound and is read from their database as normal. Only the train feed is
substituted.

WHAT IS IN THE FILE
Exactly the columns the journey builder reads, and nothing else — no internal
ids, no derived values, no model output. Times are ISO strings so the file is
readable and reloads without a timezone guess.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
import config

COLUMNS = [
    "train_number", "train_date", "serial_number", "station_code",
    "scheduled_arrival_time", "scheduled_departure_time",
    "actual_arrival_time", "actual_departure_time",
    "train_type", "train_sub_type",
    "train_source_station", "train_destination_station",
    "wtt_stop_flag", "ptt_stoppage_flag",
    "traffic_allowance_seconds", "engineering_allowance_seconds",
    "distance_from_source_km",
]

QUERY = f"""
    SELECT {', '.join(COLUMNS)}
      FROM {{main}}
     WHERE train_date = %s
     ORDER BY train_number, serial_number
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dates", nargs="+", help="corridor dates, YYYY-MM-DD")
    ap.add_argument("-o", "--out", default="demo/days",
                    help="directory to write into (default demo/days)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(**config.DB)
    conn.autocommit = True

    print("=" * 74)
    print(f"EXPORT DEMO DAYS   from {config.NAMES['main']}  ->  {out}")
    print("=" * 74)

    total = 0
    for date in args.dates:
        with conn.cursor() as cur:
            cur.execute(QUERY.format(main=config.NAMES["main"]), (date,))
            rows = cur.fetchall()
        if not rows:
            print(f"  {date}: NO ROWS — skipped")
            continue

        path = out / f"{date}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(COLUMNS)
            for r in rows:
                w.writerow(["" if v is None else v for v in r])

        observed = sum(1 for r in rows if r[6] is not None)
        trains = len({r[0] for r in rows})
        total += len(rows)
        print(f"  {date}: {len(rows):>6,} stops | {observed:>6,} observed "
              f"| {trains:>3} trains | {path.stat().st_size/1e6:.2f} MB")

    conn.close()
    print("-" * 74)
    print(f"  {total:,} rows written")
    print()
    print("  Copy the whole directory across, then on their machine:")
    print("      python demo/feed_loader.py --day-file demo/days/<date>.csv \\")
    print("             --from 15:54 --hours 3 --speed 30 --every 0.1667")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
