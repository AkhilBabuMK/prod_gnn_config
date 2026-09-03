# -*- coding: utf-8 -*-
"""
STAND-IN FOR THE RAILWAY TEAM'S LOADER.

Reproduces, on our own database, exactly what they will do:

  1. load the whole day's TIMETABLE up front, actuals NULL
  2. stream OBSERVATIONS in every few minutes, filling those actuals
  3. stamp `arrival_time = now()` on every row it writes

and it builds the table in THEIR shape, not ours — `train_date` as DATE,
`serial_number` as NUMERIC, and `train_number` UNPADDED ('1053', not '01053').
So a demo run against it exercises the real production mapping, not a
convenient version of it.

    python demo/feed_loader.py                     # 1x, real time
    python demo/feed_loader.py --speed 6           # 6 September min per real min
    python demo/feed_loader.py --from 09:00 --hours 2

Point the forecaster at it with:

    IRPILOT_PROFILE=prod IRPILOT_TBL_MAIN=demo_gnn_input_table \
    IRPILOT_TBL_GOODS_RUNNING=goods_running \
    IRPILOT_TBL_GOODS_SCHEDULE=goods_schedule \
    IRPILOT_TBL_MAINTENANCE=maintenance \
    IRPILOT_TBL_ASSET_FAILURE=asset_failure \
    IRPILOT_TBL_TSR=tsr IRPILOT_TBL_TSR_FROM=from_datetime \
    IRPILOT_TBL_TSR_TO=to_datetime \
    python run_prod.py --follow

WHY THE TIMETABLE GOES IN FIRST. A journey needs booked times for stops the
train has not reached yet — that is what a forecast is measured against.
Loading rows only as events happen gave 11 journeys instead of 96 and zero
forecasts, with no error anywhere.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
import config

TABLE = "demo_gnn_input_table"

DDL = f"""
DROP TABLE IF EXISTS {TABLE};
CREATE TABLE {TABLE} (
    train_number                  VARCHAR,
    train_date                    DATE,
    serial_number                 NUMERIC,
    station_code                  VARCHAR,
    scheduled_arrival_time        TIMESTAMP,
    scheduled_departure_time      TIMESTAMP,
    actual_arrival_time           TIMESTAMP,
    actual_departure_time         TIMESTAMP,
    train_type                    VARCHAR,
    train_sub_type                VARCHAR,
    train_source_station          VARCHAR,
    train_destination_station     VARCHAR,
    wtt_stop_flag                 NUMERIC,
    ptt_stoppage_flag             NUMERIC,
    traffic_allowance_seconds     NUMERIC,
    engineering_allowance_seconds NUMERIC,
    distance_from_source_km       NUMERIC,
    arrival_time                  TIMESTAMPTZ,
    PRIMARY KEY (train_number, train_date, serial_number)
);
CREATE INDEX ON {TABLE} (arrival_time);
"""

# The timetable. Actuals are withheld — that is the whole point.
TIMETABLE = f"""
INSERT INTO {TABLE}
SELECT ltrim(train_number,'0'), train_date::date, serial_number, station_code,
       scheduled_arrival_time, scheduled_departure_time,
       NULL, NULL,
       train_type, train_sub_type, train_source_station,
       train_destination_station, wtt_stop_flag, ptt_stoppage_flag,
       traffic_allowance_seconds, engineering_allowance_seconds,
       distance_from_source_km, now()
  FROM main_table
"""

REVEAL = f"""
UPDATE {TABLE} d
   SET actual_arrival_time   = m.actual_arrival_time,
       actual_departure_time = m.actual_departure_time,
       arrival_time          = now()
  FROM main_table m
 WHERE d.train_number  = ltrim(m.train_number,'0')
   AND d.train_date    = m.train_date::date
   AND d.serial_number = m.serial_number
   AND GREATEST(m.actual_arrival_time, m.actual_departure_time) >  %s
   AND GREATEST(m.actual_arrival_time, m.actual_departure_time) <= %s
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2025-09-27")
    ap.add_argument("--from", dest="frm", default="06:00",
                    help="September time to start streaming from")
    ap.add_argument("--hours", type=float, default=8.0,
                    help="how many September hours to stream")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="September minutes per real minute (1 = real time)")
    ap.add_argument("--every", type=float, default=5.0,
                    help="real minutes between pushes")
    args = ap.parse_args()

    day = dt.date.fromisoformat(args.date)
    hh, mm = (int(x) for x in args.frm.split(":"))
    start = dt.datetime.combine(day, dt.time(hh, mm))

    step_min = args.every * args.speed        # September minutes per push
    cycles = int((args.hours * 60) / step_min)
    sleep_s = args.every * 60

    c = psycopg2.connect(**config.DB); c.autocommit = True
    cur = c.cursor()

    print("=" * 74)
    print(f"FEED LOADER (stand-in for the railway team's loader)")
    print(f"  table {TABLE}  |  September {args.date} from {args.frm}")
    print(f"  {args.speed:g}x speed | push every {args.every:g} real min "
          f"= {step_min:g} September min | {cycles} pushes")
    print("=" * 74)

    cur.execute(DDL)
    cur.execute(TIMETABLE)
    print(f"  timetable loaded: {cur.rowcount:,} stops, actuals withheld")

    # everything before the start time is already history by the time we begin
    cur.execute(REVEAL, (dt.datetime(2000, 1, 1), start))
    print(f"  warm start: {cur.rowcount:,} stops already reported "
          f"up to September {args.frm}")
    print()

    for i in range(cycles):
        lo = start + dt.timedelta(minutes=step_min * i)
        hi = lo + dt.timedelta(minutes=step_min)
        cur.execute(REVEAL, (lo, hi))
        print(f"  [{dt.datetime.now():%H:%M:%S}] +{cur.rowcount:5d} stops "
              f"-> September {hi:%H:%M}", flush=True)
        if i < cycles - 1:
            time.sleep(sleep_s)

    print("\n  loader finished.")
    c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
