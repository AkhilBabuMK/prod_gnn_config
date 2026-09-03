# -*- coding: utf-8 -*-
"""
ONE-COMMAND PRODUCTION SETUP.

    export IRPILOT_SCHEMA=prod
    export IRPILOT_DB_HOST=...  IRPILOT_DB_NAME=...
    export IRPILOT_DB_USER=...  IRPILOT_DB_PASSWORD=...
    export IRPILOT_GNN_ROOT=/opt/irpilot/jabalpur_scenario_gnn
    python setup_prod.py

Creates the objects we write to, loads the station reference, and then CHECKS
EVERY ASSUMPTION we make about their database before the loop is ever started.
Safe to re-run: nothing is dropped, nothing of theirs is touched, every
statement is idempotent.

WHAT IT CREATES (ours)
    model_state     the carried GRU memory, one row per 3-minute tick
    forecast        one row per (issue time, train, station)
    infra_topology  the corridor graph, so a past forecast can always be
                    matched to the graph that produced it
    station_ref     station_id -> code and name, for the read views
    forecast_read / forecast_latest    the forecast in their terms

WHAT IT READS (theirs, never written)
    gnn_input_table, goods_train_running, goods_train_schedule,
    maintenance_block, asset_failure, temporary_speed_restriction

The checks matter more than the DDL. Every one corresponds to a real failure
found during integration, and most of those failures were SILENT: the loop kept
running and the forecasts quietly got worse.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import psycopg2
import config

DDL = """
CREATE TABLE IF NOT EXISTS model_state (
    tick_ts        TIMESTAMPTZ PRIMARY KEY,
    station_memory BYTEA NOT NULL,
    service_memory BYTEA NOT NULL,
    section_memory BYTEA NOT NULL,
    station_state  JSONB NOT NULL,
    prev_delay     JSONB NOT NULL,
    checkpoint_sha TEXT  NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_model_state_created ON model_state (created_at DESC);

CREATE TABLE IF NOT EXISTS forecast (
    issued_at      TIMESTAMPTZ NOT NULL,
    instance_id    TEXT        NOT NULL,
    station_id     INTEGER     NOT NULL,
    hop            INTEGER,
    lead_min       REAL,
    pred_delay_min REAL,
    lo80           REAL,
    hi80           REAL,
    model_version  TEXT,
    PRIMARY KEY (issued_at, instance_id, station_id)
);
CREATE INDEX IF NOT EXISTS ix_forecast_issued ON forecast (issued_at DESC);

CREATE TABLE IF NOT EXISTS infra_topology (
    version   TEXT PRIMARY KEY,
    topology  JSONB NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS station_ref (
    station_id    INTEGER PRIMARY KEY,
    station_code  VARCHAR(12) NOT NULL,
    station_name  TEXT,
    is_junction   BOOLEAN
);
"""

VIEWS = """
-- Dropped first, not CREATE OR REPLACE: replace cannot rename or reorder the
-- columns of an existing view, so an older definition would block the new one.
-- These two views are ours and hold no data, so dropping costs nothing.
DROP VIEW IF EXISTS forecast_latest;
DROP VIEW IF EXISTS forecast_read;

CREATE VIEW forecast_read AS
SELECT f.issued_at,
       f.instance_id,
       split_part(f.instance_id, '_', 1)                AS train_number,
       r.station_code,
       r.station_name,
       f.hop                                            AS stops_ahead,
       f.lead_min                                       AS minutes_ahead,
       f.issued_at + (f.lead_min * INTERVAL '1 minute') AS predicted_arrival,
       round(f.pred_delay_min::numeric, 1)              AS predicted_delay_min,
       -- A predicted arrival BEFORE the issue time means the train is already
       -- past the moment we expect it and has not reported.
       (f.lead_min < 0)                                 AS is_overdue,
       f.lo80, f.hi80, f.model_version
  FROM forecast f
  LEFT JOIN station_ref r ON r.station_id = f.station_id;

CREATE VIEW forecast_latest AS
SELECT * FROM forecast_read
 WHERE issued_at = (SELECT max(issued_at) FROM forecast);
"""

_ok = _bad = 0


def check(name, fn):
    global _ok, _bad
    try:
        detail = fn()
        _ok += 1
        print(f"  [OK]   {name:42} {detail}")
    except Exception as exc:
        _bad += 1
        print(f"  [FAIL] {name:42} {exc}")


def main() -> int:
    print("=" * 78)
    print(f"IRPILOT PRODUCTION SETUP   profile: {config.SCHEMA}")
    print(f"  database {config.DB['user']}@{config.DB['host']}:"
          f"{config.DB['port']}/{config.DB['dbname']}")
    print("=" * 78)

    if config.SCHEMA != "prod":
        print()
        print("  NOTE: IRPILOT_SCHEMA is not 'prod', so the table names checked")
        print("  below are the simulation ones. Set IRPILOT_SCHEMA=prod.")

    conn = psycopg2.connect(**config.DB)
    conn.autocommit = True

    # ── files we ship ───────────────────────────────────────────────────────
    print("\nFILES")

    def t_ckpt():
        p = Path(config.CHECKPOINT)
        if not p.exists():
            raise RuntimeError(f"missing: {p}")
        return f"{p.stat().st_size / 1e6:.1f} MB  {p.name}"
    check("checkpoint", t_ckpt)

    def t_topo():
        d = json.loads(Path(config.TOPOLOGY).read_text(encoding="utf-8"))
        return f"{len(d['stations'])} stations, {len(d['sections'])} sections"
    check("topology", t_topo)

    def t_ds():
        n = len(list(Path(config.DATASET_DIR).glob("*.json")))
        if n == 0:
            raise RuntimeError("no day files; one is needed to bootstrap "
                               "feature widths at boot")
        return f"{n} day file(s) — one is enough"
    check("dataset dir", t_ds)

    def t_side():
        # Reads their tables where available, the shipped CSVs otherwise, and
        # says which. Empty from both is fatal: a missing lookup does not raise
        # anywhere else, it just quietly costs ~14% of trains their coach count.
        from sim.build_journeys_db import load_side_tables
        coaches, platforms = load_side_tables(conn)
        from cris_pipeline.build_journeys import (
            load_side_tables as csv_side)
        c_csv, p_csv = csv_side()
        agree = (sum(1 for k in set(coaches) & set(c_csv)
                     if coaches[k] == c_csv[k]),
                 sum(1 for k in set(platforms) & set(p_csv)
                     if platforms[k] == p_csv[k]))
        return (f"{len(coaches)} trains, {len(platforms)} platforms "
                f"(agree with shipped CSV: {agree[0]}/{len(c_csv)}, "
                f"{agree[1]}/{len(p_csv)})")
    check("coach + platform lookup", t_side)

    # ── their tables ────────────────────────────────────────────────────────
    print("\nTHEIR TABLES (read-only)")
    need = {
        "main": ["train_number", "serial_number", "station_code",
                 "scheduled_arrival_time", "scheduled_departure_time",
                 "actual_arrival_time", "actual_departure_time",
                 "train_type", "train_sub_type", "train_source_station",
                 "train_destination_station", "wtt_stop_flag",
                 "ptt_stoppage_flag", "traffic_allowance_seconds",
                 "engineering_allowance_seconds", "distance_from_source_km"],
        "goods_running":  ["coatrainid", "sqncnumb", "station",
                           "arvltime", "dprttime"],
        "goods_schedule": ["coatrainid", "sqncnumb", "schdarvl", "schddprt"],
        "maintenance":    ["block_section", "permitted_start_time",
                           "permitted_end_time", "actual_clear_time"],
        "asset_failure":  ["block_section", "event_start_date",
                           "event_end_date", "affected_trains", "division"],
        "coaches":        ["train_number", "no_of_coaches"],
        "pf_info":        ["train_number", "station_code", "pf_number"],
    }
    for key, cols in need.items():
        tbl = config.NAMES[key]

        def t_tbl(tbl=tbl, cols=cols):
            with conn.cursor() as cur:
                cur.execute(f"SELECT {', '.join(cols)} FROM {tbl} LIMIT 1")
                cur.execute(f"SELECT count(*) FROM {tbl}")
                return f"{cur.fetchone()[0]:,} rows"
        check(f"{key} -> {tbl}", t_tbl)

    def t_tsr():
        with conn.cursor() as cur:
            cur.execute(f"""SELECT block_section, passenger_train_speed,
                                   {config.NAMES['tsr_from']},
                                   {config.NAMES['tsr_to']}
                              FROM {config.NAMES['tsr']} LIMIT 1""")
            cur.execute(f"SELECT count(*) FROM {config.NAMES['tsr']}")
            return f"{cur.fetchone()[0]:,} rows, date cast parses"
    check(f"tsr -> {config.NAMES['tsr']}", t_tsr)

    def t_changed():
        with conn.cursor() as cur:
            cur.execute(f"SELECT max({config.NAMES['changed_at']}) "
                        f"FROM {config.NAMES['main']}")
            return f"newest write {cur.fetchone()[0]}"
    check(f"changed_at -> {config.NAMES['changed_at']}", t_changed)

    def t_unique():
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT count(*) FROM (
                    SELECT 1 FROM {config.NAMES['main']}
                     GROUP BY train_number, {config.NAMES['train_date']},
                              serial_number
                    HAVING count(*) > 1 LIMIT 1) q""")
            if cur.fetchone()[0]:
                raise RuntimeError("DUPLICATE stop keys. A journey would be "
                                   "built from more than one row per stop.")
            return "one row per (train, date, serial)"
    check("stop key is unique", t_unique)

    def t_pad():
        with conn.cursor() as cur:
            cur.execute(f"""SELECT DISTINCT length(train_number)
                              FROM {config.NAMES['main']} ORDER BY 1""")
            lens = [r[0] for r in cur.fetchall()]
            return f"lengths {lens}; padded to 5 on read"
    check("train_number format", t_pad)

    # ── ours ────────────────────────────────────────────────────────────────
    print("\nOUR OBJECTS")
    with conn.cursor() as cur:
        cur.execute(DDL)
    check("tables created",
          lambda: "model_state, forecast, infra_topology, station_ref")

    def t_load_topo():
        topo = json.loads(Path(config.TOPOLOGY).read_text(encoding="utf-8"))
        version = str(topo.get("meta", {}).get("version", "v1"))
        with conn.cursor() as cur:
            cur.execute("DELETE FROM infra_topology WHERE version = %s",
                        (version,))
            cur.execute("INSERT INTO infra_topology (version, topology) "
                        "VALUES (%s, %s)", (version, json.dumps(topo)))
            cur.execute("DELETE FROM station_ref")
            for s in topo["stations"].values():
                cur.execute("""INSERT INTO station_ref
                                 (station_id, station_code, station_name,
                                  is_junction)
                               VALUES (%s, %s, %s, %s)""",
                            (int(s["station_id"]), s["code"], s.get("name"),
                             bool(s.get("is_junction"))))
        return f"version {version}, {len(topo['stations'])} stations"
    check("topology + station_ref loaded", t_load_topo)

    with conn.cursor() as cur:
        cur.execute(VIEWS)
    check("read views created", lambda: "forecast_read, forecast_latest")

    conn.close()

    print("=" * 78)
    print(f"  {_ok} passed, {_bad} failed")
    if _bad:
        print("  NOT READY — fix the failures above before starting the loop.")
    else:
        print("  READY.")
        print("    start:  python run_prod.py")
        print("    read:   SELECT * FROM forecast_latest;")
    print("=" * 78)
    return 1 if _bad else 0


if __name__ == "__main__":
    sys.exit(main())
