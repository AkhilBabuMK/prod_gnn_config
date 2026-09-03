# -*- coding: utf-8 -*-
"""
irpilot_serving/config.py
=========================
Single place for everything environment-specific.

DESIGN RULE FOR THIS WHOLE PACKAGE
----------------------------------
This package is a CONSUMER of the GNN project. It imports from it and never
writes to it. Nothing under jabalpur_scenario_gnn/ is modified, ever. If a
change there ever becomes necessary, that is a signal the boundary is wrong.

Everything is read from the environment so the same code runs against the
simulation database today and a production database later with no edits.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Where the GNN project lives (read-only dependency) ────────────────────────
# Default assumes irpilot_serving/ sits beside jabalpur_scenario_gnn/.
GNN_ROOT = Path(os.getenv(
    "IRPILOT_GNN_ROOT",
    Path(__file__).resolve().parent.parent / "jabalpur_scenario_gnn",
)).resolve()


def bootstrap_gnn_path() -> None:
    """Put the GNN project on sys.path so `cris_pipeline` and the top-level
    `scenario_events` module both import.

    scenario_events.py lives at the GNN PROJECT ROOT, not inside cris_pipeline,
    and dataset_cris.py imports it absolutely (`from scenario_events import`).
    Miss this and the package fails at import time, not at run time.
    """
    p = str(GNN_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Database ──────────────────────────────────────────────────────────────────
DB = {
    "host":     os.getenv("IRPILOT_DB_HOST", "localhost"),
    "port":     int(os.getenv("IRPILOT_DB_PORT", "5432")),
    "dbname":   os.getenv("IRPILOT_DB_NAME", "irpilot_sim"),
    "user":     os.getenv("IRPILOT_DB_USER", "postgres"),
    "password": os.getenv("IRPILOT_DB_PASSWORD", os.getenv("PGPASSWORD", "root")),
}

# The maintenance database used only to CREATE the simulation database.
DB_ADMIN = dict(DB, dbname=os.getenv("IRPILOT_DB_ADMIN", "postgres"))

# ── Model artefacts (read-only) ───────────────────────────────────────────────
# v13a — the checkpoint the published head-to-head actually evaluated
# (corridor_v13a.pt, sha f941fb83a91105dc, epoch 22).
#
# This previously defaulted to model_v13b_epoch18.pt, which appears in NO
# published comparison and sits in the low-seed cluster: spike recall ~72-74%
# against v13a/v3's ~82%. That was never a deliberate choice, and it is why
# serving numbers looked worse than the published ones.
CHECKPOINT = Path(os.getenv(
    "IRPILOT_CHECKPOINT",
    GNN_ROOT / "releases" / "cris-v13" / "model_v13a_epoch22.pt"))
TOPOLOGY = Path(os.getenv(
    "IRPILOT_TOPOLOGY", GNN_ROOT / "data_cris" / "topology_jbp_sta_cris.json"))

# CrisDataset is constructed at boot ONLY to bootstrap feature widths. Every
# index it exposes — service_to_idx (301), station, section, class, group —
# comes from the topology and is byte-identical whether the directory holds all
# 30 day files or one; measured. But it refuses to construct on an empty
# directory, so exactly one day file must ship.
#
# This is env-driven because the full directory is 450 MB against 14 MB for a
# single file, and it is dead weight in a deployment: nothing reads it at
# runtime, the live data comes from Postgres.
DATASET_DIR = Path(os.getenv(
    "IRPILOT_DATASET_DIR", GNN_ROOT / "data_cris" / "dataset_v13"))

# ── Source CSVs, used ONLY to seed the simulation database ────────────────────
# In production these do not exist; CRIS lands files and their loader fills the
# same tables. Nothing downstream of the database knows the difference.
DATA_ROOT = GNN_ROOT / "data_standardized_timestamp_all"
CRIS_DIR = (DATA_ROOT / "cris_data_standardized_timestamp"
            / "3_cris_data_all_cleaned_files_standardized")

SRC = {
    "train_running":  DATA_ROOT / "train_running_clean_clean.csv",
    "goods_running":  DATA_ROOT / "goodsrunning_data_clean.csv",
    "station":        CRIS_DIR / "track_infra_station.csv",
    "blocksctn":      CRIS_DIR / "track_infra_blocksctn.csv",
    "sttnline":       CRIS_DIR / "track_infra_sttnline.csv",
    "platform":       CRIS_DIR / "track_infra_platform.csv",
    "psr":            CRIS_DIR / "track_infra_block_psr.csv",
    "maintenance":    CRIS_DIR / "maintenance_block.csv",
    "tsr":            CRIS_DIR / "temporary_speed_restriction.csv",
    "asset_failure":  CRIS_DIR / "asset_failure_details.csv",
    "coaches":        CRIS_DIR / "no_of_coaches_data.csv",
}

# ── Where the data actually lives ─────────────────────────────────────────────
# Their production database holds everything we need, but under its own table
# names, across two schemas, and with three columns that differ from ours.
# Rather than create views in their database or fork the code, the names are
# looked up here and substituted into the queries.
#
# IRPILOT_PROFILE=sim   (default) our seeded simulation database
# IRPILOT_PROFILE=prod            their database, untouched
#
# Only names appear here — never a value, never a filter. These strings are
# interpolated into SQL, so nothing user-supplied may ever reach this dict.
_SIM_NAMES = {
    "main":           "main_table",
    "goods_running":  "goods_running",
    "goods_schedule": "goods_schedule",
    "maintenance":    "maintenance",
    "asset_failure":  "asset_failure",
    "tsr":            "tsr",
    "coaches":        "coaches",
    "pf_info":        "pf_info",
    # Replay only: the reported-at log. No production counterpart — live, the
    # observation table IS the present and there is nothing to reveal.
    "feed_chunk":     "feed_chunk",
    # column expressions
    "train_date":     "train_date",
    "changed_at":     "changed_at",
    "tsr_from":       "from_datetime",
    "tsr_to":         "to_datetime",
}

_PROD_NAMES = {
    "main":           "data.gnn_input_table",
    "goods_running":  "test.goods_train_running",
    "goods_schedule": "test.goods_train_schedule",
    "maintenance":    "data.maintenance_block",
    "asset_failure":  "test.asset_failure",
    "tsr":            "data.temporary_speed_restriction",

    # Coach counts and platform allocations. The pipeline reads these from two
    # CSVs; production reads their live tables instead, because a shipped CSV
    # is a snapshot that silently ages — a rake gets shorter, a platform moves,
    # and we keep answering with last July's value. The CSVs stay as fallback.
    "coaches":        "test.coaches",
    "pf_info":        "test.pf_info",
    "feed_chunk":     "feed_chunk",      # replay only; absent in production

    # train_date is DATE there and text here; ::text gives 'YYYY-MM-DD',
    # which is the format build_journey expects.
    "train_date":     "train_date::text",

    # They have no changed_at. `arrival_time` is when the row landed in their
    # database — the same question ours answers. NOT record_update_timestamp:
    # that is when CRIS updated the event, so a backdated correction would
    # never be picked up by an incremental read.
    "changed_at":     "arrival_time",

    # UNVERIFIED CAST: todatetime is stored VARCHAR there. ::timestamp is
    # correct only if the text is ISO-like. If it is dd/mm/yyyy this misreads
    # the day as the month below the 13th and throws above it. Confirm one
    # sample value before trusting this in production.
    "tsr_from":       "fromdatetime",
    "tsr_to":         "todatetime::timestamp",
}

# ── Where WE are allowed to write ────────────────────────────────────────────
# Their database splits read from write: `test` holds the feeds and is
# read-only to us, `data` is where we may create things. Our four tables and
# two views must therefore be schema-qualified, or they land wherever
# search_path happens to point — `public` if we are lucky, a permission error
# if we are not, and worst of all a table in the wrong schema that everything
# then silently reads from.
#
# Empty means "use search_path", which is right for a single-schema database
# like the simulation one.
WRITE_SCHEMA = os.getenv("IRPILOT_WRITE_SCHEMA", "").strip().strip(".")


def out(name: str) -> str:
    """Qualify one of OUR objects with the write schema."""
    return f"{WRITE_SCHEMA}.{name}" if WRITE_SCHEMA else name


# Every object we create or write. Named here so there is one list to audit
# against a permissions grant, and one place to change if a name collides.
OUT = {n: out(n) for n in (
    "forecast", "model_state", "infra_topology", "station_ref",
    "forecast_read", "forecast_latest",
)}

# Any of ours can be renamed without touching code, the same way their tables
# can. Their forecast table is called forecast_output:
#     IRPILOT_OUT_FORECAST=forecast_output
# A bare name is qualified with WRITE_SCHEMA; a name containing a dot is taken
# exactly as given, in case one object has to live somewhere else entirely.
for _k in list(OUT):
    _v = os.getenv("IRPILOT_OUT_" + _k.upper())
    if _v:
        OUT[_k] = _v if "." in _v else out(_v)

# WHICH SET OF TABLE NAMES TO USE. This is NOT a Postgres schema — their
# database has real schemas called `data` and `test`, and calling this one
# "schema" too was a mistake worth undoing. It selects a naming profile:
#   sim   the development database, everything unqualified
#   prod  their database, each table qualified with the schema it lives in
# Where WE create things is a separate setting entirely: IRPILOT_WRITE_SCHEMA.
PROFILE = os.getenv("IRPILOT_PROFILE",
                    os.getenv("IRPILOT_SCHEMA", "sim"))   # old name still works
if PROFILE not in ("sim", "prod"):
    raise SystemExit(f"IRPILOT_PROFILE must be 'sim' or 'prod', got {PROFILE!r}")
NAMES = dict(_PROD_NAMES if PROFILE == "prod" else _SIM_NAMES)

# Any single name can be overridden without touching this file, so one table
# renamed on their side does not require a code change:
#     IRPILOT_TBL_MAIN=data.gnn_input_v2
#     IRPILOT_TBL_CHANGED_AT=loaded_at
for _k in list(NAMES):
    _v = os.getenv("IRPILOT_TBL_" + _k.upper())
    if _v:
        NAMES[_k] = _v


# ── Simulation clock ──────────────────────────────────────────────────────────
TICK_MIN = 3          # model cadence, matches build_dataset.SNAPSHOT_SCHEDULE
# How often to ask the database what is new. 5 matches how often CRIS delivers.
# Env-driven so a slower feed can be matched without a code change, and so the
# cost of a late feed can be measured rather than assumed.
FEED_MIN = int(os.getenv("IRPILOT_FEED_MIN", "5"))

# How far BEFORE the last read to look again, every window.
#
# The incremental read asks "what changed since I last looked?" and filters on
# a timestamp. Ours (changed_at) is stamped by the database with now() at the
# moment of the write, so it can never be older than a read that already
# happened. THEIRS IS NOT: `arrival_time` is stamped by the railways team when
# they write the record and send it. That is the sender's clock, not ours, and
# the row commits here some time later.
#
# So this sequence loses a row permanently:
#     09:00  we read, remember 09:00
#     08:58  they stamp a record
#     09:01  it commits here
#     09:05  we ask for > 09:00 — the row says 08:58, and is never returned
#
# Overlapping the window closes it. Re-reading a row costs nothing: rebuilding
# an unchanged journey produces the identical journey, verified byte-for-byte.
# Measured at 15 min: 110 trains instead of 69, 0.41 s instead of 0.23 s —
# irrelevant against a 5-minute budget, and it buys tolerance for clock skew,
# transmission delay and out-of-order batch loads.
FEED_LOOKBACK_MIN = int(os.getenv("IRPILOT_FEED_LOOKBACK_MIN", "15"))

# HOW TO ASK "WHICH TRAINS CHANGED SINCE I LAST LOOKED?"
#
# There are two clocks in play and they must never be compared to each other:
#
#   write  the REAL time a row landed in the database. Ours is `changed_at`,
#          stamped now() by the trigger, so it is always today and always moves
#          forward. Compared against our own last-read time — both real.
#
#   event  the time the train actually moved: actual_arrival_time /
#          actual_departure_time. On a replayed September day these are 2025
#          timestamps, and the real clock is 2026. Comparing one against the
#          other is never true, which returns nothing, forever, in silence.
#
# `write` is right for a genuinely live feed. `event` is right whenever a past
# day is being fed through in real time — the demo — and is also the only
# option when their table has no reliable write-time column at all.
#
# auto: event while the data is more than a day behind the clock, write
# otherwise. Measured at boot from the newest event in the table.
CHANGED_BY = os.getenv("IRPILOT_CHANGED_BY", "auto").strip().lower()
if CHANGED_BY not in ("auto", "write", "event"):
    raise SystemExit(f"IRPILOT_CHANGED_BY must be auto, write or event, "
                     f"got {CHANGED_BY!r}")
