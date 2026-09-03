#!/usr/bin/env bash
# Fill in the database details, then:  source env.sh
#
# Nothing else needs editing. Every environment-specific value the service
# reads is set here.

# ── WHICH SET OF TABLE NAMES TO USE ──────────────────────────────────────────
# NOT a Postgres schema. This picks a naming profile:
#   prod  your tables, each qualified with the schema it lives in (below)
#   sim   the development database
# Where we CREATE things is the separate IRPILOT_WRITE_SCHEMA setting.
export IRPILOT_PROFILE=prod

# ── their database ───────────────────────────────────────────────────────────
export IRPILOT_DB_HOST=localhost
export IRPILOT_DB_PORT=5432
export IRPILOT_DB_NAME=irpilotdb
export IRPILOT_DB_USER=irpilot
export IRPILOT_DB_PASSWORD=

# ── where this package was unpacked ──────────────────────────────────────────
export IRPILOT_GNN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/jabalpur_scenario_gnn"
export IRPILOT_DATASET_DIR="$IRPILOT_GNN_ROOT/data_cris/dataset_v13"

# ── WHERE WE CREATE OUR TABLES ───────────────────────────────────────────────
# Everything this service creates goes here and nowhere else:
#   forecast, model_state, infra_topology, station_ref,
#   forecast_read, forecast_latest
# Leave blank to use the default search_path.
export IRPILOT_WRITE_SCHEMA=data

# Rename any of ours the same way. The forecast table is called forecast_output:
export IRPILOT_OUT_FORECAST=forecast_output
# export IRPILOT_OUT_MODEL_STATE=...
# export IRPILOT_OUT_STATION_REF=...

# ── WHERE WE READ FROM ───────────────────────────────────────────────────────
# One line per table, so each can sit in whichever schema it actually lives in.
# Qualify every one — an unqualified name resolves through search_path, which
# is exactly how a table ends up being read from the wrong schema.
export IRPILOT_TBL_MAIN=data.gnn_input_table
export IRPILOT_TBL_GOODS_RUNNING=test.goods_train_running
export IRPILOT_TBL_GOODS_SCHEDULE=test.goods_train_schedule
export IRPILOT_TBL_MAINTENANCE=data.maintenance_block
export IRPILOT_TBL_ASSET_FAILURE=test.asset_failure
export IRPILOT_TBL_TSR=data.temporary_speed_restriction
export IRPILOT_TBL_COACHES=test.coaches
export IRPILOT_TBL_PF_INFO=test.pf_info

# The column on the main table that says when a row was written. Used to ask
# "what is new since I last looked?" — but ONLY when comparing against real
# time (see IRPILOT_CHANGED_BY below). Leave as-is; it is ignored entirely in
# event mode.
export IRPILOT_TBL_CHANGED_AT=arrival_time

# ── HOW TO ASK "WHAT CHANGED?" ───────────────────────────────────────────────
# Two clocks exist and must never be compared to each other:
#   write  when a row LANDED in the database (arrival_time, real time)
#   event  when the train ACTUALLY MOVED (actual_arrival_time etc, whatever
#          day the data itself is from)
#
# A live feed needs `write`. Replaying or streaming a PAST day needs `event` —
# comparing that day's dates against today's clock returns nothing, forever,
# with no error.
#
#   auto   (default) decide automatically: event if the newest data in the
#          table is more than a day behind right now, write otherwise.
#   write  always use IRPILOT_TBL_CHANGED_AT above.
#   event  always use the actual movement times.
#
# Leave on auto unless told otherwise — it is safe either way, since it never
# mixes the two clocks.
export IRPILOT_CHANGED_BY=auto

# If the TSR guard reports windows that end before they start, the date text is
# not ISO and needs an explicit parse:
# export IRPILOT_TBL_TSR_TO="to_timestamp(todatetime,'DD/MM/YYYY HH24:MI')"
