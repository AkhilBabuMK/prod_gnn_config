#!/usr/bin/env bash
# Fill in the database details, then:  source env.sh
#
# Nothing else needs editing. Every environment-specific value the service
# reads is set here.

# ── which set of table names to use ──────────────────────────────────────────
# prod = their tables. sim = the development database. There is no other value.
export IRPILOT_SCHEMA=prod

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

# The column on the main table that says when a row was written. We compare it
# against wall time to ask "what is new since I last looked?".
export IRPILOT_TBL_CHANGED_AT=arrival_time

# If the TSR guard reports windows that end before they start, the date text is
# not ISO and needs an explicit parse:
# export IRPILOT_TBL_TSR_TO="to_timestamp(todatetime,'DD/MM/YYYY HH24:MI')"
