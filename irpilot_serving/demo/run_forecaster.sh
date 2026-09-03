#!/usr/bin/env bash
# Point the production loop at the demo feed table, using the PROD mapping.
# Only the main table is overridden; everything else keeps the local names,
# since the demo database has our goods/maintenance/tsr tables, not theirs.
set -e
cd "$(dirname "$0")/.."

export PGPASSWORD="${PGPASSWORD:-root}"
export IRPILOT_PROFILE=prod
export IRPILOT_TBL_MAIN=demo_gnn_input_table
export IRPILOT_TBL_GOODS_RUNNING=goods_running
export IRPILOT_TBL_GOODS_SCHEDULE=goods_schedule
export IRPILOT_TBL_MAINTENANCE=maintenance
export IRPILOT_TBL_ASSET_FAILURE=asset_failure
export IRPILOT_TBL_TSR=tsr
export IRPILOT_TBL_TSR_FROM=from_datetime
export IRPILOT_TBL_TSR_TO=to_datetime

exec python -u run_prod.py "$@"
