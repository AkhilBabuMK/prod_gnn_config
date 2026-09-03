# -*- coding: utf-8 -*-
"""
cris_pipeline/config.py
=======================
Single source of truth for the CRIS (official) JBP-STA corridor pipeline.

Replaces the self-scraped railradar data that every model up to
`nextevent_v2.pt` was trained on. See cris_pipeline/README.md for the
measured justification behind each decision here.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

PKG_DIR     = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent                       # jabalpur_scenario_gnn/
REPO_ROOT   = PROJECT_DIR.parent

DATA_ROOT   = PROJECT_DIR / "data_standardized_timestamp_all"
CRIS_DIR    = (DATA_ROOT / "cris_data_standardized_timestamp"
                         / "3_cris_data_all_cleaned_files_standardized")

RUNNING_CSV = DATA_ROOT / "train_running_clean_clean.csv"
GOODS_CSV   = DATA_ROOT / "goodsrunning_data_clean.csv"

# Infrastructure / auxiliary tables
STATION_CSV   = CRIS_DIR / "track_infra_station.csv"
BLOCKSCTN_CSV = CRIS_DIR / "track_infra_blocksctn.csv"
STTNLINE_CSV  = CRIS_DIR / "track_infra_sttnline.csv"
PLATFORM_CSV  = CRIS_DIR / "track_infra_platform.csv"
PSR_CSV       = CRIS_DIR / "track_infra_block_psr.csv"
MAINT_CSV     = CRIS_DIR / "maintenance_block.csv"
TSR_CSV       = CRIS_DIR / "temporary_speed_restriction.csv"
ASSET_CSV     = CRIS_DIR / "asset_failure_details.csv"
DETENTION_CSV = CRIS_DIR / "df_cleaned (2).csv"
COACHES_CSV   = CRIS_DIR / "no_of_coaches_data.csv"
PF_INFO_CSV   = CRIS_DIR / "PF_INFO.csv"

OUT_DIR       = PROJECT_DIR / "data_cris"
TOPOLOGY_PATH = OUT_DIR / "topology_jbp_sta_cris.json"
JOURNEYS_DIR  = OUT_DIR / "journeys"
DATASET_DIR   = OUT_DIR / "dataset"


# ── Corridor definition ───────────────────────────────────────────────────────

# Corridor scope is taken from the infrastructure data itself, not inferred:
# track_infra_blocksctn.MAVSCTNCODE names each section, and 'JBP-STA' resolves
# to exactly these 22 stations. Geographic order STA -> JBP (verified against
# distance_from_source_km in the running data and lat/long in
# track_infra_station.csv), so station_id is monotone along the corridor and
# direction can be read straight off the index.
CORRIDOR_ORDER = [
    "STA", "LGCE", "UHR", "MYR", "BUU", "GNWA", "UDR", "PKRD", "JKE", "PTWA",
    "KTE", "KTES", "MDRR", "NWR", "SNRR", "SBD", "DDCE", "SHR", "GSPR", "DOE",
    "ADTL", "JBP",
]
CORRIDOR = set(CORRIDOR_ORDER)

# The authoritative section name in track_infra_blocksctn.MAVSCTNCODE.
# Stations such as NKJ / KMZ sit on adjacent named sections (KTES-NKJ,
# KMZ-KTES, KTE-SDL) and are deliberately NOT part of this corridor.
SECTION_CODE = "JBP-STA"

# Block posts: MACHALTFLAG='Y', station class 'D', zero platform lines.
# Measured: 0.0% of their stop-rows carry an actual arrival/departure time, so
# they can never be supervised. They stay in the graph as pass-through nodes
# (they subdivide sections and carry occupancy) but are excluded from targets.
BLOCK_POSTS = {"GNWA", "MDRR", "SNRR"}

# We forecast at EVERY station the train moves through, not only where it
# halts. Measured: just 23.7% of corridor stop-rows are booked halts, so
# restricting targets to halts would discard ~3/4 of the supervision.
# Only the three block posts are excluded, because they never report an actual.
TARGET_STATIONS = [c for c in CORRIDOR_ORDER if c not in BLOCK_POSTS]   # 19

# Real junctions per CRIS track_infra_station.MACJUNCTION == 'Y'.
# The previous topology listed 8, adding BUU/UDR/NWR/SBD as "degree>=3"
# junctions — but that degree was an artifact of the duplicate aggregate
# sections below. Only these four are junctions.
JUNCTIONS = {"STA", "KTE", "KTES", "JBP"}

# CRIS lists BOTH the subdivided legs through a block post AND an aggregate
# section spanning it. Loading both double-counts track capacity and creates
# phantom junctions. Verified by exact distance arithmetic:
#     BUU-GNWA 6.24 + GNWA-UDR 6.42 = 12.66 == BUU-UDR  12.66
#     KTES-MDRR 3.93 + MDRR-NWR 7.85 = 11.78 == KTES-NWR 11.79
#     NWR-SNRR 7.64 + SNRR-SBD 6.96 = 14.60 == NWR-SBD  14.60
# We keep the finer subdivision (better occupancy resolution) and drop these.
DUPLICATE_SECTIONS = {
    frozenset(("BUU", "UDR")),
    frozenset(("KTES", "NWR")),
    frozenset(("NWR", "SBD")),
}


# ── Service filtering ─────────────────────────────────────────────────────────

# Train types excluded from SUPERVISION (they remain actors in the graph —
# see build_journeys.py, is_target_eligible).
#
# The original rule here excluded anything with a huge median delay, on the
# stated grounds that its "scheduled" time was a nominal path rather than a
# published timetable. THAT REASONING WAS WRONG and the measurement that
# disproves it is worth keeping:
#
#   scheduled leg time between adjacent corridor stations, p50 minutes
#     SUF 8 | MEX 9 | VNDB 8 | DRNT 8 | TOD 8 | TRST 8 | AMTB 7 | PEXP 8
#
# Every type has a normal, real timetable inside the corridor. The huge medians
# are INHERITED, not fabricated — median delay at the FIRST corridor stop is
# 209 min for TOD and 318 for PEXP, against 5-6 min for SUF/MEX. These trains
# arrive already hours late and then run normally (actual/scheduled leg ratio
# 1.38x for TOD vs 1.12x for SUF — consistent with low priority, not with bad
# data). Not a date bug either: only 11.7% of TOD delays sit near a 24 h
# multiple.
#
# So TOD/TRST/AMTB are now supervised. They add ~7,440 stops (+16%) and are
# exactly the low-priority services that get held, which is the model's
# weakest case. Their large absolute delay is not a problem because the target
# is a DELTA against current delay, not a level.
#
# Still excluded, and why:
#   PEXP  parcel express — real leg structure, but median 530 min and 90.6% of
#         stops beyond 4 h late; n=127. Too extreme and too few to be worth it.
#   TRSF  railway-staff special — 0% of stops ever early (SUF 10.2%, MEX 14.6%),
#         so its schedule shows no slack at all; n=38.
#   PPSP  never observed on this corridor; kept for safety.
#   INGL  inaugural one-off — 20.6% early but 35.3% beyond 4 h, i.e. bimodal
#         and unmodellable; n=34.
NOMINAL_SCHEDULE_TYPES = {"PEXP", "PPSP", "TRSF", "INGL"}

# Freight has no published timetable either (median "delay" 324 min). Freight
# is used ONLY for occupancy / conflict features and is never a target.
FREIGHT_IS_TARGET = False


# ── Split ─────────────────────────────────────────────────────────────────────

# Chronological 20 / 5 / 5 over the 30 available days (2025-09-01 .. 2025-09-30).
# Chronological (not interleaved) because a forecasting system must be tested
# forward in time.
N_TRAIN_DAYS = 20
N_VAL_DAYS   = 5
N_TEST_DAYS  = 5


# ── Feature scaling ───────────────────────────────────────────────────────────

# The old feature set clamped junction ETAs at 180 min. On the 117-station
# division graph the nearest junction was usually further away than that, so
# the dim saturated: measured std 0.048 (eta_nearest) and 0.0064 (eta_second).
# The corridor is ~238 km end to end with 4 junctions; 90 min covers the
# realistic approach window without saturating.
JCT_ETA_CLAMP_MIN = 90.0

# Measured end to end from the 21 CRIS block sections (STA -> JBP).
CORRIDOR_LENGTH_KM = 188.87
