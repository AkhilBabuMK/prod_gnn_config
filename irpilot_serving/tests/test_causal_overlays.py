# -*- coding: utf-8 -*-
"""
Prove the rollout overlays are causal at T0 — and still carry what a controller
legitimately knows.

Two failure modes, opposite directions, and both matter:

  LEAK      the forecaster sees something that has not happened yet
  BLINDNESS the forecaster ignores something a controller genuinely holds

A fix that only avoids leaks by knowing nothing is not a fix.

    python irpilot_serving/tests/test_causal_overlays.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "irpilot_serving"))
sys.path.insert(0, str(ROOT / "jabalpur_scenario_gnn"))

import pandas as pd                                              # noqa: E402

import config                                                    # noqa: E402
config.bootstrap_gnn_path()

import cris_pipeline.build_dataset as BD                          # noqa: E402
from cris_pipeline.overlays import _sec_key, CODE_TO_ID           # noqa: E402
from live.overlays_causal import (                                # noqa: E402
    CausalFreightOverlay, CausalDisruptionOverlay,
)

BD._init_topology(json.loads(
    (ROOT / "jabalpur_scenario_gnn" / "data_cris" /
     "topology_jbp_sta_cris.json").read_text(encoding="utf-8")))
PAIRS = {(int(s["from_id"]), int(s["to_id"])) for s in BD._SECTIONS.values()}

DAY = "2025-09-27"
T0 = pd.Timestamp(f"{DAY} 09:00")
M = lambda h, m: h * 60 + m

_ok = _fail = 0


def check(name, cond, detail=""):
    global _ok, _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    _ok += bool(cond)
    _fail += (not cond)


def freight(rows):
    return CausalFreightOverlay.at_t0(pd.DataFrame(rows), T0)


def main() -> int:
    print("=" * 78)
    print(f"CAUSAL OVERLAYS AT T0 = {T0}")
    print("=" * 78)

    jke, pkrd = CODE_TO_ID["JKE"], CODE_TO_ID["PKRD"]
    key = _sec_key(jke, pkrd)

    # ── LEAK: an observation after T0 must be invisible ─────────────────────
    print("\n1. NO LEAK — observations after T0 are not visible\n")
    ov = freight([
        {"coatrainid": "G1", "sqncnumb": 1, "station": "JKE",
         "arvltime": pd.Timestamp(f"{DAY} 11:00"),      # AFTER T0
         "dprttime": pd.Timestamp(f"{DAY} 11:30"),
         "schdarvl": pd.NaT, "schddprt": pd.NaT},
    ])
    check("a freight that arrives at 11:00 is invisible at 09:00",
          ov.station_count(DAY, jke, M(11, 10)) == 0,
          f"count at 11:10 = {ov.station_count(DAY, jke, M(11, 10))}")

    # ── KNOWN: freight standing at T0 is carried forward ────────────────────
    print("\n2. NOT BLIND — freight ON the corridor at T0 is carried forward\n")
    ov = freight([
        {"coatrainid": "G2", "sqncnumb": 1, "station": "JKE",
         "arvltime": pd.Timestamp(f"{DAY} 08:50"),      # standing at T0
         "dprttime": pd.NaT, "schdarvl": pd.NaT, "schddprt": pd.NaT},
    ])
    check("standing freight occupies the station at T0",
          ov.station_count(DAY, jke, M(9, 0)) == 1)
    check("...and is still there a few minutes into the horizon",
          ov.station_count(DAY, jke, M(9, 20)) == 1,
          "carried on a typical 37 min dwell")

    ov = freight([
        {"coatrainid": "G3", "sqncnumb": 1, "station": "JKE",
         "arvltime": pd.Timestamp(f"{DAY} 08:30"),
         "dprttime": pd.Timestamp(f"{DAY} 08:55"),      # in transit at T0
         "schdarvl": pd.NaT, "schddprt": pd.NaT},
        {"coatrainid": "G3", "sqncnumb": 2, "station": "PKRD",
         "arvltime": pd.NaT, "dprttime": pd.NaT,
         "schdarvl": pd.NaT, "schddprt": pd.NaT},
    ])
    check("in-transit freight occupies its section at T0",
          ov.section_count(DAY, key, M(9, 0)) == 1)

    # ── KNOWN: goods BOOKED to run later — the point you raised ─────────────
    print("\n3. SCHEDULED GOODS — booked to arrive after T0, still counted\n")
    ov = freight([
        {"coatrainid": "G4", "sqncnumb": 1, "station": "JKE",
         "arvltime": pd.NaT, "dprttime": pd.NaT,        # not seen yet
         "schdarvl": pd.Timestamp(f"{DAY} 10:00"),      # BOOKED inside horizon
         "schddprt": pd.Timestamp(f"{DAY} 10:20")},
        {"coatrainid": "G4", "sqncnumb": 2, "station": "PKRD",
         "arvltime": pd.NaT, "dprttime": pd.NaT,
         "schdarvl": pd.Timestamp(f"{DAY} 10:40"), "schddprt": pd.NaT},
    ])
    check("a goods train booked for 10:00 is counted at 10:10",
          ov.station_count(DAY, jke, M(10, 10)) == 1,
          "ignoring it would understate congestion")
    check("...and occupies the section on its booked leg",
          ov.section_count(DAY, key, M(10, 30)) == 1)
    check("but it is NOT counted before it is due",
          ov.station_count(DAY, jke, M(9, 30)) == 0)

    # ── BLOCKS: planned window known; real clearance is not ─────────────────
    print("\n4. BLOCKS — the PERMITTED window is known, the real clearance is not\n")
    sec = "PKRD-JKE"
    d = CausalDisruptionOverlay.at_t0(
        blocks=[(sec, pd.Timestamp(f"{DAY} 08:40"), pd.Timestamp(f"{DAY} 09:30"),
                 pd.Timestamp(f"{DAY} 10:15"))],       # really cleared 10:15
        failures=[], tsrs=[], t0=T0, section_pairs=PAIRS)
    check("a block running at T0 is in force", d.state(DAY, key, M(9, 0))[0] == 1.0)
    check("it is in force to its PERMITTED end", d.state(DAY, key, M(9, 25))[0] == 1.0)
    check("it does NOT extend to the real clearance we cannot know",
          d.state(DAY, key, M(10, 0))[0] == 0.0,
          "10:15 clearance is invisible at 09:00")

    d = CausalDisruptionOverlay.at_t0(
        blocks=[(sec, pd.Timestamp(f"{DAY} 11:00"), pd.Timestamp(f"{DAY} 12:00"),
                 None)],
        failures=[], tsrs=[], t0=T0, section_pairs=PAIRS)
    check("a block PLANNED for 11:00 IS known at 09:00",
          d.state(DAY, key, M(11, 30))[0] == 1.0,
          "planned work is on the controller's desk")

    # ── FAILURES: unplanned, so a future one cannot be known ────────────────
    print("\n5. FAILURES — unplanned, so a future one must be invisible\n")
    d = CausalDisruptionOverlay.at_t0(
        blocks=[], failures=[(sec, pd.Timestamp(f"{DAY} 11:00"),
                              pd.Timestamp(f"{DAY} 11:40"), 4)],
        tsrs=[], t0=T0, section_pairs=PAIRS)
    check("a failure that starts at 11:00 is invisible at 09:00",
          d.state(DAY, key, M(11, 20))[0] == 0.0,
          "nobody knows a signal will fail in two hours")

    d = CausalDisruptionOverlay.at_t0(
        blocks=[], failures=[(sec, pd.Timestamp(f"{DAY} 08:45"), None, 4)],
        tsrs=[], t0=T0, section_pairs=PAIRS)
    check("a failure already open at T0 IS known",
          d.state(DAY, key, M(8, 50))[0] > 0.0)

    print("\n" + "=" * 78)
    print(f"  {_ok} passed, {_fail} failed")
    print("=" * 78)
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
