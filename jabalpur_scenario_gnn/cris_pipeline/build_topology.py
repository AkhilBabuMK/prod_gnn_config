# -*- coding: utf-8 -*-
"""
cris_pipeline/build_topology.py
===============================
Builds the JBP-STA corridor topology from the OFFICIAL CRIS infrastructure
tables, replacing the hand-derived topology_jbp_sta.json.

What changes versus the previous topology:
  - junctions: 8 -> 4   (CRIS MACJUNCTION; the other 4 were phantoms created
                         by duplicate aggregate sections)
  - sections:  24 -> 21 (three aggregate duplicates removed)
  - main_lines / max_speed / distance: real MANNUMBLINES, MANMAXSPEED,
                                       MANINTRDIST instead of guesses
  - platform capacity: real MANNUMBTRAINALLWD holding capacity added
  - PSR time-loss per section added

Output schema is deliberately identical to topology_jbp_sta.json so the
existing dataset/model code can consume it unchanged.

Usage:
    python -m cris_pipeline.build_topology
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import pandas as pd

from .config import (
    STATION_CSV, BLOCKSCTN_CSV, STTNLINE_CSV, PLATFORM_CSV, PSR_CSV,
    CORRIDOR, CORRIDOR_ORDER, BLOCK_POSTS, JUNCTIONS, DUPLICATE_SECTIONS,
    SECTION_CODE, TOPOLOGY_PATH, OUT_DIR,
)

# ── Train class taxonomy (unchanged from the previous pipeline) ───────────────

TRAIN_TYPE_MAP = {
    "SUF": "superfast", "MEX": "mail_express", "TOD": "passenger",
    "VNDB": "vande_bharat", "DRNT": "duronto", "AMTB": "amrit_bharat",
    "TRST": "superfast", "PEXP": "passenger", "MEMU": "memu",
    "DEMU": "demu", "RAJ": "rajdhani", "SHTB": "shatabdi",
    "HMSF": "humsafar", "JSHT": "jan_shatabdi", "INTC": "intercity",
    "TRSF": "passenger", "PPSP": "passenger", "INGL": "passenger",
}
CLASS_PROPERTIES = {
    "rajdhani":     {"priority": 1.00, "speed_factor": 1.160},
    "vande_bharat": {"priority": 0.97, "speed_factor": 1.140},
    "shatabdi":     {"priority": 0.94, "speed_factor": 1.100},
    "duronto":      {"priority": 0.91, "speed_factor": 1.060},
    "humsafar":     {"priority": 0.87, "speed_factor": 1.015},
    "amrit_bharat": {"priority": 0.83, "speed_factor": 0.995},
    "superfast":    {"priority": 0.78, "speed_factor": 0.975},
    "jan_shatabdi": {"priority": 0.72, "speed_factor": 0.940},
    "mail_express": {"priority": 0.65, "speed_factor": 0.890},
    "intercity":    {"priority": 0.60, "speed_factor": 0.870},
    "passenger":    {"priority": 0.45, "speed_factor": 0.690},
    "memu":         {"priority": 0.42, "speed_factor": 0.710},
    "demu":         {"priority": 0.40, "speed_factor": 0.665},
    "freight":      {"priority": 0.20, "speed_factor": 0.550},
    "coal_freight": {"priority": 0.18, "speed_factor": 0.520},
}
CLASS_TO_GROUP = {
    "rajdhani": "premium", "vande_bharat": "premium", "shatabdi": "premium",
    "duronto": "premium", "humsafar": "premium", "amrit_bharat": "premium",
    "superfast": "passenger", "jan_shatabdi": "passenger",
    "mail_express": "passenger", "intercity": "passenger",
    "passenger": "passenger", "memu": "commuter", "demu": "commuter",
    "freight": "freight", "coal_freight": "freight",
}
TRAIN_CLASS_ORDER = list(CLASS_PROPERTIES)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section_key(a: int, b: int) -> str:
    return f"{min(a, b)},{max(a, b)}"


def _assert_connected(stations: dict, sections: dict) -> None:
    """The corridor must be one connected component covering every station.

    Guards a real failure mode: scoping edges by MAVSCTNCODage drops the
    KTE-KTES link, which silently splits the corridor in two and would let a
    dataset build and train while message passing could never cross Katni.
    """
    adj: dict[int, set[int]] = {int(s["station_id"]): set()
                                for s in stations.values()}
    for sec in sections.values():
        a, b = int(sec["from_id"]), int(sec["to_id"])
        adj[a].add(b)
        adj[b].add(a)

    start = next(iter(adj))
    seen, stack = {start}, [start]
    while stack:
        for nb in adj[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)

    if len(seen) != len(adj):
        by_id = {int(s["station_id"]): s["code"] for s in stations.values()}
        missing = sorted(by_id[i] for i in set(adj) - seen)
        raise RuntimeError(
            f"Corridor graph is disconnected: {len(seen)}/{len(adj)} stations "
            f"reachable. Unreachable: {missing}")

    isolated = sorted(stations[k]["code"] for k in stations
                      if not adj[int(stations[k]["station_id"])])
    if isolated:
        raise RuntimeError(f"Stations with no section edge: {isolated}")


def build_topology() -> dict:
    stn  = pd.read_csv(STATION_CSV, low_memory=False)
    bsec = pd.read_csv(BLOCKSCTN_CSV, low_memory=False)
    line = pd.read_csv(STTNLINE_CSV, low_memory=False)
    plat = pd.read_csv(PLATFORM_CSV, low_memory=False)
    psr  = pd.read_csv(PSR_CSV, low_memory=False)

    code_to_id = {code: i for i, code in enumerate(CORRIDOR_ORDER)}

    # ── Per-station attributes from CRIS ──────────────────────────────────────
    stn = stn[stn.MAVSTTNCODE.isin(CORRIDOR)].set_index("MAVSTTNCODE")

    plat_n = (plat[plat.MAVSTTNCODE.isin(CORRIDOR)]
              .groupby("MAVSTTNCODE").MAVPLATFORMNUMB.nunique())
    plat_len = (plat[plat.MAVSTTNCODE.isin(CORRIDOR)]
                .groupby("MAVSTTNCODE").MANLENGTH.max())

    lines = line[line.MAVSTTNCODE.isin(CORRIDOR)]
    n_lines   = lines.groupby("MAVSTTNCODE").MAVLINENUMB.nunique()
    hold_cap  = lines.groupby("MAVSTTNCODE").MANNUMBTRAINALLWD.sum()
    loop_n    = (lines[lines.MACLINECATEGORY == "L"]
                 .groupby("MAVSTTNCODE").MAVLINENUMB.nunique())

    stations: dict[str, dict] = {}
    for code in CORRIDOR_ORDER:
        sid       = code_to_id[code]
        is_block  = code in BLOCK_POSTS
        row       = stn.loc[code] if code in stn.index else None
        platforms = int(plat_n.get(code, 0))
        stations[str(sid)] = {
            "station_id":     sid,
            "code":           code,
            "name":           (str(row.MAVSTTNNAME).strip() if row is not None
                               else code),
            "is_junction":    code in JUNCTIONS,
            "is_division":    not is_block,      # block posts carry no targets
            "is_block_post":  is_block,
            "platforms":      platforms,
            "platform_len_m": float(plat_len.get(code, 0.0) or 0.0),
            "main_lines":     2,                 # whole corridor is double track
            "loop_lines":     int(loop_n.get(code, 0)),
            "total_lines":    int(n_lines.get(code, 0)),
            "holding_capacity": int(hold_cap.get(code, 0)),
            "category":       ("block_post" if is_block
                               else "junction" if code in JUNCTIONS
                               else "station"),
            "lat":            float(row.MANLATITUDE)  if row is not None else 0.0,
            "lon":            float(row.MANLONGITUDE) if row is not None else 0.0,
            "crew_change":    (str(getattr(row, "MACCREWCHNG", "N")) == "Y")
                              if row is not None else False,
            "traction_change": (str(getattr(row, "MACTRTNCHNG", "N")) == "Y")
                              if row is not None else False,
        }

    # ── PSR time-loss per block section ───────────────────────────────────────
    psr = psr.copy()
    psr["A"] = psr.MAVBLCKSCTN.astype(str).str.split("-").str[0]
    psr["B"] = psr.MAVBLCKSCTN.astype(str).str.split("-").str[-1]
    psr_loss: dict[frozenset, float] = defaultdict(float)
    for r in psr[psr.A.isin(CORRIDOR) & psr.B.isin(CORRIDOR)].itertuples():
        psr_loss[frozenset((r.A, r.B))] += float(r.MANTIMELOSSPSGR or 0.0)

    # ── Sections ──────────────────────────────────────────────────────────────
    # MAVSCTNCODE=='JBP-STA' is the authority on which STATIONS form the
    # corridor (it resolves to exactly the 22 in config.CORRIDOR_ORDER), but it
    # is too strict for EDGES: KTE-KTES (1.48 km) is administratively filed
    # under a neighbouring section code, and filtering on it alone splits the
    # corridor into two disconnected components. Edges are therefore every
    # block section with both endpoints on the corridor.
    bsec = bsec[bsec.MAVFROMSTTNCODE.isin(CORRIDOR)
                & bsec.MAVTOSTTNCODE.isin(CORRIDOR)]

    sections: dict[str, dict] = {}
    dropped_duplicates: list[str] = []
    for r in bsec.itertuples():
        a_code, b_code = r.MAVFROMSTTNCODE, r.MAVTOSTTNCODE
        pair = frozenset((a_code, b_code))
        if pair in DUPLICATE_SECTIONS:
            dropped_duplicates.append(f"{a_code}-{b_code}")
            continue
        a, b = code_to_id[a_code], code_to_id[b_code]
        key  = _section_key(a, b)
        if key in sections:
            # CRIS lists each section once per direction; merge by taking the
            # more restrictive speed so the section can't look faster than its
            # slowest direction.
            sections[key]["max_speed"] = min(sections[key]["max_speed"],
                                             int(r.MANMAXSPEED))
            continue

        dist_km   = float(r.MANINTRDIST)
        main_ln   = int(r.MANNUMBLINES)
        max_speed = int(r.MANMAXSPEED)
        # Base run time from distance and line speed, with the standard 15%
        # allowance for acceleration/braking at each end.
        base_run  = (dist_km / max(max_speed, 1)) * 60.0 * 1.15

        sections[key] = {
            "from_id":       min(a, b),
            "to_id":         max(a, b),
            "from_code":     CORRIDOR_ORDER[min(a, b)],
            "to_code":       CORRIDOR_ORDER[max(a, b)],
            "distance_km":   round(dist_km, 2),
            "base_run_min":  round(base_run, 2),
            "track_type":    "double_elec" if main_ln >= 2 else "single_elec",
            "max_speed":     max_speed,
            "section_class": "mainline",
            "main_lines":    main_ln,
            "signal_type":   str(r.MAVSGNLTYPE),
            "psr_time_loss_min": round(psr_loss.get(pair, 0.0), 1),
        }

    junction_ids = sorted(code_to_id[c] for c in JUNCTIONS)

    _assert_connected(stations, sections)

    return {
        "meta": {
            "corridor":              "JBP-STA",
            "source":                "CRIS track_infra_* (official)",
            "n_stations":            len(stations),
            "n_halt_stations":       len(stations) - len(BLOCK_POSTS),
            "n_block_posts":         len(BLOCK_POSTS),
            "n_sections":            len(sections),
            "n_junctions":           len(junction_ids),
            "dropped_duplicate_sections": sorted(set(dropped_duplicates)),
            "all_double_track":      all(s["main_lines"] >= 2
                                         for s in sections.values()),
        },
        "stations":             stations,
        "sections":             sections,
        "code_to_id":           code_to_id,
        "junction_ids":         junction_ids,
        "outside_junction_ids": [],
        "trains":               [],          # filled by build_journeys.py
        "train_classes":        CLASS_PROPERTIES,
        "train_class_order":    TRAIN_CLASS_ORDER,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TOPOLOGY_PATH))
    args = ap.parse_args()

    topo = build_topology()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(topo, fh, indent=1)

    m = topo["meta"]
    print(f"Wrote {args.out}")
    print(f"  stations      : {m['n_stations']} "
          f"({m['n_halt_stations']} halts + {m['n_block_posts']} block posts)")
    print(f"  sections      : {m['n_sections']}")
    print(f"  junctions     : {m['n_junctions']} "
          f"({', '.join(sorted(JUNCTIONS))})")
    print(f"  dropped dups  : {m['dropped_duplicate_sections']}")
    print(f"  double track  : {m['all_double_track']}")


if __name__ == "__main__":
    main()
