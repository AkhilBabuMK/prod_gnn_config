# -*- coding: utf-8 -*-
"""
cris_pipeline/dataset_cris.py
=============================
Feature spec for the official CRIS corridor data.

This is a re-derivation, not a port. Every dim from the old spec was checked
against measured statistics on `dataset_v12` and kept, rescaled, or dropped:

  DROPPED (provably constant on this corridor / in this data)
    is_single_track     all 21 corridor sections are MANNUMBLINES=2
    eta_third_jct       only 4 junctions exist; the third never binds
    route_len           corridor journeys are a fixed 19-stop chain
    journey_stage       duplicate of route_progress (identical values)

  RESCALED (previously saturated)
    eta_nearest/second_jct   clamp 180 -> 90 min (measured std 0.048 / 0.0064)
    base_run_min_norm        /180 -> /20 (corridor legs are 2.5-9.4 min)
    dwell / time_at_stop     /90 -> /30

  FIXED SEMANTICS
    platform occupancy   now counts only BOOKED HALTS. A passing train
                         occupies the running line, not a platform, and 76% of
                         corridor stop-rows are pass-throughs.
    capacity occupancy   passengers + freight against real MANNUMBTRAINALLWD.
    from_is_jct/to_is_jct  derived from station identity (the old lookup read a
                         key that never existed and returned False always).

  NEW CHANNELS
    freight_in_section / at_station / approaching_next
    section_disrupted / section_run_multiplier / psr_time_loss
    entry_delay / entry_trend / has_upstream  (inherited from outside corridor)
    at_booked_halt                            (halt vs pass-through)

  JOURNEY HISTORY
    Per train, the sequence of stops already completed on this run, with the
    per-leg delay delta. The old model only ever saw `delay_min` and a
    one-tick `delay_velocity`, and its GRU memory was keyed by SERVICE and
    truncated at 12 ticks — so it could not represent how delay had propagated
    along the run. `history` here is [T, MAX_HISTORY, HIST_FEAT_DIM] and is
    consumed by model.JourneyHistoryEncoder before message passing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from scenario_events import (
    EVENT_FEAT_DIM, CONFLICT_EDGE_DIM,
    compute_events_and_conflict_edges, event_features, scenario_labels,
)

# ── Dims ──────────────────────────────────────────────────────────────────────

BASE_TRAIN_FEAT_DIM = 38
TRAIN_FEAT_DIM      = BASE_TRAIN_FEAT_DIM + EVENT_FEAT_DIM     # 38 + 8 = 46

# LEGACY FEATURE WIDTH — v3 and everything before it.
#
# The two timetable-slack dims (36 next_allowance_min, 37 slack_ahead_min)
# arrived with v6 (commit 228e567). Checkpoints older than that expect 44 =
# 36 + 8 and will not load against the current 46.
#
# Dropping them is a clean slice rather than a re-layout because they were
# APPENDED to the end of the base block. Verified by diffing
# releases/cris-v3/code against releases/cris-v6/code_as_trained: those two
# fields, plus the `sched_allowance_min` that feeds them, were the ONLY
# functional change to build_dataset.py and _train_features between the two
# releases. Dims 0..35 compute identically, so a v3 checkpoint scored with
# INCLUDE_SLACK off sees exactly the features it was trained on.
#
# Set it from the checkpoint with match_checkpoint_features() BEFORE building
# the dataset -- CrisDataset snapshots the width at construction.
INCLUDE_SLACK = True


def set_slack_features(on: bool) -> None:
    """Include or drop the two timetable-slack dims, and resize accordingly."""
    global INCLUDE_SLACK, BASE_TRAIN_FEAT_DIM, TRAIN_FEAT_DIM
    INCLUDE_SLACK = bool(on)
    BASE_TRAIN_FEAT_DIM = 38 if INCLUDE_SLACK else 36
    TRAIN_FEAT_DIM = BASE_TRAIN_FEAT_DIM + EVENT_FEAT_DIM


def match_checkpoint_features(ck_config: dict) -> None:
    """Match the feature width to the checkpoint about to be scored.

    A no-op for anything trained at v6 or later. check_feature_dims() still
    runs afterwards and still fails loudly on any width this cannot explain.
    """
    want = int(ck_config.get("train_in_dim", TRAIN_FEAT_DIM))
    set_slack_features(want - EVENT_FEAT_DIM >= 38)
STATION_FEAT_DIM    = 20     # 17 + loop_lines / crew_change / traction_change
SECTION_FEAT_DIM    = 10
MAX_HORIZON         = 8

MAX_HISTORY   = 20      # a full corridor run is 19 stops
HIST_FEAT_DIM = 8

TRAIN_CLASS_ORDER = [
    "rajdhani", "vande_bharat", "shatabdi", "duronto", "humsafar",
    "amrit_bharat", "superfast", "jan_shatabdi", "mail_express", "intercity",
    "passenger", "memu", "demu", "freight", "coal_freight",
]
TRAIN_GROUP_ORDER = ["premium", "passenger", "commuter", "freight"]

_L90   = math.log1p(90.0)
_L240  = math.log1p(240.0)
_L1440 = math.log1p(1440.0)


def _f(v, d: float = 0.0) -> float:
    try:
        if v is None:
            return d
        v = float(v)
        return d if math.isnan(v) else v
    except (TypeError, ValueError):
        return d


def _slog(v: float, scale: float = _L240) -> float:
    """Signed log — preserves early-running (negative) delays."""
    return math.copysign(math.log1p(abs(v)), v) / scale


def _eta_norm(v, cap: float = 90.0) -> float:
    """Log-scaled ETA in [0, 1].

    ETAs are computed as (departure delay + scheduled leg), so an early-running
    train can produce a negative value — clamp at 0 before the log.
    """
    return math.log1p(max(0.0, min(_f(v, cap), cap))) / math.log1p(cap)


# ── Station / section ─────────────────────────────────────────────────────────

def _station_features(snap: dict, ids: List[int],
                      topo: Dict[int, dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    m = {n["station_id"]: n for n in snap["station_nodes"]}
    feats, crit = [], []
    for sid in ids:
        n = m.get(sid, {})
        st = topo.get(sid, {})
        cap = max(int(st.get("holding_capacity", 2)), 1)
        feats.append([
            _f(n.get("platform_occupancy_ratio")),
            _f(n.get("capacity_occupancy_ratio")),
            _f(n.get("available_capacity_ratio"), 1.0),
            min(_f(n.get("freight_occupancy")) / cap, 1.5),
            # `passing_train_count` was dropped: a pass-through has
            # arrival == departure, so it is never positioned "at" a station —
            # measured 98.4% zero. That traffic is already carried by
            # trains_in_section on both adjacent sections.
            _f(n.get("incoming_pressure")),
            min(_f(n.get("max_train_delay_min")) / 120.0, 2.0),
            _f(n.get("delayed_train_density")),
            _f(n.get("approach_density")),
            _f(n.get("congestion_score")),
            _f(n.get("temporal_congestion_level")),
            _f(n.get("steps_since_critical_norm")),
            _f(n.get("critical_flag")),
            _f(n.get("tod_sin")), _f(n.get("tod_cos"), 1.0),
            float(bool(st.get("is_junction"))),
            min(int(st.get("platforms", 0)) / 7.0, 1.0),
            min(cap / 11.0, 1.0),
            # WHERE TRAINS GET HELD. Delay on this corridor is created by
            # holds, and hold rate per station is almost entirely explained by
            # fixed infrastructure we already hold in the topology and were
            # not passing to the model:
            #   corr(crew_change,   hold rate) = +0.944
            #   corr(traction_chg,  hold rate) = +0.906
            #   corr(loop_lines,    hold rate) = +0.877
            # Only JBP/KTE/STA change crew, and they hold 25.1/19.2/13.6% of
            # trains against 2-5% elsewhere. Those three stations are also the
            # top of the "held with no other train present" category (KTE 60,
            # STA 51, JBP 40 events) — i.e. this is the missing explanation for
            # it: crew and traction changes, not a mystery.
            min(int(st.get("loop_lines", 0)) / 9.0, 1.0),
            float(bool(st.get("crew_change"))),
            float(bool(st.get("traction_change"))),
        ])
        crit.append(_f(n.get("critical_flag")))
    return (torch.tensor(feats, dtype=torch.float),
            torch.tensor(crit, dtype=torch.float))


def _section_features(snap: dict, keys: List[Tuple[int, int]],
                      junction_ids: set) -> torch.Tensor:
    em = {}
    for e in snap.get("section_edges", []):
        a, b = e["from_station"], e["to_station"]
        em[(min(a, b), max(a, b))] = e

    feats = []
    for (a, b) in keys:
        e = em.get((a, b), {})
        feats.append([
            _f(e.get("base_run_min_norm"), 0.3),
            _f(e.get("trains_in_section_norm")),
            _f(e.get("freight_in_section_norm")),
            _f(e.get("current_multiplier"), 1.0) - 1.0,   # centred on 0
            _f(e.get("is_disrupted")),
            _f(e.get("psr_time_loss_norm")),
            float(a in junction_ids),
            float(b in junction_ids),
            _f(e.get("tod_sin")), _f(e.get("tod_cos"), 1.0),
        ])
    return torch.tensor(feats, dtype=torch.float)


# ── Journey history ───────────────────────────────────────────────────────────

def _journey_history(inst: dict, stop_idx: int, t: Optional[float] = None,
                     at_station: bool = False) -> Tuple[List[List[float]], int]:
    """Per-leg propagation history for the run so far.

    Returns (rows, length) where `rows` is MAX_HISTORY x HIST_FEAT_DIM, oldest
    first, zero-padded at the END so index 0 is always the journey start.
    """
    stops = inst["stops"]
    hist: List[List[float]] = []
    # Seed from the builder's carried estimate, not 0.0 — a train whose first
    # corridor stops went unreported was not on time there. Use the
    # arrival-safe variant: `delay_est` has absorbed that stop's own departure,
    # which for a train still standing there has not happened yet.
    prev_delay = (_f(stops[0].get("delay_est_arr", stops[0].get("delay_est")))
                  if stops else 0.0)

    for k in range(min(stop_idx + 1, len(stops))):
        s = stops[k]
        # Is this the stop the train is standing at RIGHT NOW? If so its
        # departure is the future and nothing derived from it may be read —
        # including the carried estimate, which folds it in.
        pending = at_station and t is not None and k == stop_idx
        d = s.get("arr_delay")
        if d is None:
            d = (s.get("delay_est_arr", s.get("delay_est")) if pending
                 else s.get("delay_est"))
        d = _f(d, prev_delay)
        dd = d - prev_delay                       # the propagation signal
        # The CURRENT stop of a train that is still standing there has NOT
        # finished: its `dep_delay` and `actual_dep_abs` are the future, and
        # they encode exactly how long the hold will last. Using them told the
        # model "you will stand here 14 minutes" three minutes into the dwell.
        # Same bug as `_path_etas`/`_junction_etas`; caught by the
        # shift_pending_departure perturbation in audit_causality.
        # Causal substitutes: dwell elapsed SO FAR, and a departure delay of at
        # least (now - booked departure).
        if pending:
            dwell = max(0.0, float(t) - _f(s.get("actual_arr_abs")))
            dep = max(d, float(t) - _f(s.get("sched_dep_abs")))
        else:
            dep = _f(s.get("dep_delay"), d)
            dwell = max(0.0, _f(s.get("actual_dep_abs")) - _f(s.get("actual_arr_abs")))
        sched_leg = 0.0
        run_over = 0.0
        if k > 0:
            p = stops[k - 1]
            sched_leg = max(0.0, _f(s.get("sched_arr_abs")) - _f(p.get("sched_dep_abs")))
            actual_leg = max(0.0, _f(s.get("actual_arr_abs")) - _f(p.get("actual_dep_abs")))
            run_over = actual_leg - sched_leg

        hist.append([
            _slog(d),                                   # delay on arrival
            math.tanh(dd / 10.0),                       # per-leg delta
            math.tanh((dep - d) / 10.0),                # dwell-induced change
            min(dwell / 30.0, 1.5),
            math.tanh(run_over / 10.0),                 # leg run over/under
            min(sched_leg / 20.0, 1.5),
            float(bool(s.get("is_booked_halt"))),
            float(bool(s.get("has_actual"))),
        ])
        prev_delay = d

    length = len(hist)
    if length > MAX_HISTORY:                       # keep the most recent window
        hist = hist[-MAX_HISTORY:]
        length = MAX_HISTORY
    while len(hist) < MAX_HISTORY:
        hist.append([0.0] * HIST_FEAT_DIM)
    return hist, length


# ── Train features ────────────────────────────────────────────────────────────

def _train_features(snap: dict, instances: Dict[str, dict],
                    cls_i: dict, grp_i: dict, srv_i: dict,
                    events) -> Tuple:
    nodes = snap.get("train_nodes", [])
    if not nodes:
        z = lambda *s: torch.zeros(*s, dtype=torch.float)
        return (z(0, TRAIN_FEAT_DIM), torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
                z(0, MAX_HORIZON), z(0, MAX_HORIZON), z(0), z(0, 5),
                z(0), z(0), z(0, MAX_HISTORY, HIST_FEAT_DIM),
                torch.zeros(0, dtype=torch.long))

    snap_t = snap.get("timestamp_abs_min")

    ev_x = (event_features(events) if events
            else torch.zeros(len(nodes), EVENT_FEAT_DIM))
    scen_y = (scenario_labels(events) if events
              else torch.zeros(len(nodes), 5))

    feats, srv, cls, grp = [], [], [], []
    tgts, masks, cur_d = [], [], []
    ndep, ndep_m = [], []
    rdw, rdw_m = [], []
    hists, hlens = [], []

    for n in nodes:
        iid = n["instance_id"]
        inst = instances.get(iid, {})
        delay = _f(n.get("delay_min"))
        cur_d.append(delay)

        feats.append([
            _slog(delay),                                             # 0
            math.tanh(_f(n.get("delay_velocity")) / 5.0),             # 1
            _eta_norm(n.get("eta_next_min")),                          # 2
            _eta_norm(n.get("eta_nearest_jct_min")),                   # 3
            _eta_norm(n.get("eta_second_jct_min")),                    # 4
            min(_f(n.get("jcts_remaining")) / 4.0, 1.0),              # 5
            _f(n.get("priority"), 0.5),                               # 6
            max(-1.0, min(1.0, _f(n.get("speed_factor"), 1.0) - 1.0)),# 7
            min(_f(n.get("stops_remaining")) / 19.0, 1.0),            # 8
            _f(n.get("route_progress_norm")),                         # 9
            _f(n.get("mode_in_transit")),                             # 10
            _f(n.get("at_booked_halt")),                              # 11
            min(_f(n.get("time_at_stop_min")) / 30.0, 1.5),           # 12
            min(_f(n.get("minutes_since_departure")) / 60.0, 1.5),    # 13
            min(_f(n.get("minutes_over_scheduled_run")) / 30.0, 1.5), # 14
            math.tanh(_f(n.get("minimum_arrival_delay_next")) / 30.0),# 15
            min(_f(n.get("n_coaches")) / 24.0, 1.5),                  # 16
            min(_f(n.get("junction_overlap_count")) / 6.0, 1.5),      # 17
            min(_f(n.get("same_line_overlap_count")) / 6.0, 1.5),     # 18
            min(_f(n.get("higher_priority_overlap_count")) / 6.0, 1.5),# 19
            min(_f(n.get("queue_ahead_count")) / 6.0, 1.5),           # 20
            1.0 - min(_f(n.get("min_junction_gap_min"), 90.0) / 90.0, 1.0),  # 21
            min(_f(n.get("freight_in_section")) / 2.0, 1.5),          # 22
            min(_f(n.get("freight_at_station")) / 3.0, 1.5),          # 23
            min(_f(n.get("freight_approaching_next")) / 3.0, 1.5),    # 24
            _f(n.get("section_disrupted")),                           # 25
            min(_f(n.get("section_run_multiplier"), 1.0) - 1.0, 2.0), # 26
            _slog(_f(n.get("entry_delay_min"))),                      # 27
            math.tanh(_f(n.get("entry_trend_min")) / 15.0),           # 28
            _f(n.get("has_upstream")),                                # 29
            _f(n.get("tod_sin")),                                     # 30
            _f(n.get("tod_cos"), 1.0),                                # 31
            _f(n.get("delay_observed"), 1.0),                         # 32
            # How soon a HIGHER-PRIORITY train reaches a station on our
            # remaining path. 1.0 = imminent overtake, 0.0 = none in the
            # horizon. Holds are how delay is created here: 84% of increments
            # above 30 min happen while standing, not while running.
            # Precedence is associated with 56.8% of large holds against a
            # 28.8% background rate -- a ~1.9x lift, NOT the 50x an earlier
            # version of this comment claimed (that control was wrong; see
            # analyse_causes.py, which reports the control alongside).
            # So this dim is one contributing signal, not the whole story:
            # 44% of large holds have no higher-priority train nearby at all.
            1.0 - min(_f(n.get("precedence_eta_min"), 75.0) / 75.0, 1.0),  # 33
            # What the timetable says about the stop being predicted into.
            # Without these the next_dep head guesses a dwell with no idea
            # whether one is scheduled: a booked halt overruns 48.4% of the
            # time, a pass-through 4.9%. Both read only the schedule, so they
            # are known in advance and safe under rollout.
            _f(n.get("next_booked_halt")),                             # 34
            min(_f(n.get("next_sched_dwell_min")) / 10.0, 1.5),        # 35
            # Timetable slack — where the train is ALLOWED to make time back.
            # 36 is the padding at the stop being predicted, 37 the total left
            # to the end of the journey. Scaled so the common range lands near
            # [0, 1]: allowance is 5.2 min on average where present, and the
            # journey totals run to a few tens of minutes.
        ] + ([
            min(_f(n.get("next_allowance_min")) / 10.0, 1.5),          # 36
            min(_f(n.get("slack_ahead_min")) / 30.0, 1.5),             # 37
        ] if INCLUDE_SLACK else []))

        h, hl = (_journey_history(
                     inst, int(n.get("stop_idx", 0)), t=snap_t,
                     at_station=not bool(n.get("mode_in_transit")))
                 if inst else ([[0.0] * HIST_FEAT_DIM] * MAX_HISTORY, 0))
        hists.append(h)
        hlens.append(max(hl, 1))          # >=1 so the encoder always has a step

        tn = n.get("train_no", "")
        srv.append(srv_i.get(tn, 0))
        cls.append(cls_i.get(n.get("train_class", "mail_express"), 0))
        grp.append(grp_i.get(n.get("train_group", "passenger"), 0))

        rt = n.get("delay_targets") or []
        rm = n.get("delay_target_mask") or []
        tgts.append([float(rt[k]) if k < len(rt) else 0.0 for k in range(MAX_HORIZON)])
        masks.append([float(rm[k]) if k < len(rm) else 0.0 for k in range(MAX_HORIZON)])
        ndep.append(_f(n.get("next_dep_target_min")))
        ndep_m.append(_f(n.get("next_dep_target_mask")))
        # "how much longer will this train stand here" — see build_dataset
        rdw.append(_f(n.get("remaining_dwell_target_min")))
        rdw_m.append(_f(n.get("remaining_dwell_target_mask")))

    base = torch.tensor(feats, dtype=torch.float)
    train_x = torch.cat([base, ev_x], dim=1)
    tgt_abs = torch.tensor(tgts, dtype=torch.float)
    cur = torch.tensor(cur_d, dtype=torch.float)

    return (train_x,
            torch.tensor(srv, dtype=torch.long),
            torch.tensor(cls, dtype=torch.long),
            torch.tensor(grp, dtype=torch.long),
            tgt_abs - cur.unsqueeze(1),
            tgt_abs, cur, scen_y,
            torch.tensor(ndep, dtype=torch.float),
            torch.tensor(ndep_m, dtype=torch.float),
            torch.tensor(rdw, dtype=torch.float),
            torch.tensor(rdw_m, dtype=torch.float),
            torch.tensor(hists, dtype=torch.float),
            torch.tensor(hlens, dtype=torch.long))


# ── Dataset ───────────────────────────────────────────────────────────────────

@dataclass
class CrisSnapshot:
    day_index: int
    split: str
    snapshot_index: int
    data: HeteroData


class CrisDataset:
    """Loader for data_cris/dataset day files."""

    def __init__(self, dataset_dir: str, topology_json: str,
                 verbose: bool = True, max_days: int = 0, lazy: bool = False):
        """`lazy=True` keeps day files on disk and materialises one day at a
        time via `iter_days()`. Holding all 4,944 snapshots as HeteroData needs
        several GB and OOMs on a 17 GB machine mid-training; training only ever
        needs one day resident because memory resets at each day boundary."""
        self.path = Path(dataset_dir)
        topo = json.loads(Path(topology_json).read_text(encoding="utf-8"))

        self._topo_st = {int(k): v for k, v in topo["stations"].items()}
        self._station_ids = sorted(self._topo_st)
        self._st_to_idx = {s: i for i, s in enumerate(self._station_ids)}
        self.num_stations = len(self._station_ids)
        self._junction_ids = set(topo["junction_ids"])

        keys = set()
        for v in topo["sections"].values():
            a, b = int(v["from_id"]), int(v["to_id"])
            keys.add((min(a, b), max(a, b)))
        self._section_keys = sorted(keys)
        self._sec_to_idx = {k: i for i, k in enumerate(self._section_keys)}
        self.num_sections = len(self._section_keys)

        self.train_feat_dim = TRAIN_FEAT_DIM
        self.station_feat_dim = STATION_FEAT_DIM
        self.section_feat_dim = SECTION_FEAT_DIM
        self.hist_feat_dim = HIST_FEAT_DIM
        self.max_history = MAX_HISTORY

        self.class_to_idx = {n: i for i, n in enumerate(TRAIN_CLASS_ORDER)}
        self.group_to_idx = {n: i for i, n in enumerate(TRAIN_GROUP_ORDER)}

        self._track_ei = self._build_track_edges()
        self._dims_checked = False

        date_re = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")
        files = sorted(f for f in self.path.glob("*.json") if date_re.match(f.name))
        if max_days:
            files = files[:max_days]
        if not files:
            raise FileNotFoundError(f"no day files in {self.path}")

        services = sorted({t["train_no"] for t in topo.get("trains", [])})
        self.service_to_idx = {t: i for i, t in enumerate(services)}
        self.num_services = max(len(services), 1)

        self.snapshots_by_split: Dict[str, List[CrisSnapshot]] = {
            "train": [], "validation": [], "test": []}

        self.lazy = lazy
        self._day_files: List[tuple[int, Path, str]] = []

        if lazy:
            # Read only the split label from each day; leave the heavy payload
            # on disk until iter_days() asks for it.
            for di, f in enumerate(files):
                day = json.loads(f.read_text(encoding="utf-8"))
                self._day_files.append((di, f, day.get("split", "train")))
                del day
            counts: Dict[str, int] = {}
            for _, _, sp in self._day_files:
                counts[sp] = counts.get(sp, 0) + 1
            self._lazy_day_counts = counts
        else:
            for di, f in enumerate(files):
                day = json.loads(f.read_text(encoding="utf-8"))
                self._load_day(di, day)
                if verbose and (di % 10 == 0 or di == len(files) - 1):
                    print(f"  loaded day {di + 1}/{len(files)} ({day['date']})")

        if verbose and lazy:
            print(f"\nCrisDataset ({self.path.name}) [lazy]")
            print(f"  Stations {self.num_stations}  Sections {self.num_sections}"
                  f"  Services {self.num_services}")
            print(f"  Train feat {TRAIN_FEAT_DIM} "
                  f"({BASE_TRAIN_FEAT_DIM} base + {EVENT_FEAT_DIM} event)"
                  f" | history {MAX_HISTORY}x{HIST_FEAT_DIM}")
            print(f"  Days — {self._lazy_day_counts}")
        elif verbose:
            print(f"\nCrisDataset ({self.path.name})")
            print(f"  Stations {self.num_stations}  Sections {self.num_sections}"
                  f"  Services {self.num_services}")
            print(f"  Train feat {TRAIN_FEAT_DIM} "
                  f"({BASE_TRAIN_FEAT_DIM} base + {EVENT_FEAT_DIM} event)"
                  f" | history {MAX_HISTORY}x{HIST_FEAT_DIM}")
            print(f"  Snapshots — train {len(self.snapshots_by_split['train'])}"
                  f"  val {len(self.snapshots_by_split['validation'])}"
                  f"  test {len(self.snapshots_by_split['test'])}")

    def _build_track_edges(self) -> torch.Tensor:
        src, dst = [], []
        for (a, b) in self._section_keys:
            ia, ib = self._st_to_idx[a], self._st_to_idx[b]
            src += [ia, ib]
            dst += [ib, ia]
        return torch.tensor([src, dst], dtype=torch.long)

    def _load_day(self, day_idx: int, day: dict) -> None:
        split = day.get("split", "train")
        instances = {i["instance_id"]: i for i in day["train_instances"]}
        routes = {i["instance_id"]: i["route"] for i in day["train_instances"]}

        for si, snap in enumerate(day["snapshots"]):
            data = self.materialise(snap, instances, routes)
            if data is not None:
                self.snapshots_by_split[split].append(
                    CrisSnapshot(day_idx, split, si, data))

    def materialise(self, snap: dict, instances: dict,
                    routes: dict) -> Optional[HeteroData]:
        if not snap.get("train_nodes"):
            return None

        st_x, st_y = _station_features(snap, self._station_ids, self._topo_st)
        sec_x = _section_features(snap, self._section_keys, self._junction_ids)

        # Declared dims must match what the builders actually produce, or the
        # model's input projections silently mis-shape at the first forward.
        if not self._dims_checked:
            for nm, got, want in (("station", st_x.shape[1], STATION_FEAT_DIM),
                                  ("section", sec_x.shape[1], SECTION_FEAT_DIM)):
                if got != want:
                    raise ValueError(
                        f"{nm} feature dim mismatch: builder emits {got}, "
                        f"{nm.upper()}_FEAT_DIM declares {want}")
            self._dims_checked = True

        events, cf_ei, cf_attr = compute_events_and_conflict_edges(
            snap, routes, self._junction_ids)

        (tr_x, srv, cls, grp, tgt_d, tgt_a, cur, scen,
         ndep, ndep_m, rdw, rdw_m, hist, hlen) = _train_features(
            snap, instances, self.class_to_idx, self.group_to_idx,
            self.service_to_idx, events)

        masks = torch.tensor(
            [[float(v) for v in (n.get("delay_target_mask") or [0] * MAX_HORIZON)]
             for n in snap["train_nodes"]], dtype=torch.float)

        d = HeteroData()
        d["station"].x = st_x
        d["station"].y_critical = st_y
        d["section"].x = sec_x
        d["train"].x = tr_x
        d["train"].service_idx = srv
        d["train"].class_idx = cls
        d["train"].group_idx = grp
        # Instance ids, so a trainer can follow the SAME run across ticks —
        # required for scheduled sampling, which must feed a train its own
        # earlier prediction rather than the observed delay.
        d["train"].iid = [n["instance_id"] for n in snap["train_nodes"]]
        # Which stop each train occupies, so scheduled sampling can map a
        # history ROW back to the STOP it describes. The history keeps only the
        # most recent MAX_HISTORY window, so row j is stop
        # max(0, stop_idx + 1 - MAX_HISTORY) + j -- not simply stop j.
        d["train"].stop_idx = [int(n.get("stop_idx", 0))
                               for n in snap["train_nodes"]]
        d["train"].target_delta = tgt_d
        d["train"].target_abs = tgt_a
        d["train"].current_delay = cur
        d["train"].target_mask = masks
        d["train"].scenario_y = scen
        d["train"].y_next_dep = ndep
        d["train"].y_next_dep_mask = ndep_m
        d["train"].y_remaining_dwell = rdw
        d["train"].y_remaining_dwell_mask = rdw_m
        d["train"].history = hist
        d["train"].history_len = hlen

        d["station", "track", "station"].edge_index = self._track_ei

        # train <-> station / section incidence
        at_s, at_t, tr_s, tr_t = [], [], [], []
        ap_s, ap_t = [], []
        for i, n in enumerate(snap["train_nodes"]):
            cur_st = n.get("current_station")
            nxt_st = n.get("next_station")
            if cur_st in self._st_to_idx:
                at_s.append(i)
                at_t.append(self._st_to_idx[cur_st])
            # Direct link to the station this train is HEADING FOR.
            #
            # Without it the train reaches that station only via
            # train->current station->track->neighbour, and `track` is
            # undirected over every adjacent station, so what arrives is a
            # blend of the station ahead and the one just left — the model
            # cannot tell which platform it is actually approaching.
            #
            # This matters because delay on this corridor is created by trains
            # being HELD at stations, not by conflicts in section (spike events:
            # in_transit d=-1.45, time_at_stop d=+1.43, while queue_ahead and
            # junction_overlap run the WRONG way at 0.16x and 0.27x lift).
            # Measured on the station ahead, every congestion feature separates
            # spikes in the physically sensible direction — temporal_congestion
            # d=+0.35, delayed_train_density +0.26, critical_flag +0.25, and the
            # worst delay already standing there 63.5 min vs 38.7. That is the
            # CAUSE of the hold, and it needs a clean one-hop path to the train.
            if nxt_st is not None and nxt_st in self._st_to_idx:
                ap_s.append(i)
                ap_t.append(self._st_to_idx[nxt_st])
            if (n.get("mode_in_transit") and nxt_st is not None
                    and cur_st is not None):
                k = (min(cur_st, nxt_st), max(cur_st, nxt_st))
                if k in self._sec_to_idx:
                    tr_s.append(i)
                    tr_t.append(self._sec_to_idx[k])

        def ei(a, b):
            return (torch.tensor([a, b], dtype=torch.long) if a
                    else torch.zeros(2, 0, dtype=torch.long))

        d["train", "at", "station"].edge_index = ei(at_s, at_t)
        d["station", "has", "train"].edge_index = ei(at_t, at_s)
        # train -> the station it is approaching, and the reverse direction so
        # a congested station also feels the pressure of who is inbound.
        d["train", "approaching", "station"].edge_index = ei(ap_s, ap_t)
        d["station", "awaits", "train"].edge_index = ei(ap_t, ap_s)
        d["train", "traverses", "section"].edge_index = ei(tr_s, tr_t)
        d["section", "carries", "train"].edge_index = ei(tr_t, tr_s)

        sec_sta_s, sec_sta_t = [], []
        for (a, b) in self._section_keys:
            k = self._sec_to_idx[(a, b)]
            sec_sta_s += [k, k]
            sec_sta_t += [self._st_to_idx[a], self._st_to_idx[b]]
        d["section", "connects", "station"].edge_index = ei(sec_sta_s, sec_sta_t)
        d["station", "borders", "section"].edge_index = ei(sec_sta_t, sec_sta_s)

        d["train", "conflicts", "train"].edge_index = cf_ei
        d["train", "conflicts", "train"].edge_attr = cf_attr
        return d

    def iter_days(self, split: str = "train"):
        """Yield one day's snapshots at a time, freeing the previous day.

        In lazy mode this is the only way to read data; in eager mode it just
        regroups what is already resident, so callers work either way.
        """
        if not self.lazy:
            groups: Dict[int, List[CrisSnapshot]] = {}
            for s in self.snapshots_by_split.get(split, []):
                groups.setdefault(s.day_index, []).append(s)
            for k in sorted(groups):
                yield groups[k]
            return

        import gc
        for di, f, sp in self._day_files:
            if sp != split:
                continue
            day = json.loads(f.read_text(encoding="utf-8"))
            instances = {i["instance_id"]: i for i in day["train_instances"]}
            routes = {i["instance_id"]: i["route"] for i in day["train_instances"]}
            out: List[CrisSnapshot] = []
            for si, snap in enumerate(day["snapshots"]):
                data = self.materialise(snap, instances, routes)
                if data is not None:
                    out.append(CrisSnapshot(di, sp, si, data))
            del day, instances, routes
            yield out
            del out
            gc.collect()

    def n_days(self, split: str = "train") -> int:
        if self.lazy:
            return sum(1 for _, _, sp in self._day_files if sp == split)
        return len({s.day_index for s in self.snapshots_by_split.get(split, [])})

    def get_split(self, split: str = "train") -> List[CrisSnapshot]:
        if self.lazy:
            raise RuntimeError("dataset opened lazily — use iter_days(split)")
        return self.snapshots_by_split.get(split, [])

    def __len__(self) -> int:
        return sum(len(v) for v in self.snapshots_by_split.values())
