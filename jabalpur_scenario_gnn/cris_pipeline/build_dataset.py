# -*- coding: utf-8 -*-
"""
cris_pipeline/build_dataset.py
==============================
Turns CRIS corridor journeys into per-day snapshot files for the GNN.

Differences from the previous builder, all driven by measurements recorded in
cris_pipeline/README.md:

  * HALT vs PASS-THROUGH is modelled. Only 23.9% of corridor stop-rows are
    booked halts. A passing train occupies the RUNNING LINE; only a halting
    train occupies a PLATFORM. The old builder counted every train at a
    station as a platform occupant, which is why `occupancy_ratio` was
    meaningless. Targets are still set at every station the train moves
    through — halt or not.

  * FREIGHT OCCUPANCY is injected from the goods data (12,895 section legs +
    6,409 station dwells). The old dataset had zero freight, which is why
    `trains_in_section` was 99.4% zero.

  * DISRUPTION STATE (maintenance blocks, asset failures, TSR) drives
    `is_disrupted` and `current_multiplier`, both hard constants before.

  * JOURNEY POSITION is preserved (`stop_idx`) so the model can rebuild the
    per-halt delay-propagation history for the run so far. The history itself
    is NOT duplicated into every snapshot — dataset.py reconstructs it from
    `train_instances`, which keeps the day files ~20x smaller.

  * ACTOR-ONLY TRAINS (nominal-schedule specials, freight) contribute
    occupancy and conflicts but carry `target_mask=0`.

Usage:
    python -m cris_pipeline.build_dataset
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path

from .config import (
    JOURNEYS_DIR, TOPOLOGY_PATH, DATASET_DIR,
    N_TRAIN_DAYS, N_VAL_DAYS, N_TEST_DAYS, JCT_ETA_CLAMP_MIN,
)
from .overlays import FreightOverlay, DisruptionOverlay, DetentionOverlay

MAX_HORIZON = 8          # corridor journeys are long enough to supervise 8 hops

# Module-level topology
_STATIONS: dict[int, dict] = {}
_SECTIONS: dict[str, dict] = {}
_CODE_TO_ID: dict[str, int] = {}
_JUNCTION_IDS: set[int] = set()
_TARGET_IDS: set[int] = set()      # stations that can carry a target
_GRAPH_IDS: list[int] = []         # every station node in the graph


def _init_topology(topo: dict) -> None:
    global _STATIONS, _SECTIONS, _CODE_TO_ID, _JUNCTION_IDS, _TARGET_IDS, _GRAPH_IDS
    _STATIONS = {int(k): v for k, v in topo["stations"].items()}
    _SECTIONS = topo["sections"]
    _CODE_TO_ID = {k: int(v) for k, v in topo["code_to_id"].items()}
    _JUNCTION_IDS = set(topo["junction_ids"])
    _TARGET_IDS = {sid for sid, s in _STATIONS.items() if not s.get("is_block_post")}
    _GRAPH_IDS = sorted(_STATIONS)


def _sec_key(a: int, b: int) -> str:
    return f"{min(a, b)},{max(a, b)}"


def _section(a: int, b: int) -> dict | None:
    return _SECTIONS.get(_sec_key(a, b))


def _base_run(a: int, b: int) -> float:
    s = _section(a, b)
    return float(s["base_run_min"]) if s else 6.0


# ── Snapshot cadence ──────────────────────────────────────────────────────────

TICK_MIN = 3        # uniform cadence, chosen from the measured leg durations

# Longest stand we are willing to supervise or to simulate. The observed
# maximum in the CRIS data is 184 min; anything beyond that is a data artefact
# rather than a hold, and in the rollout this doubles as the guard that stops a
# train sitting on a platform forever because the model keeps saying "a bit
# longer".
MAX_REMAINING_DWELL_MIN = 190.0

# Does a standing train's delay GROW as it sits past its booked departure?
#
# False reproduces every checkpoint v2..v12 exactly. The rollout rebuilds
# snapshots live through this module, so flipping this unconditionally would
# silently change what v6 sees at inference and quietly invalidate every
# number in releases/cris-v6. It is set from the checkpoint's own config, so
# a model is always scored under the convention it was trained on.
HONEST_STANDING_DELAY = False


def _make_schedule() -> list[int]:
    """Tick cadence over a day.

    Was an adaptive 5/10/15/30 min schedule, which silently destroyed a quarter
    of the supervision: a train that passed two stations between ticks never
    produced a training example for those legs at all. Measured on the corridor,
    station-to-station legs run p25 6 min / p50 8 min, so a coarse tick skips
    the fast legs — exactly the ones where delay is being gained or lost.

        tick  3 min ->  0.3% of legs skipped
        tick  5 min -> 13.9%
        tick 10 min -> 57.7%   (roughly the old average)

    3 min buys effectively complete journey coverage for ~2.8x the snapshots.
    Empty ticks cost nothing — snapshots with no trains are dropped by main().
    """
    return list(range(0, 1440, TICK_MIN))


SNAPSHOT_SCHEDULE = _make_schedule()


# ── Journey loading ───────────────────────────────────────────────────────────

def _load_journeys(journeys_dir: Path) -> dict[str, list[dict]]:
    """{corridor_date: [journey, ...]}"""
    by_date: dict[str, list[dict]] = defaultdict(list)
    for train_dir in sorted(journeys_dir.iterdir()):
        if not train_dir.is_dir():
            continue
        for f in sorted(train_dir.glob("*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if rec.get("stops"):
                by_date[rec["corridor_date"]].append(rec)
    return by_date


def _build_instance(rec: dict, day_idx: int) -> dict | None:
    """Journey -> instance in ABSOLUTE minutes (day_idx * 1440 + relative)."""
    off = day_idx * 1440
    stops = []

    for s in rec["stops"]:
        sid = s.get("station_id")
        if sid is None:
            sid = _CODE_TO_ID.get(s.get("station_code", ""))
        if sid is None:
            continue

        sa = s.get("sched_arr_min")
        sd = s.get("sched_dep_min", sa)
        if sa is None and sd is None:
            continue
        sa = sa if sa is not None else sd
        sd = sd if sd is not None else sa

        # Observed times when present; otherwise fall back to schedule shifted
        # by the delay carried into this stop (filled after the loop).
        aa = s.get("actual_arr_min")
        ad = s.get("actual_dep_min")

        stops.append({
            "station":        int(sid),
            "station_code":   s.get("station_code"),
            "is_target":      bool(s.get("is_division_stop")) and int(sid) in _TARGET_IDS,
            "is_booked_halt": bool(s.get("is_booked_halt")),
            "has_actual":     bool(s.get("has_actual")),
            "sched_arr_abs":  float(sa) + off,
            "sched_dep_abs":  float(sd) + off,
            "actual_arr_abs": (None if aa is None else float(aa) + off),
            "actual_dep_abs": (None if ad is None else float(ad) + off),
            "arr_delay":      s.get("arr_delay_min"),
            "dep_delay":      s.get("dep_delay_min"),
            "distance_km":    s.get("distance_km"),
            # Timetable slack at this stop (pathing + engineering allowance).
            # Schedule-only, so it is known for every future stop.
            "sched_allowance_min": s.get("sched_allowance_min"),
        })

    if len(stops) < 2:
        return None

    # ── Delay estimate for unobserved stops ───────────────────────────────────
    # ~19% of corridor stops carry no actual time. The estimate is used for the
    # train's POSITION and its `delay_min` feature; targets remain gated on
    # has_actual, so nothing is invented as supervision.
    #
    # Carrying 0.0 into the gaps (the obvious implementation) is badly wrong.
    # Train 11705 on 2025-09-28 ran ~800 min late for its whole corridor
    # traverse but had no report at its first four stops: defaulting to 0
    # produced a fake "on time -> +811 min" jump at the fifth, and placed the
    # train in the graph ~13 hours before it was really there, polluting every
    # other train's occupancy and conflict features. 192 such events (2.7% of
    # h1 targets) carried 66% of total absolute error.
    #
    # So: forward-fill from the last observation, and BACK-fill leading gaps
    # from the first observation — a train 811 min late at SHR was already
    # ~811 late at JBP ten minutes earlier.
    # The leading gap must be filled from something known BEFORE the corridor,
    # not from the first in-corridor report. Filling it from stops[obs[0]] is
    # what this used to do, and it reads the future: a train sitting at its
    # first corridor stop with no report was given the delay it would be
    # observed with several stops later. audit_raw_causality catches it, and
    # because the filled value also sets actual_arr_abs it moves _find_position
    # too, so the train appears somewhere the future decided -- which is why a
    # schedule-only field like is_booked_halt showed a difference.
    #
    # Upstream entry delay is the causal substitute: it is observed outside the
    # corridor, strictly before any of this. 35.4% of gapped journeys have it.
    # Where there is no upstream either, we genuinely do not know, and 0 is the
    # honest statement of that -- see the 11705 note above for why 0 was
    # rejected as a GLOBAL default, which is a different question from using it
    # only when nothing observable exists.
    obs = [i for i, st in enumerate(stops) if st["arr_delay"] is not None]
    if obs and obs[0] == 0:
        carried = float(stops[0]["arr_delay"])
    else:
        entry = (rec.get("entry") or {})
        carried = (float(entry.get("entry_delay_min") or 0.0)
                   if entry.get("has_upstream") else 0.0)
    for st in stops:
        # `delay_est_arr` is the estimate that is legitimate for a train
        # STANDING here right now: everything observed strictly earlier, plus
        # this stop's own ARRIVAL. It must never absorb this stop's departure,
        # because a standing train has not departed yet — that value is its
        # future. `delay_est` (below) may, and is only safe to read once the
        # train has actually left, i.e. in_transit.
        #
        # 0.64% of stops report a departure but no arrival, which is exactly
        # where the two diverge. Small, but it leaked into delay_now ->
        # station.max_train_delay and eta_next, and audit_causality's
        # shift_pending_departure probe catches it.
        st["delay_est_arr"] = round(
            float(st["arr_delay"]) if st["arr_delay"] is not None else carried,
            2)
        if st["arr_delay"] is not None:
            carried = float(st["arr_delay"])
        elif st["dep_delay"] is not None:
            carried = float(st["dep_delay"])
        st["delay_est"] = round(carried, 2)
        # Position filling uses the arrival-safe value for the arrival, so a
        # missing arrival is not back-dated from a departure that has not
        # happened. The departure fill may use the full estimate: it is only
        # ever read once the train has left.
        if st["actual_arr_abs"] is None:
            st["actual_arr_abs"] = st["sched_arr_abs"] + st["delay_est_arr"]
        if st["actual_dep_abs"] is None:
            st["actual_dep_abs"] = max(st["sched_dep_abs"] + carried,
                                       st["actual_arr_abs"])
    # Enforce monotonicity so the position finder cannot go backwards.
    for i in range(1, len(stops)):
        stops[i]["actual_arr_abs"] = max(stops[i]["actual_arr_abs"],
                                         stops[i - 1]["actual_dep_abs"])
        stops[i]["actual_dep_abs"] = max(stops[i]["actual_dep_abs"],
                                         stops[i]["actual_arr_abs"])

    return {
        "instance_id":   f"{rec['train_no']}_{rec['corridor_date']}",
        "train_no":      rec["train_no"],
        "train_class":   rec.get("train_class", "mail_express"),
        "train_group":   rec.get("train_group", "passenger"),
        "train_type":    rec.get("train_type", ""),
        "priority":      float(rec.get("priority", 0.65)),
        "speed_factor":  float(rec.get("speed_factor", 0.89)),
        "direction":     rec.get("direction", "DN"),
        "n_coaches":     int(rec.get("n_coaches", 0)),
        "target_eligible": bool(rec.get("is_target_eligible", True)),
        "entry":         rec.get("entry", {}),
        "route":         [s["station"] for s in stops],
        "stops":         stops,
        "corridor_date": rec["corridor_date"],
        "journey_date":  rec.get("journey_date"),
    }


# ── Position ──────────────────────────────────────────────────────────────────

# Stages of a stand. One supervised example per stage, so a long hold teaches
# the model at several points on the evidence curve without swamping it.
_DWELL_STAGE_EDGES = (5.0, 15.0, 30.0)


def _dwell_stage(time_at_stop_min: float) -> int:
    t = float(time_at_stop_min or 0.0)
    for k, e in enumerate(_DWELL_STAGE_EDGES):
        if t < e:
            return k
    return len(_DWELL_STAGE_EDGES)


def _find_position(inst: dict, t: float) -> dict | None:
    stops = inst["stops"]
    if t < stops[0]["actual_arr_abs"]:
        return None
    if t > stops[-1]["actual_dep_abs"] + 30:
        return None

    for i, s in enumerate(stops):
        if t <= s["actual_arr_abs"]:
            j = max(i - 1, 0)
            return {"mode": "in_transit", "stop_idx": j,
                    "station": stops[j]["station"], "next_station": s["station"]}
        if t <= s["actual_dep_abs"]:
            return {"mode": "at_station", "stop_idx": i, "station": s["station"],
                    "next_station": (stops[i + 1]["station"]
                                     if i + 1 < len(stops) else None)}
    return None


def _junction_etas(inst: dict, stop_idx: int, at_station: bool = False,
                   t: float | None = None) -> list[tuple[float, int]]:
    stops = inst["stops"]
    if stop_idx >= len(stops):
        return []
    dep_delay = _causal_dep_delay(stops[stop_idx], t, at_station)
    base = float(stops[stop_idx]["sched_dep_abs"])
    out = []
    for k in range(stop_idx + 1, len(stops)):
        if stops[k]["station"] in _JUNCTION_IDS:
            leg = max(0.0, float(stops[k]["sched_arr_abs"]) - base)
            out.append((round(dep_delay + leg, 2), stops[k]["station"]))
    return sorted(out)


# How far ahead to look for a train that will overtake us, and how close two
# trains must be at the same station to count as a precedence interaction.
PATH_ETA_HORIZON_MIN = 75.0
PRECEDENCE_WINDOW_MIN = 20.0
PRIORITY_MARGIN = 0.05


def _causal_dep_delay(stop: dict, t: float | None, at_station: bool) -> float:
    """Departure delay from `stop` using only what is known at time `t`.

    `stop["dep_delay"]` is the OBSERVED departure delay. For a train already in
    transit that is past and causal. For a train still STANDING at the station
    it is the future — and it is exactly the quantity that says how long the
    hold will last, so reading it is reading the answer.

    Caught on 22190 at UDR: three minutes into a fourteen-minute hold, its ETAs
    were already built on the +36 departure delay it would not incur for
    another eleven minutes. `audit_causality.py` cannot see this, because it
    perturbs stops AFTER the current position and this is the current stop's
    own field.

    Causal replacement: a train still standing at t is at least (t - booked
    departure) late, and no less late than it arrived. Both known now, and it
    grows as the train continues to sit.
    """
    if not at_station or t is None:
        return float(stop.get("dep_delay") or 0.0)
    arr = float(stop.get("arr_delay") or 0.0)
    return max(arr, float(t) - float(stop["sched_dep_abs"]))


def _path_etas(inst: dict, stop_idx: int, at_station: bool = False,
               t: float | None = None) -> list[tuple[int, float]]:
    """ETA to EVERY upcoming station within the horizon, not only junctions.

    `_junction_etas` covers 4 of the 22 corridor stations, and every conflict
    feature was built on it. Measured on the corridor, that is precisely the
    wrong 4: a hold (dwell extension > 10 min) coincides with a higher-priority
    train passing through the same station 58.7% of the time versus 1.2% at
    stops with no hold — a 50x lift — and the NON-junction stations have the
    HIGHEST rates (SBD 83.0%, DDCE 81.8%, DOE 80.5%, NWR 75.4%) while the
    junctions we actually watch are the lowest (JBP 47.3%, STA 41.3%,
    KTE 40.6%). 56% of holds happen where we never looked.

    Worked example — 22190 on 2025-09-30 was held 14 min at UDR to let 12670
    (prio 0.78) and then 12293, a Duronto (prio 0.91 vs its own 0.78), pass;
    it departed at the exact minute the Duronto cleared. UDR is not a junction,
    so `higher_priority_overlap_count` read 0 at every tick of that hold.

    Schedule-based and causal: uses the departure delay already observed at the
    current stop plus BOOKED running times.
    """
    stops = inst["stops"]
    if stop_idx >= len(stops):
        return []
    dep_delay = _causal_dep_delay(stops[stop_idx], t, at_station)
    base = float(stops[stop_idx]["sched_dep_abs"])
    out = []
    # A train STANDING at a station is still competing for that station, so its
    # own stop must be in the list with ETA 0. Omitting it made precedence
    # invisible exactly while the hold was happening: on the 22190/UDR case
    # `higher_priority_overlap_count` read 2 while approaching, then 1, then 0
    # as it sat there waiting for the Duronto that had not yet arrived. The
    # model has to keep predicting throughout the hold — above all the dwell
    # extension — so it needs to see that the train it is waiting for is still
    # to come. Excluded when in transit: that stop is behind us.
    if at_station:
        out.append((stops[stop_idx]["station"], 0.0))
    for k in range(stop_idx + 1, len(stops)):
        leg = max(0.0, float(stops[k]["sched_arr_abs"]) - base)
        eta = dep_delay + leg
        if eta > PATH_ETA_HORIZON_MIN:
            break
        out.append((stops[k]["station"], round(eta, 2)))
    return out


class _StationState:
    def __init__(self):
        self.cong = defaultdict(float)
        self.crit = defaultdict(bool)
        self.steps = defaultdict(int)

    def update(self, sid: int, cong: float, is_crit: bool) -> None:
        self.cong[sid] = round(cong, 4)
        self.crit[sid] = is_crit
        self.steps[sid] = 0 if is_crit else min(self.steps[sid] + 1, 20)


# ── Snapshot ──────────────────────────────────────────────────────────────────

def build_snapshot(day_idx, t, instances, st_state, day_meta,
                   prev_delay, freight, disrupt, date_str,
                   seen_targets: set | None = None) -> dict:
    """`seen_targets` persists across the ticks of one day and holds
    (instance_id, stop_idx, mode) keys that have already been supervised.

    Without it, a train standing at a platform for 40 minutes emits the same
    (journey prefix -> next stop) example at every tick. Measured on the old
    3-30 min schedule, 45.4% of consecutive supervised samples were the SAME
    train at the SAME stop, so nearly half the training signal said "nothing
    changed" — which is precisely the behaviour the model learned and then
    could not shake in rollout.

    Keyed by MODE as well as position, so each prefix still contributes both
    real decision points — one while the train stands at the station and one
    while it runs the leg — but never more than that.
    """
    minute_of_day = t % 1440
    dow = day_meta["day_of_week"]
    tod_sin = math.sin(2 * math.pi * minute_of_day / 1440.0)
    tod_cos = math.cos(2 * math.pi * minute_of_day / 1440.0)
    dow_sin = math.sin(2 * math.pi * dow / 7.0)
    dow_cos = math.cos(2 * math.pi * dow / 7.0)

    platform_occ = defaultdict(int)      # halting trains only
    line_occ     = defaultdict(int)      # passing trains at a station
    stn_max_delay = defaultdict(float)
    delayed_cnt   = defaultdict(int)
    approaching   = defaultdict(int)
    jct_pressure  = defaultdict(float)
    sec_occ       = defaultdict(int)
    nodes = []

    for inst in instances:
        pos = _find_position(inst, t)
        if pos is None:
            continue

        stops, i = inst["stops"], pos["stop_idx"]
        stop, mode = stops[i], pos["mode"]
        cur, nxt = pos["station"], pos["next_station"]

        # Observed delay where we have it, otherwise the carried estimate —
        # never a silent 0, which would fabricate an on-time train.
        d_arr = stop.get("arr_delay")
        d_dep = stop.get("dep_delay")
        if mode == "in_transit":
            # The train has left, so BOTH timestamps at this stop are in the
            # past and either may be read.
            primary, secondary = d_dep, d_arr
            fallback = float(stop.get("delay_est", 0.0))
        else:
            # Standing here now. Its departure has NOT happened, so d_dep is
            # the future and must not be touched — not as a value, not as a
            # fallback. See delay_est_arr in _build_instance.
            primary, secondary = d_arr, None
            fallback = float(stop.get("delay_est_arr",
                                      stop.get("delay_est", 0.0)))
        if primary is not None:
            delay_now = float(primary)
        elif secondary is not None:
            delay_now = float(secondary)
        else:
            delay_now = fallback
        delay_observed = int(primary is not None or secondary is not None)

        # THE HONEST DELAY FOR A STANDING TRAIN.
        #
        # `delay_now` above is the ARRIVAL delay at this stop. Once the train
        # pulls in, that number is fixed and never moves again, however long
        # the train then sits there. Measured over every stand in the data:
        #
        #     minutes standing   we said   truth   missed
        #     0-3                   92.9    87.9      0
        #     6-15                 118.9   121.1      2
        #     15-30                177.3   192.0     15
        #     30+                  192.2   236.8     45
        #
        # A train 45 minutes past its booked departure was presented to the
        # model as the 20 minutes late it had been on arrival. 78% of every
        # delay rise over 30 min on this corridor IS standing time, so the
        # single most informative quantity was the one held constant.
        #
        # The honest number is what a controller reads off the board: how late
        # this train would be if it left right now. Both terms are the current
        # clock and the printed timetable -- nothing from the future.
        #
        # max(), not replace: early in a booked dwell the clock has not yet
        # reached the booked departure, and the arrival delay is still the
        # operative figure.
        if (HONEST_STANDING_DELAY and mode == "at_station"
                and stop.get("sched_dep_abs") is not None):
            delay_now = max(delay_now, float(t) - float(stop["sched_dep_abs"]))

        # Platform vs running line — only a booked halt occupies a platform.
        if mode == "at_station":
            if stop["is_booked_halt"]:
                platform_occ[cur] += 1
            else:
                line_occ[cur] += 1
        stn_max_delay[cur] = max(stn_max_delay[cur], delay_now)
        if delay_now >= 8.0:
            delayed_cnt[cur] += 1

        if mode == "in_transit" and nxt is not None:
            sec_occ[_sec_key(cur, nxt)] += 1

        # ETA to the next station — CAUSAL ESTIMATE ONLY.
        #
        # This used to be `stops[i+1]["actual_arr_abs"] - t`, i.e. the OBSERVED
        # arrival time at the very stop being predicted. That is a target leak:
        #     eta_next + minimum_arrival_delay_next
        #       = (actual_arr - t) + (t - sched_arr)
        #       = actual_arr - sched_arr
        #       = arr_delay  == THE TARGET
        # Measured on the built dataset, 87.15% of h1 targets (99.91% of the
        # in-transit ones) were reconstructable to the exact minute from those
        # two dims. The model was reading the answer, not forecasting it; it is
        # what made a 4 h-horizon rollout score MAE 1.6 min.
        #
        # Replacement uses only what is knowable at time t: when the train
        # actually left (past, observed) or is expected to leave if it is still
        # standing, plus the BOOKED running time for the leg.
        eta_next = 0.0
        if i + 1 < len(stops):
            sched_leg = max(0.0, float(stops[i + 1]["sched_arr_abs"])
                            - float(stop["sched_dep_abs"]))
            if mode == "in_transit":
                exp_dep = float(stop["actual_dep_abs"])       # already gone
            else:
                exp_dep = max(float(t), float(stop["sched_dep_abs"]) + delay_now)
            eta_next = max(0.0, exp_dep + sched_leg - t)
            if eta_next <= 25.0:
                approaching[stops[i + 1]["station"]] += 1

        _at = (mode == "at_station")
        jetas = _junction_etas(inst, i, at_station=_at, t=t)
        petas = _path_etas(inst, i, at_station=_at, t=t)
        for eta_j, jid in jetas:
            if eta_j < 150:
                jct_pressure[jid] += math.exp(-eta_j / 35.0)

        eta_1 = jetas[0][0] if jetas else JCT_ETA_CLAMP_MIN
        eta_2 = jetas[1][0] if len(jetas) > 1 else JCT_ETA_CLAMP_MIN

        iid = inst["instance_id"]
        vel = delay_now - prev_delay.get(iid, delay_now)
        prev_delay[iid] = delay_now

        dwell = (max(0.0, t - float(stop["actual_arr_abs"]))
                 if mode == "at_station" else 0.0)

        # What the TIMETABLE says about the stop we are predicting into.
        #
        # The next_dep head predicts departure = arrival + dwell, so it has to
        # guess a dwell — but without these two dims it cannot know whether a
        # dwell is even scheduled there. The same observed number means
        # opposite things: a 5-minute stand at a booked halt is routine, at a
        # pass-through it is a hold starting. Measured on the training split, a
        # booked halt overruns its booked dwell 48.4% of the time against 4.9%
        # for a pass-through — a 10x difference in what the number implies.
        #
        # Both come straight from the schedule, so they are known in advance
        # for every future stop and are safe under rollout (no actuals read).
        nxt_stop = stops[i + 1] if i + 1 < len(stops) else None
        if nxt_stop is not None:
            next_booked_halt = int(nxt_stop["is_booked_halt"])
            next_sched_dwell = max(0.0, float(nxt_stop["sched_dep_abs"])
                                        - float(nxt_stop["sched_arr_abs"]))
        else:
            next_booked_halt, next_sched_dwell = 0, 0.0

        # Timetable slack: at the stop being predicted, and cumulatively over
        # the rest of the journey. Both are pure schedule, so they are known
        # for every future stop and safe under rollout — same footing as
        # next_sched_dwell above.
        #
        # This is the model's only view of WHERE it is allowed to make time
        # back. Without it, a padded section and a genuinely slow one look
        # identical, which is why recovery was the weakest thing the model did.
        next_allowance = (float(nxt_stop.get("sched_allowance_min") or 0.0)
                          if nxt_stop is not None else 0.0)
        slack_ahead = sum(float(s.get("sched_allowance_min") or 0.0)
                          for s in stops[i + 1:])

        mins_since_dep = run_over = min_arr_delay_next = 0.0
        if mode == "in_transit" and i + 1 < len(stops):
            nx = stops[i + 1]
            mins_since_dep = max(0.0, t - float(stop["actual_dep_abs"]))
            sched_leg = max(0.0, float(nx["sched_arr_abs"]) - float(stop["sched_dep_abs"]))
            run_over = max(0.0, mins_since_dep - sched_leg)
            min_arr_delay_next = t - float(nx["sched_arr_abs"])

        # Freight and disruption on the leg being run right now
        f_sec = f_stn = f_appr = 0
        dis_sev = 0.0
        dis_mult = 1.0
        if nxt is not None:
            key = _sec_key(cur, nxt)
            f_sec = freight.section_count(date_str, key, minute_of_day)
            dis_sev, dis_mult = disrupt.state(date_str, key, minute_of_day)
        f_stn = freight.station_count(date_str, cur, minute_of_day)
        if nxt is not None:
            f_appr = freight.approaching_count(date_str, nxt, minute_of_day)

        entry = inst.get("entry", {})
        # Upstream SHAPE — see build_journeys._entry_features. Constant
        # for the run, but it conditions every stop in it.
        up_vol = float(entry.get("upstream_volatility") or 0.0)
        up_rec = float(entry.get("upstream_recovered_min") or 0.0)
        up_r5  = float(entry.get("upstream_rate5") or 0.0)
        up_n   = float(entry.get("upstream_stops_observed") or 0.0)

        nodes.append({
            "instance_id": iid,
            "train_no": inst["train_no"],
            "train_class": inst["train_class"],
            "train_group": inst["train_group"],
            "current_station": cur,
            "next_station": nxt,
            "mode_in_transit": int(mode == "in_transit"),
            "at_booked_halt": int(stop["is_booked_halt"]),
            "next_booked_halt": next_booked_halt,
            "next_sched_dwell_min": round(next_sched_dwell, 2),
            "next_allowance_min": round(next_allowance, 2),
            "slack_ahead_min": round(slack_ahead, 2),
            "headway_ahead_min": 0.0,   # filled by the conflict pass
            "headway_behind_min": 0.0,
            "upstream_volatility": round(up_vol, 2),
            "upstream_recovered_min": round(up_rec, 2),
            "upstream_rate5": round(up_r5, 2),
            "upstream_stops_n": round(up_n, 2),
            "delay_min": round(delay_now, 2),
            # 1 = this stop actually reported; 0 = carried estimate. Lets the
            # model discount a delay it is inferring rather than observing.
            "delay_observed": delay_observed,
            "delay_velocity": round(vel, 2),
            "time_at_stop_min": round(dwell, 2),
            "eta_next_min": round(eta_next, 2),
            "eta_nearest_jct_min": round(eta_1, 2),
            "eta_second_jct_min": round(eta_2, 2),
            "jcts_remaining": len(jetas),
            "priority": inst["priority"],
            "speed_factor": inst["speed_factor"],
            "n_coaches": inst["n_coaches"],
            "stops_remaining": len(stops) - 1 - i,
            "route_progress_norm": round(i / max(len(stops) - 1, 1), 4),
            "minutes_since_departure": round(mins_since_dep, 2),
            "minutes_over_scheduled_run": round(run_over, 2),
            "minimum_arrival_delay_next": round(min_arr_delay_next, 2),
            # Freight / disruption — new channels
            "freight_in_section": f_sec,
            "freight_at_station": f_stn,
            "freight_approaching_next": f_appr,
            "section_disrupted": round(dis_sev, 3),
            "section_run_multiplier": round(dis_mult, 3),
            # Inherited delay from outside the corridor
            "entry_delay_min": float(entry.get("entry_delay_min", 0.0)),
            "entry_trend_min": float(entry.get("upstream_delay_trend_min", 0.0)),
            "entry_km_before": float(entry.get("km_before_corridor", 0.0)),
            "has_upstream": int(bool(entry.get("has_upstream"))),
            # Conflict counters, filled below
            "junction_overlap_count": 0,
            "same_line_overlap_count": 0,
            "higher_priority_overlap_count": 0,
            "queue_ahead_count": 0,
            "min_junction_gap_min": JCT_ETA_CLAMP_MIN,
            "tod_sin": round(tod_sin, 4),
            "tod_cos": round(tod_cos, 4),
            # Position in the journey — dataset.py rebuilds the propagation
            # history from this rather than duplicating it into every snapshot.
            "stop_idx": i,
            "target_eligible": int(inst["target_eligible"]),
            "precedence_eta_min": PATH_ETA_HORIZON_MIN,
            "_jetas": jetas,
            "_petas": petas,
        })

    # ── Path conflicts: EVERY upcoming station, not only junctions ────────────
    # See `_path_etas` for the measurement that forced this. The counters keep
    # their names and meaning; only their SCOPE changed, from 4 junctions to
    # all 22 corridor stations, which is where 56% of holds actually occur.
    for a in nodes:
        ea = dict(a["_petas"])
        gap_min = JCT_ETA_CLAMP_MIN
        prec_gap = PATH_ETA_HORIZON_MIN          # to the next train that outranks us
        # SIGNED HEADWAY. `min_junction_gap_min` below takes abs(x - y), so a
        # train eight minutes AHEAD of us and one eight minutes BEHIND produce
        # the identical feature -- and they mean opposite things. The one in
        # front can stop us dead; the one behind cannot touch us.
        #
        # Two independent studies name headway THE determinant of delay
        # propagation on a corridor (the GAT propagation work, and RSTGCN on
        # the Indian network, which introduces hourly headway as its key new
        # traffic feature). We were destroying the sign of the quantity the
        # literature says matters most.
        #
        # ahead_gap: minutes until the nearest train IN FRONT reaches a station
        #            we are also heading for. Small = we are closing on it.
        # behind_gap: the same for the nearest train behind us -- kept separate
        #            because it matters for whether WE will be held to let it
        #            through, not for whether we are blocked.
        ahead_gap = behind_gap = PATH_ETA_HORIZON_MIN
        for b in nodes:
            if a is b:
                continue
            eb = dict(b["_petas"])
            for s, x in ea.items():
                y = eb.get(s)
                if y is None:
                    continue
                gap = abs(x - y)
                if s in _JUNCTION_IDS:
                    gap_min = min(gap_min, gap)
                # y < x  =>  b reaches this station BEFORE us: it is ahead.
                if y < x:
                    ahead_gap = min(ahead_gap, x - y)
                else:
                    behind_gap = min(behind_gap, y - x)
                if gap <= PRECEDENCE_WINDOW_MIN:
                    a["junction_overlap_count"] += 1
                    if b["train_group"] == a["train_group"]:
                        a["same_line_overlap_count"] += 1
                    if b["priority"] > a["priority"] + PRIORITY_MARGIN:
                        a["higher_priority_overlap_count"] += 1
                        # How soon the overtaking happens is the operative
                        # signal: a Duronto due at my next stop in 8 minutes
                        # means I am about to be put inside for it.
                        prec_gap = min(prec_gap, x)
                if 0.0 <= (y - x) <= 15.0:
                    a["queue_ahead_count"] += 1
        a["min_junction_gap_min"] = round(gap_min, 2)
        a["precedence_eta_min"] = round(prec_gap, 2)
        a["headway_ahead_min"] = round(ahead_gap, 2)
        a["headway_behind_min"] = round(behind_gap, 2)

    # ── Targets: every station the train moves through ────────────────────────
    by_iid = {i["instance_id"]: i for i in instances}
    for n in nodes:
        inst = by_iid[n["instance_id"]]
        stops, i = inst["stops"], n["stop_idx"]

        # One example per (journey prefix, mode) per day — see the docstring.
        # The node still enters the graph and still contributes occupancy and
        # conflict pressure; only its SUPERVISION is suppressed.
        # STAGED SUPERVISION.
        #
        # The key used to be (instance, stop, mode), which caps a train-stop at
        # two lessons however long the train stands. Measured: nodes per
        # train-stop median 3 / max 37, supervised median 1 / max 2, and only
        # 25.8% of train nodes carried any loss at all. Worse, the surviving
        # lesson is always the FIRST tick, so `time_at_stop` at every
        # supervised standing node was at most 3 minutes -- while the rollout
        # queries the model at 45-minute stands. 1,347 observations of trains
        # standing over 15 minutes received zero gradient.
        #
        # So the model was tested only at the one moment the answer is
        # unknowable, and never once in the window where the evidence is
        # overwhelming. It learned the only thing that window teaches: when a
        # train has just arrived, guess small.
        #
        # Adding the dwell stage to the key gives roughly five lessons across a
        # long hold instead of one, each at a genuinely different point on the
        # evidence curve, while a clean pass-through still contributes one.
        # Capping the stages is deliberate -- supervising every tick would let
        # a 40-minute hold emit 13 near-identical easy examples and drown the
        # hard first one, which is the failure the original dedup was written
        # to prevent.
        fresh = True
        if seen_targets is not None:
            key = (n["instance_id"], i, n["mode_in_transit"],
                   _dwell_stage(n.get("time_at_stop_min", 0.0))
                   if HONEST_STANDING_DELAY else 0)
            fresh = key not in seen_targets
            if fresh:
                seen_targets.add(key)

        tg, mk = [], []
        for h in range(MAX_HORIZON):
            k = i + h + 1
            ok = (fresh and k < len(stops) and stops[k]["is_target"]
                  and stops[k]["has_actual"] and stops[k]["arr_delay"] is not None
                  and inst["target_eligible"])
            tg.append(round(float(stops[k]["arr_delay"]), 2) if ok else 0.0)
            mk.append(1 if ok else 0)
        n["delay_targets"] = tg
        n["delay_target_mask"] = mk
        n["next_delay_target_min"] = tg[0] if mk[0] else 0.0

        # Departure delay at the next stop — lets the rollout continue past it.
        dep_t, dep_m = 0.0, 0
        if mk[0]:
            s_n = stops[i + 1]
            dv = s_n.get("dep_delay")
            dep_t = float(dv if dv is not None else s_n["arr_delay"])
            dep_m = 1
        n["next_dep_target_min"] = round(dep_t, 2)
        n["next_dep_target_mask"] = dep_m

        # HOW MUCH LONGER WILL THIS TRAIN STAND HERE?
        #
        # The question the model was never asked. It is asked when it will
        # reach the next station and how long it will stop THERE -- both about
        # the future -- and never about the present.
        #
        # That omission is why a hold cannot survive in a rollout. A train's
        # departure from a stop is written when the model is still one stop
        # BACK, before it has arrived, and is never revisited. So a hold that
        # begins after T0 cannot lengthen, the platform frees on schedule, the
        # queue behind never forms, and the cascade we exist to forecast
        # cannot be generated.
        #
        # It is also a far better posed problem than the one we have been
        # losing at. Delay rises over 30 min number 295 in the whole training
        # split; standing observations number 18,964. Same physical event, far
        # denser supervision -- because a spike IS a long stand, and asking
        # "how much longer" turns a rare-event regression into a duration
        # estimate. The signal is strong and monotone:
        #
        #     minutes already stood   median wait left   p90
        #     0-5                            0            10
        #     10-15                          5            27
        #     20-25                         10            38
        #     30+                           12            44
        #
        # SCOPE BUG, fixed 2026-08-05 -- read this before touching the lines
        # below. This block first shipped as
        #
        #     if mode == "at_station" and stop.get("actual_dep_abs") ...
        #
        # but this is the TARGETS loop, which binds `stops` and `i` and never
        # binds `stop` or `mode`. Both leaked from the node-construction loop
        # above and held the LAST train processed in the snapshot, so every
        # node got one identical target: an unrelated train's departure minus
        # the clock. Symptoms in dataset_v13: 788 in-transit nodes carrying a
        # dwell target, and all 81 snapshots of a day sharing a single target
        # value. The head trained on it correctly collapsed to a constant.
        # Always index this stop explicitly.
        #
        # Gated on `fresh` so a stand contributes one lesson per dwell stage
        # rather than one per tick -- a 40 min hold would otherwise emit 13
        # near-identical examples and outweigh the 30 stops around it.
        #
        # Causal: `actual_dep_abs` here is this stop's own departure, which is
        # the value being PREDICTED, never read as an input. build_snapshot
        # deliberately refuses to read it as a feature (see the at_station
        # branch of the delay block above) precisely because it is the future.
        rem_t, rem_m = 0.0, 0
        if fresh and not n["mode_in_transit"]:
            s_here = stops[i]
            if s_here.get("actual_dep_abs") is not None:
                rem = float(s_here["actual_dep_abs"]) - float(t)
                if 0.0 <= rem <= MAX_REMAINING_DWELL_MIN:
                    rem_t, rem_m = rem, 1
        n["remaining_dwell_target_min"] = round(rem_t, 2)
        n["remaining_dwell_target_mask"] = rem_m
        n.pop("_jetas", None)
        n.pop("_petas", None)

    # ── Station nodes ─────────────────────────────────────────────────────────
    station_nodes = []
    for sid in _GRAPH_IDS:
        st = _STATIONS[sid]
        plats = max(int(st.get("platforms", 0)), 0)
        cap = max(int(st.get("holding_capacity", 2)), 1)

        p_occ = platform_occ.get(sid, 0)
        l_occ = line_occ.get(sid, 0)
        f_occ = freight.station_count(date_str, sid, minute_of_day)
        total_occ = p_occ + l_occ + f_occ

        # Platform pressure only means something where platforms exist.
        plat_ratio = (min(1.0, p_occ / plats) if plats else 0.0)
        cap_ratio = min(1.0, total_occ / cap)
        press = min(1.0, jct_pressure.get(sid, 0.0) / 5.0)
        maxd = stn_max_delay.get(sid, 0.0)
        dens = min(1.0, delayed_cnt.get(sid, 0) / cap)
        appr = min(1.0, (approaching.get(sid, 0)
                         + freight.approaching_count(date_str, sid, minute_of_day)) / 5.0)

        is_crit = cap_ratio > 0.85 or maxd > 30.0
        cong = st_state.cong[sid]
        steps = st_state.steps[sid]
        st_state.update(sid, cap_ratio * 0.5 + press * 0.5, is_crit)

        station_nodes.append({
            "station_id": sid,
            "code": st["code"],
            "platform_occupancy_ratio": round(plat_ratio, 4),
            "capacity_occupancy_ratio": round(cap_ratio, 4),
            "available_capacity_ratio": round(max(0.0, (cap - total_occ) / cap), 4),
            "freight_occupancy": f_occ,
            "passing_train_count": l_occ,
            "incoming_pressure": round(press, 4),
            "max_train_delay_min": round(maxd, 2),
            "delayed_train_density": round(dens, 4),
            "approach_density": round(appr, 4),
            "congestion_score": round(cong, 4),
            "temporal_congestion_level": round(cong * 0.7 + dens * 0.3, 4),
            "steps_since_critical_norm": round(min(1.0, steps / 20.0), 4),
            "critical_flag": int(is_crit),
            "tod_sin": round(tod_sin, 4), "tod_cos": round(tod_cos, 4),
            "dow_sin": round(dow_sin, 4), "dow_cos": round(dow_cos, 4),
        })

    # ── Section edges ─────────────────────────────────────────────────────────
    section_edges = []
    for key, sec in _SECTIONS.items():
        a, b = int(sec["from_id"]), int(sec["to_id"])
        k = _sec_key(a, b)
        pax = sec_occ.get(k, 0)
        frt = freight.section_count(date_str, k, minute_of_day)
        sev, mult = disrupt.state(date_str, k, minute_of_day)
        cap = max(int(sec.get("main_lines", 2)), 1)
        section_edges.append({
            "from_station": a, "to_station": b,
            "base_run_min_norm": round(min(1.0, sec["base_run_min"] / 20.0), 4),
            "trains_in_section_norm": round(min(1.5, (pax + frt) / cap), 4),
            "freight_in_section_norm": round(min(1.5, frt / cap), 4),
            "current_multiplier": round(mult, 3),
            "is_disrupted": round(sev, 3),
            "psr_time_loss_norm": round(min(1.0, sec.get("psr_time_loss_min", 0.0) / 60.0), 4),
            "tod_sin": round(tod_sin, 4), "tod_cos": round(tod_cos, 4),
        })

    return {
        "timestamp_abs_min": float(t),
        "minute_of_day": float(minute_of_day),
        "train_nodes": nodes,
        "station_nodes": station_nodes,
        "section_edges": section_edges,
    }


# ── Driver ────────────────────────────────────────────────────────────────────

def _drop_partial_days(by_date: dict[str, list], dates: list[str],
                       min_trains: int = 70) -> list[str]:
    """Drop truncated days at the extract boundaries.

    Journeys are filed by the day they reach the corridor, so the first day is
    missing every train that departed the day before the extract starts, and
    the tail days are missing trains that had not yet arrived. Measured on this
    extract: 2025-09-01 has 33 trains and 2025-10-02 has 4, against a steady
    state of 80-100. Leaving them in would put a near-empty day in the test
    split and quietly flatter the metrics.
    """
    kept = [d for d in dates if len(by_date[d]) >= min_trains]
    dropped = [f"{d}({len(by_date[d])})" for d in dates if d not in set(kept)]
    if dropped:
        print(f"  dropping partial boundary days: {', '.join(dropped)}")
    return kept


def _assign_splits(dates: list[str]) -> dict[str, str]:
    """Chronological, counted back from the end — a forecaster must be tested
    forward in time, and the most recent days are the ones we care about."""
    n = len(dates)
    n_test = min(N_TEST_DAYS, max(1, n - 2))
    n_val = min(N_VAL_DAYS, max(1, n - n_test - 1))
    out = {}
    for i, d in enumerate(dates):
        if i >= n - n_test:
            out[d] = "test"
        elif i >= n - n_test - n_val:
            out[d] = "validation"
        else:
            out[d] = "train"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journeys", default=str(JOURNEYS_DIR))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--out", default=str(DATASET_DIR))
    ap.add_argument("--honest-standing-delay", action="store_true",
                    help="a standing train's delay grows past its booked "
                         "departure, and stands are supervised in stages. "
                         "The v13 convention; build to a NEW --out.")
    ap.add_argument("--keep-existing", action="store_true",
                    help="do NOT delete day files from a previous build "
                         "first. Almost never what you want.")
    ap.add_argument("--max-days", type=int, default=0)
    args = ap.parse_args()

    global HONEST_STANDING_DELAY
    HONEST_STANDING_DELAY = bool(args.honest_standing_delay)
    print(f"  honest standing delay: {HONEST_STANDING_DELAY}")

    topo = json.loads(Path(args.topology).read_text(encoding="utf-8"))
    _init_topology(topo)
    print(f"Topology: {len(_STATIONS)} stations ({len(_TARGET_IDS)} targetable), "
          f"{len(_SECTIONS)} sections, {len(_JUNCTION_IDS)} junctions")

    print("Loading overlays ...")
    freight = FreightOverlay.load()
    section_pairs = {(int(s["from_id"]), int(s["to_id"]))
                     for s in _SECTIONS.values()}
    disrupt = DisruptionOverlay.load(section_pairs)
    detent = DetentionOverlay.load()
    print(f"  freight    : {freight.n_legs} legs, {freight.n_dwells} dwells")
    print(f"  disruption : {disrupt.n_blocks} blocks, {disrupt.n_failures} failures")
    print(f"  detentions : {detent.n} records")

    by_date = _load_journeys(Path(args.journeys))
    dates = sorted(by_date)
    dates = _drop_partial_days(by_date, dates)
    if args.max_days:
        dates = dates[:args.max_days]
    splits = _assign_splits(dates)
    n_by_split = defaultdict(int)
    for d in dates:
        n_by_split[splits[d]] += 1
    print(f"Dates: {len(dates)}  ({dates[0]} .. {dates[-1]})  "
          f"train={n_by_split['train']} val={n_by_split['validation']} "
          f"test={n_by_split['test']}")

    out = Path(args.out)
    # Wipe stale day files. A rebuild writes one file per day it produces, so
    # a day that existed in a previous build but not this one SURVIVES and is
    # silently picked up by everything that globs this directory. That is not
    # hypothetical: a corrupted build produced 30 days, the clean rebuild
    # produced 29, and the orphaned 2025-10-01.json carried 4,087 duplicate
    # train instances into every integrity check until it was found by hand.
    if out.exists() and not args.keep_existing:
        for stale in out.glob("*.json"):
            stale.unlink()
    out.mkdir(parents=True, exist_ok=True)

    n_days = n_snaps = n_nodes = n_targets = 0
    for di, date in enumerate(dates):
        d_obj = dt.date.fromisoformat(date)
        day_meta = {"day_of_week": d_obj.weekday()}

        instances = []
        for rec in by_date[date]:
            inst = _build_instance(rec, di)
            if inst:
                det = detent.get(inst["train_no"], inst.get("journey_date") or date)
                inst["detention"] = det or {}
                instances.append(inst)
        if not instances:
            continue

        st_state = _StationState()
        prev_delay: dict[str, float] = {}
        seen_targets: set = set()
        snaps = []
        for m in SNAPSHOT_SCHEDULE:
            snap = build_snapshot(di, di * 1440 + m, instances, st_state,
                                  day_meta, prev_delay, freight, disrupt, date,
                                  seen_targets)
            if snap["train_nodes"]:
                snaps.append(snap)
        if not snaps:
            continue

        day_obj = {
            "day_index": di,
            "date": date,
            "split": splits[date],
            "day_of_week": d_obj.weekday(),
            "is_weekend": d_obj.weekday() >= 5,
            "train_instances": instances,
            "snapshots": snaps,
        }
        (out / f"{date}.json").write_text(json.dumps(day_obj, separators=(",", ":")),
                                          encoding="utf-8")
        n_days += 1
        n_snaps += len(snaps)
        n_nodes += sum(len(s["train_nodes"]) for s in snaps)
        n_targets += sum(sum(n["delay_target_mask"][0] for n in s["train_nodes"])
                         for s in snaps)
        print(f"  {date} [{splits[date][:5]:5}] {len(instances):3d} trains, "
              f"{len(snaps):3d} snapshots", flush=True)

    meta = {
        "name": "JBP-STA corridor (official CRIS)",
        "version": "cris_v1",
        "num_days": n_days,
        "date_range": {"start": dates[0], "end": dates[-1]},
        "split": {"train": N_TRAIN_DAYS, "validation": N_VAL_DAYS,
                  "test": N_TEST_DAYS, "strategy": "chronological"},
        "num_stations": len(_GRAPH_IDS),
        "num_sections": len(_SECTIONS),
        "num_services": len(topo.get("trains", [])),
        "max_horizon": MAX_HORIZON,
        "snapshot_schedule": SNAPSHOT_SCHEDULE,
        "honest_standing_delay": HONEST_STANDING_DELAY,
        "station_ids": _GRAPH_IDS,
        "target_station_ids": sorted(_TARGET_IDS),
        "junction_ids": sorted(_JUNCTION_IDS),
        "train_class_order": topo.get("train_class_order", []),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone. days={n_days} snapshots={n_snaps:,} train-nodes={n_nodes:,} "
          f"h1-targets={n_targets:,}")
    print(f"  avg trains/snapshot: {n_nodes / max(n_snaps, 1):.1f}")
    print(f"  output: {out}")


if __name__ == "__main__":
    main()
