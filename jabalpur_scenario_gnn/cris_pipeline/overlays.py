# -*- coding: utf-8 -*-
"""
cris_pipeline/overlays.py
=========================
Exogenous state the passenger timetable cannot express, indexed so a snapshot
can ask "what is happening on this section / at this station, right now?".

Three overlays, all new to this pipeline:

  FreightOverlay     — goods occupancy per block section and per station.
                       The old dataset contained ZERO freight, which is why
                       `trains_in_section` was 99.4% zero and the station
                       congestion channel was ~95% zero. Freight is the main
                       perturbation source on a mixed-traffic corridor.

  DisruptionOverlay  — maintenance blocks, asset failures and TSRs. These give
                       `is_disrupted` and `current_multiplier` real values;
                       both were hard constants before.

  DetentionOverlay   — CRIS detention records (cause + minutes). Used as an
                       auxiliary supervision signal, never as an input feature
                       (it is only known after the fact).
"""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from .config import (
    GOODS_CSV, MAINT_CSV, ASSET_CSV, TSR_CSV, DETENTION_CSV,
    CORRIDOR, CORRIDOR_ORDER, DUPLICATE_SECTIONS,
)

CODE_TO_ID = {c: i for i, c in enumerate(CORRIDOR_ORDER)}


def _sec_key(a: int, b: int) -> str:
    return f"{min(a, b)},{max(a, b)}"


def _add_interval(store: dict, key, start, end, value=None) -> int:
    """Register [start, end] into `store[date][key]`, SPLIT ACROSS EVERY DAY.

    Snapshots are queried as (date_string, minute_of_day) with minute_of_day in
    [0, 1440). An interval was previously filed only under the day it STARTED,
    with minutes measured from that day's midnight, so anything running past
    00:00 simply vanished on the following day -- a goods train standing at a
    platform from 23:40 to 01:00 stopped occupying it at midnight while it was
    still physically there.

    Measured before this fix: 306 of 6,409 station dwells (4.8%) and 129 of
    12,895 section legs crossed midnight, losing a median 65 and 10 minutes
    respectively, with one dwell losing 4,198 minutes.

    Splitting per day makes the interval visible on every day it covers, with
    each piece expressed in that day's own minutes. Disruptions use the same
    helper: none of them currently cross midnight, but relying on that staying
    true is not a plan.

    `start`/`end` are pandas Timestamps. Returns the number of pieces written.
    """
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0
    n = 0
    day = start.normalize()
    while day <= end.normalize():
        lo = max(start, day)
        hi = min(end, day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        s_min = (lo - day).total_seconds() / 60.0
        e_min = (hi - day).total_seconds() / 60.0
        if e_min >= s_min:
            rec = (s_min, e_min) if value is None else (s_min, e_min, value)
            store[day.strftime("%Y-%m-%d")][key].append(rec)
            n += 1
        day = day + pd.Timedelta(days=1)
    return n


def _parse_section_string(s) -> tuple[str, str] | None:
    """'JKE-PKRD' -> ('JKE','PKRD') when both ends are on the corridor."""
    if not isinstance(s, str) or "-" not in s:
        return None
    parts = [p.strip().upper() for p in s.split("-")]
    if len(parts) != 2:
        return None
    a, b = parts
    if a in CORRIDOR and b in CORRIDOR:
        return a, b
    return None


def _expand_section(a: str, b: str) -> list[tuple[str, str]]:
    """Resolve an aggregate section onto the sections the topology actually has.

    CRIS names the track through a block post two ways — the subdivided legs
    ('BUU-GNWA' + 'GNWA-UDR') and one aggregate spanning it ('BUU-UDR') — and
    `config.DUPLICATE_SECTIONS` documents the exact distance arithmetic proving
    they are the same rails. `build_topology` keeps the subdivision and DROPS
    the aggregate, so the graph has no BUU-UDR edge.

    This module did not know that. `_parse_section_string` accepts 'BUU-UDR'
    (both ends are on the corridor), `_sec_key` turns it into a key no journey
    ever queries, and the event is stored in a dict nobody reads: no error, no
    warning, no counter. Measured on the current extract, that silently
    discarded 101 of 435 corridor disruptions — 23% — including 80 maintenance
    blocks totalling 7,417 minutes of blocked track, one of them a 163-minute
    ENGG-OPENLINE closure on 2025-09-10.

    It is the aggregate form that operational staff actually write: across
    asset_failure, maintenance, TSR and goods running, the aggregate appears
    2,242 times against 1 for the subdivision. So this is the normal case.

    Expansion is safe because a block post has no platform and no loop — a
    train cannot be held or turned back there — so a disruption anywhere
    between the two real stations blocks the whole stretch, hence both halves.

    Pairs that are not aggregates are returned unchanged.
    """
    if frozenset((a, b)) not in DUPLICATE_SECTIONS:
        return [(a, b)]
    i, j = CORRIDOR_ORDER.index(a), CORRIDOR_ORDER.index(b)
    if i > j:
        i, j = j, i
    return [(CORRIDOR_ORDER[k], CORRIDOR_ORDER[k + 1]) for k in range(i, j)]


_SECTION_KM: dict[tuple[str, str], float] = {}


def _section_km(a: str, b: str) -> float:
    """distance_km for one real section, from the built topology.

    Loaded once and cached. Falls back to 1.0 (equal weighting) if the topology
    has not been built yet, so importing this module never depends on it.
    """
    global _SECTION_KM
    if not _SECTION_KM:
        try:
            import json
            from .config import TOPOLOGY_PATH
            topo = json.load(open(TOPOLOGY_PATH, encoding="utf-8"))
            for sec in topo["sections"].values():
                km = float(sec.get("distance_km") or 0.0) or 1.0
                _SECTION_KM[(sec["from_code"], sec["to_code"])] = km
                _SECTION_KM[(sec["to_code"], sec["from_code"])] = km
        except Exception:                                       # pragma: no cover
            _SECTION_KM = {"__missing__": 1.0}
    return float(_SECTION_KM.get((a, b), 1.0))


def _split_leg(a: str, b: str, start, end) -> list[tuple[tuple[str, str], object, object]]:
    """Cut one movement across an aggregate section into its real sections.

    Freight NEVER reports at a block post — there are zero goods rows at GNWA,
    MDRR or SNRR — so consecutive goods stops jump 'BUU' -> 'UDR' and the leg
    was stored under the aggregate key the topology dropped. Measured: 2,071 of
    13,029 corridor freight legs (15.9%) landed there, and six of the twenty-one
    real sections (BUU-GNWA, GNWA-UDR, KTES-MDRR, MDRR-NWR, NWR-SNRR, SNRR-SBD
    — 29% of the corridor) reported ZERO freight for the whole month. A train
    following a goods train through that stretch was told the track was clear.

    Unlike a disruption, which closes the whole stretch at once, a moving train
    occupies ONE block section at a time — that is precisely what the block post
    is there for, so two trains can follow each other through. Giving both
    halves the full interval would make one goods train read as two. So the
    interval is cut in proportion to distance: 6.24 km of BUU-GNWA against
    6.42 km of GNWA-UDR puts the boundary at 49.3% of the run.

    Constant speed across the stretch is the assumption. It is the weakest part
    of this, but the two halves of every aggregate here differ by at most 1 km,
    so the boundary moves very little even if the train accelerates.

    Ordinary sections return a single unchanged piece.
    """
    pieces = _expand_section(a, b)
    if len(pieces) == 1:
        return [(pieces[0], start, end)]
    if pd.isna(start) or pd.isna(end) or end <= start:
        return [(p, start, end) for p in pieces]
    # Walk the stretch in the direction of travel, not corridor order.
    if CORRIDOR_ORDER.index(a) > CORRIDOR_ORDER.index(b):
        pieces = [(y, x) for x, y in reversed(pieces)]
    kms = [_section_km(x, y) for x, y in pieces]
    total = sum(kms) or 1.0
    out, cursor = [], start
    span = end - start
    for n, (piece, km) in enumerate(zip(pieces, kms)):
        stop = end if n == len(pieces) - 1 else cursor + span * (km / total)
        out.append((piece, cursor, stop))
        cursor = stop
    return out


# ── Freight ───────────────────────────────────────────────────────────────────

class FreightOverlay:
    """Goods-train occupancy, resolved to (date, section) and (date, station).

    Each freight leg contributes an interval [depart_from, arrive_at]. A
    snapshot at absolute minute t counts every leg whose interval covers t.
    """

    def __init__(self) -> None:
        # date -> section_key -> list[(start_min, end_min)]
        self.section_intervals: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list))
        # date -> station_id -> list[(arrive_min, depart_min)]  OBSERVED
        self.station_intervals: dict[str, dict[int, list]] = defaultdict(
            lambda: defaultdict(list))
        # date -> station_id -> list[(departed_min, arrives_min)] for freight
        # that is IN TRANSIT TOWARD that station.
        #
        # This is how "is a freight due here soon" must be answered. Two wrong
        # ways were tried first:
        #   1. observed arrival within (t, t+45].  Reads the future outright --
        #      you cannot know when a train will arrive before it arrives.
        #   2. BOOKED arrival within (t, t+45].  Causal, but useless: freight
        #      on this corridor arrives a median 324 min after its booked time
        #      and only 2.8% land within +/-30 min. A schedule nobody keeps is
        #      not information.
        # What IS knowable at t: a freight left X at 09:10 and has not yet
        # reached Y. That is an observation, it is directional, and it is
        # exactly the physical fact that matters -- something is occupying the
        # approach. Counting those intervals is causal AND meaningful.
        self.inbound_intervals: dict[str, dict[int, list]] = defaultdict(
            lambda: defaultdict(list))
        self.n_legs = 0
        self.n_dwells = 0

    @classmethod
    def load(cls) -> "FreightOverlay":
        ov = cls()
        try:
            g = pd.read_csv(GOODS_CSV, low_memory=False)
        except Exception as exc:                                # pragma: no cover
            print(f"  ! freight overlay unavailable ({exc})")
            return ov

        g = g[g.station.isin(CORRIDOR)].copy()
        for c in ("arvltime", "dprttime", "schdarvl"):
            g[c] = pd.to_datetime(g[c], errors="coerce")
        g = g.sort_values(["coatrainid", "coatrainsqnc"])

        for _, run in g.groupby("coatrainid", sort=False):
            rows = run.to_dict("records")
            for i, r in enumerate(rows):
                code = str(r["station"])
                sid  = CODE_TO_ID.get(code)
                if sid is None:
                    continue

                arr, dep = r["arvltime"], r["dprttime"]

                # Station dwell — freight sitting in a loop blocks capacity.
                if pd.notna(arr) and pd.notna(dep) and dep > arr:
                    ov.n_dwells += _add_interval(
                        ov.station_intervals, sid, arr, dep)

                # Section transit to the next reported station.
                if i + 1 >= len(rows):
                    continue
                nxt  = rows[i + 1]
                nsid = CODE_TO_ID.get(str(nxt["station"]))
                narr = nxt["arvltime"]
                if nsid is None or pd.isna(dep) or pd.isna(narr) or narr <= dep:
                    continue
                # A goods leg that spans a block post covers TWO real sections;
                # see _split_leg. Ordinary legs come back as a single unchanged
                # piece, so this is a no-op for 84% of legs.
                for (pa, pb), p_dep, p_arr in _split_leg(
                        code, str(nxt["station"]), dep, narr):
                    p_sid, p_nsid = CODE_TO_ID[pa], CODE_TO_ID[pb]
                    ov.n_legs += _add_interval(
                        ov.section_intervals, _sec_key(p_sid, p_nsid),
                        p_dep, p_arr)
                    # Directional: this freight is inbound to the station at the
                    # far end of the piece it is on, until it gets there. Before
                    # the split, a block post never appeared as a destination, so
                    # a train approaching one was told no freight was ahead.
                    _add_interval(ov.inbound_intervals, p_nsid, p_dep, p_arr)

        return ov

    def section_count(self, date: str, key: str, minute: float) -> int:
        return sum(1 for s, e in self.section_intervals.get(date, {}).get(key, [])
                   if s <= minute <= e)

    def station_count(self, date: str, sid: int, minute: float) -> int:
        return sum(1 for s, e in self.station_intervals.get(date, {}).get(sid, [])
                   if s <= minute <= e)

    def approaching_count(self, date: str, sid: int, minute: float,
                          window: float = 45.0) -> int:
        """Freight BOOKED to arrive at `sid` within `window` minutes.

        Counts freight that has ALREADY LEFT the previous station and has not
        yet reached `sid`. Every input is an observation at or before `minute`;
        `window` is retained for signature compatibility and ignored, since
        occupancy of the approach is the signal, not a predicted arrival time.

        The original version counted freight whose OBSERVED arrival fell in
        (minute, minute+window] -- a direct future read feeding
        `freight_approaching_next` (train dim 24) and station
        `approach_density`. No perturbation audit could catch it: both audits
        shift the PASSENGER journey and never touch the freight overlay. It was
        found by reading the code.
        """
        return sum(1 for s, e in self.inbound_intervals.get(date, {}).get(sid, [])
                   if s <= minute <= e)


# ── Disruption ────────────────────────────────────────────────────────────────

class DisruptionOverlay:
    """Maintenance blocks, asset failures and TSRs on corridor sections."""

    def __init__(self) -> None:
        # date -> section_key -> list[(start_min, end_min, severity)]
        self.events: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list))
        # section_key -> permanent speed penalty multiplier (TSR)
        self.tsr_multiplier: dict[str, float] = {}
        self.n_blocks = self.n_failures = self.n_tsr = 0

    @classmethod
    def load(cls, section_pairs: set | None = None) -> "DisruptionOverlay":
        """`section_pairs` is {(station_id_a, station_id_b)} from the topology,
        needed to spread station-level speed restrictions onto their sections."""
        ov = cls()
        ov._load_maintenance()
        ov._load_failures()
        ov._load_tsr(section_pairs)
        return ov

    def _add(self, sec: tuple[str, str], start, end, severity: float) -> None:
        if pd.isna(start):
            return
        if pd.isna(end) or end <= start:
            end = start + pd.Timedelta(minutes=30)
        # An aggregate section spanning a block post becomes the two real
        # sections it is made of; anything else passes through untouched. See
        # _expand_section -- without this the event lands on a key the topology
        # dropped and is silently never read.
        for a_code, b_code in _expand_section(sec[0], sec[1]):
            a, b = CODE_TO_ID[a_code], CODE_TO_ID[b_code]
            # Split per day, same as freight. No disruption in the current
            # extract crosses midnight, but a maintenance block starting 23:00
            # is entirely plausible and would otherwise stop applying at 00:00
            # while the track was still closed.
            _add_interval(self.events, _sec_key(a, b), start, end, severity)

    def _load_maintenance(self) -> None:
        try:
            m = pd.read_csv(MAINT_CSV, low_memory=False)
        except Exception as exc:                                # pragma: no cover
            print(f"  ! maintenance overlay unavailable ({exc})")
            return
        for c in ("permitted_start_time", "permitted_end_time",
                  "actual_clear_time"):
            m[c] = pd.to_datetime(m[c], errors="coerce", dayfirst=True)
        for r in m.itertuples():
            sec = _parse_section_string(r.block_section)
            if sec is None:
                continue
            # A block takes a line out of use entirely — the strongest signal.
            end = (r.actual_clear_time if pd.notna(r.actual_clear_time)
                   else r.permitted_end_time)
            self._add(sec, r.permitted_start_time, end, severity=1.0)
            self.n_blocks += 1

    def _load_failures(self) -> None:
        try:
            a = pd.read_csv(ASSET_CSV, low_memory=False)
        except Exception as exc:                                # pragma: no cover
            print(f"  ! asset-failure overlay unavailable ({exc})")
            return
        a = a[a.division == "JBP"].copy()
        for c in ("event_start_date", "event_end_date"):
            a[c] = pd.to_datetime(a[c], errors="coerce")
        for r in a.itertuples():
            sec = _parse_section_string(r.block_section)
            if sec is None:
                continue
            # Scale severity by how many trains the failure actually detained.
            n_aff = float(r.affected_trains or 0)
            self._add(sec, r.event_start_date, r.event_end_date,
                      severity=min(1.0, 0.4 + 0.1 * n_aff))
            self.n_failures += 1

    def _load_tsr(self, section_pairs: set | None = None) -> None:
        """Permanent/temporary speed restrictions -> run-time multipliers.

        Two forms appear in the CRIS file and both matter:
          * SECTION restrictions ('PTWA-JKE') — applied to that section.
          * STATION restrictions ('JBP', 'KTE', 'STA') — a yard/point
            restriction. Applied to every section incident to that station,
            because a 10 km/h crawl through Jabalpur yard slows every train
            entering or leaving it.

        Section-form rows on this corridor carry a goods speed but a NaN
        passenger speed, so passenger run time is only affected by the
        station-form rows. Ignoring those was leaving the multiplier identical
        to the block severity, making two feature dims exact duplicates.
        """
        try:
            t = pd.read_csv(TSR_CSV, low_memory=False)
        except Exception as exc:                                # pragma: no cover
            print(f"  ! TSR overlay unavailable ({exc})")
            return

        def apply(key: str, spd: float) -> None:
            mult = min(4.0, 100.0 / spd)
            self.tsr_multiplier[key] = max(self.tsr_multiplier.get(key, 1.0), mult)
            self.n_tsr += 1

        for r in t.itertuples():
            spd = r.Passenger_Train_Speed
            if pd.isna(spd) or float(spd) <= 0:
                continue
            spd = float(spd)

            sec = _parse_section_string(r.Block_Section)
            if sec is not None:
                for a_code, b_code in _expand_section(*sec):
                    apply(_sec_key(CODE_TO_ID[a_code], CODE_TO_ID[b_code]), spd)
                continue

            code = str(r.Block_Section).strip().upper()
            if code in CORRIDOR and section_pairs:
                sid = CODE_TO_ID[code]
                for a, b in section_pairs:
                    if sid in (a, b):
                        apply(_sec_key(a, b), spd)

    def state(self, date: str, key: str, minute: float) -> tuple[float, float]:
        """Returns (is_disrupted, run_time_multiplier) for a section right now."""
        sev = 0.0
        for s, e, v in self.events.get(date, {}).get(key, []):
            if s <= minute <= e:
                sev = max(sev, v)
        mult = self.tsr_multiplier.get(key, 1.0)
        if sev > 0.0:
            mult = max(mult, 1.0 + sev)     # a live block at least doubles run time
        return sev, mult


# ── Detention causes (auxiliary supervision) ──────────────────────────────────

class DetentionOverlay:
    """CRIS detention records: (train, date) -> total minutes lost, by cause.

    Only ever used as a TARGET. Detentions are recorded after the fact, so
    exposing them as inputs would leak.
    """

    CAUSE_GROUPS = ["TRAFFIC", "OUT OF PATH", "ALARM CHAIN PULLING",
                    "ENGINEERING", "INCIDENT", "LOCO", "OTHER"]

    def __init__(self) -> None:
        self.by_train_day: dict[tuple[str, str], dict] = {}
        self.n = 0

    @classmethod
    def load(cls) -> "DetentionOverlay":
        ov = cls()
        try:
            d = pd.read_csv(DETENTION_CSV, low_memory=False)
        except Exception as exc:                                # pragma: no cover
            print(f"  ! detention overlay unavailable ({exc})")
            return ov

        d = d[d.DIVISION_CODE.isin(["JBP", "JABALPUR"])].copy()
        d["tn"] = d.TRAIN_NUMBER.astype(str).str.strip().str.zfill(5)
        d["date"] = pd.to_datetime(d.TRAIN_START_DATE, errors="coerce")

        for r in d.itertuples():
            if pd.isna(r.date):
                continue
            key = (r.tn, r.date.strftime("%Y-%m-%d"))
            rec = ov.by_train_day.setdefault(
                key, {g: 0.0 for g in cls.CAUSE_GROUPS} | {"total": 0.0})
            desc = str(r.DETENTION_CODE_DESCR or "").upper()
            grp = next((g for g in cls.CAUSE_GROUPS if g in desc), None)
            if grp is None:
                grp = "LOCO" if "LOCO" in desc else "OTHER"
            mins = float(r.DETENTION_TIME or 0.0)
            rec[grp] += mins
            rec["total"] += mins
            ov.n += 1

        return ov

    def get(self, train_no: str, journey_date: str) -> dict | None:
        return self.by_train_day.get((train_no, journey_date))
