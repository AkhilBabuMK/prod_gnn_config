# -*- coding: utf-8 -*-
"""
Causal overlays for the ROLLOUT: observed up to T0, PLANNED beyond it.

WHAT THE ROLLOUT IS ALLOWED TO KNOW
-----------------------------------
rollout_cris.py states the rule, and it is the right one:

    known    : the working timetable, and planned exogenous events (maintenance
               blocks, TSRs, freight paths). A controller has these at T0.
    predicted: every corridor arrival and departure, and all occupancy and
               conflict state derived from them.

A controller at 09:00 genuinely holds the block plan and the goods paths, so
using them forward is correct, not a leak. But the overlays as built read the
ARCHIVE, so past T0 they answer with what actually happened. Three gaps:

  1. FREIGHT used observed arrival/departure past T0. The docstring says
     "freight paths" — the booked schedule — but the loader indexes `arvltime`
     and `dprttime`. `schdarvl` is parsed and never used to build an interval.

  2. BLOCKS used `actual_clear_time` past T0. That is when the block really
     finished, which nobody knows at T0. The plan is `permitted_end_time`.
     73% of blocks overrun it, median 5 min, p90 32 min.

  3. ASSET FAILURES were visible before they happened. This one is not a
     drift from the rule — it is outside it. A failure is UNPLANNED; at 09:00
     nobody knows a signal will fail at 11:00. 2.7 corridor failures a day,
     each visible up to 4 hours early.

WHAT THIS BUILDS INSTEAD
------------------------
One overlay per forecast, valid at a given T0:

    minute <= T0    exactly what was observed by then
    minute >  T0    freight standing or in transit at T0, carried forward on
                    its typical duration; goods BOOKED to run later, on their
                    booked path; blocks on their PERMITTED window whether they
                    have started or not; TSRs as normal; and NO asset failure
                    that had not already begun.

Your point about scheduled goods is what makes case 3 of the freight rule
necessary: a goods train not yet on the corridor at T0 but booked to arrive
inside the horizon still takes capacity, and ignoring it would understate
congestion. It is included on its booked path, with the honest caveat that
freight here runs a median 324 minutes behind that path.

Nothing under jabalpur_scenario_gnn/ is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                              # noqa: E402

import config                                                    # noqa: E402
config.bootstrap_gnn_path()

from cris_pipeline.config import CORRIDOR                        # noqa: E402
from cris_pipeline.overlays import (                             # noqa: E402
    FreightOverlay, DisruptionOverlay, _add_interval, _sec_key,
    _parse_section_string, CODE_TO_ID,
)

# Forward duration for something already moving at T0, from the OBSERVED
# distribution of completed events on this corridor:
#     section transit  p50 17 min   p90 27   p99 49
#     station dwell    p50 37 min   p90 144  p99 742
# The median is the right choice here: this is a best estimate of when a
# specific train clears, not a bound on how long a lost report may persist.
TYPICAL_TRANSIT_MIN = 17.0
TYPICAL_DWELL_MIN = 37.0


def _ts(v):
    return pd.NaT if v is None else pd.Timestamp(v)


class CausalFreightOverlay(FreightOverlay):
    """Freight as a controller would know it at T0."""

    @classmethod
    def at_t0(cls, df: pd.DataFrame, t0: pd.Timestamp,
              horizon_min: int = 240) -> "CausalFreightOverlay":
        """`df` needs: coatrainid, sqncnumb, station, arvltime, dprttime,
        schdarvl, schddprt.  Observed columns are used only up to `t0`."""
        ov = cls()
        if df.empty:
            return ov
        for c in ("arvltime", "dprttime", "schdarvl", "schddprt"):
            if c not in df.columns:
                df[c] = pd.NaT
            df[c] = pd.to_datetime(df[c], errors="coerce")
        df = df.sort_values(["coatrainid", "sqncnumb"], kind="mergesort")
        horizon_end = t0 + pd.Timedelta(minutes=horizon_min)

        for _, run in df.groupby("coatrainid", sort=False):
            rows = run.to_dict("records")
            for i, r in enumerate(rows):
                sid = CODE_TO_ID.get(str(r["station"]))
                if sid is None:
                    continue
                # Observations are only knowable once they have happened.
                arr = r["arvltime"] if r["arvltime"] <= t0 else pd.NaT
                dep = r["dprttime"] if r["dprttime"] <= t0 else pd.NaT

                # ── station occupancy ───────────────────────────────────────
                if pd.notna(arr):
                    if pd.notna(dep):
                        end = dep                       # fully observed, past
                    else:
                        # STANDING AT T0. It is there now and will clear at
                        # some point; carry it forward on a typical dwell.
                        end = arr + pd.Timedelta(minutes=TYPICAL_DWELL_MIN)
                        end = max(end, t0)
                    if end > arr:
                        ov.n_dwells += _add_interval(
                            ov.station_intervals, sid, arr, end)
                elif pd.notna(r["schdarvl"]) and r["schdarvl"] > t0:
                    # BOOKED TO ARRIVE LATER and not yet seen. It still takes
                    # capacity when it gets here — this is the scheduled goods
                    # case. Booked freight runs a median 324 min late on this
                    # corridor, so treat the timing as weak evidence, not fact.
                    s_arr = r["schdarvl"]
                    if s_arr <= horizon_end:
                        s_dep = (r["schddprt"] if pd.notna(r["schddprt"])
                                 and r["schddprt"] > s_arr
                                 else s_arr + pd.Timedelta(minutes=TYPICAL_DWELL_MIN))
                        ov.n_dwells += _add_interval(
                            ov.station_intervals, sid, s_arr, s_dep)

                # ── section occupancy, this stop -> the next ────────────────
                if i + 1 >= len(rows):
                    continue
                nxt = rows[i + 1]
                nsid = CODE_TO_ID.get(str(nxt["station"]))
                if nsid is None:
                    continue
                key = _sec_key(sid, nsid)
                narr = nxt["arvltime"] if nxt["arvltime"] <= t0 else pd.NaT

                if pd.notna(dep) and pd.notna(narr):
                    leg_start, leg_end = dep, narr            # past, observed
                elif pd.notna(dep):
                    # IN TRANSIT AT T0 — departed, no arrival reported. This is
                    # the freight physically on the section right now, which
                    # the original loader dropped entirely.
                    leg_start = dep
                    leg_end = max(dep + pd.Timedelta(minutes=TYPICAL_TRANSIT_MIN), t0)
                elif (pd.notna(r["schddprt"]) and r["schddprt"] > t0
                      and r["schddprt"] <= horizon_end):
                    # BOOKED to run this leg later.
                    leg_start = r["schddprt"]
                    leg_end = (nxt["schdarvl"]
                               if pd.notna(nxt["schdarvl"])
                               and nxt["schdarvl"] > leg_start
                               else leg_start + pd.Timedelta(minutes=TYPICAL_TRANSIT_MIN))
                else:
                    continue
                if leg_end <= leg_start:
                    continue
                ov.n_legs += _add_interval(ov.section_intervals, key,
                                           leg_start, leg_end)
                _add_interval(ov.inbound_intervals, nsid, leg_start, leg_end)
        return ov


class CausalDisruptionOverlay(DisruptionOverlay):
    """Blocks, failures and TSRs as a controller would know them at T0."""

    @classmethod
    def at_t0(cls, blocks, failures, tsrs, t0: pd.Timestamp,
              section_pairs=None) -> "CausalDisruptionOverlay":
        """`blocks`   : (block_section, permitted_start, permitted_end, actual_clear)
        `failures`: (block_section, start, end, affected_trains)
        `tsrs`    : (block_section, passenger_speed, from_dt, to_dt)"""
        ov = cls()

        for sec_s, p_start, p_end, cleared in blocks:
            sec = _parse_section_string(sec_s)
            p_start, p_end, cleared = _ts(p_start), _ts(p_end), _ts(cleared)
            if sec is None or pd.isna(p_start):
                continue
            # A block is PLANNED, so its window is legitimately known at T0
            # whether it has started or not.
            if pd.notna(cleared) and cleared <= t0:
                end = cleared                    # already cleared, observed
            else:
                # Still running, or not yet started. The plan is what a
                # controller holds — NOT actual_clear_time, which is the real
                # finish and is unknown at T0. 73% of blocks overrun it.
                end = p_end if pd.notna(p_end) else p_start
                end = max(end, t0) if p_start <= t0 else end
            if end > p_start:
                ov._add(sec, p_start, end, severity=1.0)
                ov.n_blocks += 1

        for sec_s, start, end, n_aff in failures:
            sec = _parse_section_string(sec_s)
            start, end = _ts(start), _ts(end)
            if sec is None or pd.isna(start):
                continue
            # A FAILURE IS UNPLANNED. One that has not begun by T0 cannot be
            # known at T0 — including it lets the forecaster see a breakdown
            # up to 4 hours before it happens.
            if start > t0:
                continue
            if pd.notna(end) and end <= t0:
                pass                              # closed, fully observed
            else:
                # Open at T0. We do not know when it ends; hold it to now.
                end = t0
            if end > start:
                ov._add(sec, start, end,
                        severity=min(1.0, 0.4 + 0.1 * float(n_aff or 0)))
                ov.n_failures += 1

        def apply(key, spd):
            mult = min(4.0, 100.0 / spd)
            ov.tsr_multiplier[key] = max(ov.tsr_multiplier.get(key, 1.0), mult)
            ov.n_tsr += 1

        for sec_s, spd, frm, to in tsrs:
            if spd is None or float(spd) <= 0:
                continue
            frm, to = _ts(frm), _ts(to)
            # A restriction is standing infrastructure, known in advance.
            if pd.notna(to) and to < t0:
                continue
            if pd.notna(frm) and frm > t0:
                continue
            sec = _parse_section_string(sec_s)
            if sec is not None:
                apply(_sec_key(CODE_TO_ID[sec[0]], CODE_TO_ID[sec[1]]), float(spd))
                continue
            code = str(sec_s).strip().upper()
            if code in CORRIDOR and section_pairs:
                sid = CODE_TO_ID[code]
                for a, b in section_pairs:
                    if sid in (a, b):
                        apply(_sec_key(a, b), float(spd))
        return ov
