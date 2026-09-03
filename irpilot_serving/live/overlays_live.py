# -*- coding: utf-8 -*-
"""
Live overlays: freight and disruption, built from the DATABASE, at a point in
time, with open-ended events handled honestly.

WHY THIS EXISTS
---------------
The pipeline's own FreightOverlay / DisruptionOverlay are built for a COMPLETE
HISTORICAL FILE. Every train in that file eventually arrived; every block
eventually cleared. Live data is a day in progress, and four things break:

  1. NEVER REFRESHED. state_engine built both overlays once at boot and never
     again, so a block declared after startup was invisible for the rest of the
     day. Measured: severity 1.0 vs 0.0 for the same section and minute,
     depending only on when the process happened to start.

  2. IN-TRANSIT FREIGHT VANISHES. A section leg is only created when BOTH the
     departure and the next arrival are known:
         if pd.isna(narr): continue
     A freight that has left X and not yet reached Y has no arrival time — and
     that is precisely the freight occupying the section right now. On the
     archive that is 0.8% of legs; live it is every freight that currently
     matters, and `trains_in_section` would read ~0 permanently.

  3. STOP ORDER IS ACCIDENTAL. The loader sorts by `coatrainsqnc`, which is
     CONSTANT on this corridor (all 15,292 rows are 1). So the sort does
     nothing and leg construction relies on incidental row order. The CSV
     happens to be ordered — 0 of 2,113 runs out of sequence — but a database
     returns rows in no particular order unless asked. Mis-ordered stops build
     freight legs between stations the train never travelled between.

  4. UNCLEARED BLOCKS END EARLY. The loader takes
         end = actual_clear_time or permitted_end_time
     Live, `actual_clear_time` is NULL until the block clears, so a block that
     overruns its permitted end silently stops existing. 73% of blocks overrun,
     median 5 min, up to 88 min.

THE SHARED FIX
--------------
An event with no end yet is OPEN, and an open event runs to `as_of` — the
moment the data describes. That is the honest statement: "it started, and
nothing has reported it finishing." It invents nothing, and it is bounded,
because `as_of` is a real timestamp.

Rebuilt each feed window, so `as_of` advances and open events grow with it.

Nothing under jabalpur_scenario_gnn/ is modified. These subclass the pipeline's
overlays and are drop-in replacements at the call site.
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


# ── How long may an OPEN event stay open? ────────────────────────────────────
#
# An event with no end runs to `as_of`. Unbounded, that turns a missing report
# into a permanent phantom: 3.3% of corridor goods stops have an arrival and no
# departure, and each would occupy its station for the rest of time.
#
# So open events get a cap, taken from the OBSERVED distribution of completed
# events on this corridor — not from a guess:
#
#   section transit, both times known : p50 17 min, p90 27, p99 49
#   station dwell,   both times known : p50 37 min, p90 144, p99 742
#
# Past the 99th percentile, "the report is missing" is a better explanation than
# "it is still there". The caps are the p99s, rounded.
#
# The same reasoning the pipeline already applies to passenger dwell:
#   MAX_REMAINING_DWELL_MIN = 190.0  # observed max 184; beyond that is an artefact
OPEN_TRANSIT_CAP_MIN = 50.0     # p99 of observed section transit
OPEN_DWELL_CAP_MIN = 750.0      # p99 of observed station dwell

# Same idea for disruptions, from the observed closed events:
#
#   maintenance block, start -> actual clear : p50 75, p90 152, p99 298, max 370
#   asset failure,     start -> end          : p50 20, p90 195, p99 50,396 (!)
#
# Blocks cap cleanly at their p99. Failures do NOT: a p99 of 35 DAYS means the
# tail end-dates are unreliable, not that outages last a month. So failures use
# the p90 instead — past ~3 hours, an unclosed failure is far more likely to be
# a missing close than a live outage.
OPEN_BLOCK_CAP_MIN = 300.0      # p99 of observed block duration
OPEN_FAILURE_CAP_MIN = 200.0    # p90; p99 is corrupt (see above)


def _known(ts, as_of):
    """A timestamp is KNOWN only once it has happened.

    THE LEAK THIS CLOSES. Rows are selected by whether ANY of their timestamps
    falls at or before `as_of` — a freight is returned because it ARRIVED at
    09:00, and the same row also carries a departure at 09:30. Reading that
    departure at 09:10 is reading the future: the overlay would show the
    station emptying at 09:30 before the train had left, and would place the
    freight in the next section using an arrival that had not occurred.

    So every timestamp is masked here. Anything after `as_of` is NaT — not yet
    observed — and the open-event rules take over, which is exactly right: at
    09:10 the honest statement is "it arrived and has not reported leaving".
    """
    if ts is None or pd.isna(ts):
        return pd.NaT
    ts = pd.Timestamp(ts)
    return ts if ts <= as_of else pd.NaT


def _open_end(start, as_of, cap_min):
    """End of an event that has started and not reported finishing.

    Runs to `as_of` — the moment the data describes — but no longer than
    `cap_min` past its start, so a lost report cannot become a permanent
    occupancy. Returns None if the cap has already been exceeded, meaning the
    event should be treated as over.
    """
    limit = start + pd.Timedelta(minutes=cap_min)
    end = min(as_of, limit)
    return end if end > start else None


class LiveFreightOverlay(FreightOverlay):
    """Freight occupancy at a point in time, from the database."""

    @classmethod
    def from_db(cls, conn, as_of: pd.Timestamp,
                lookback_hours: int = 72) -> "LiveFreightOverlay":
        """Build from `goods_running`.

        `lookback_hours` bounds how far back we look for movements that might
        still be running. It is NOT a filter on train_date: freight occupying
        this corridor on a given day frequently started days earlier — measured,
        57 of 80 movements active on 2025-09-27 had an earlier train_date, the
        oldest ten days before. We select on OBSERVED TIME, never on the date
        the movement was booked.
        """
        lo = as_of - pd.Timedelta(hours=lookback_hours)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT coatrainid, sqncnumb, station, arvltime, dprttime
                  FROM {gr}
                 WHERE station = ANY(%s)
                   AND (arvltime BETWEEN %s AND %s
                     OR dprttime BETWEEN %s AND %s)
                 ORDER BY coatrainid, sqncnumb
            """.format(gr=config.NAMES["goods_running"]),
                (list(CORRIDOR), lo, as_of, lo, as_of))
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=["coatrainid", "sqncnumb", "station",
                                         "arvltime", "dprttime"])
        return cls.from_frame(df, as_of)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, as_of: pd.Timestamp) -> "LiveFreightOverlay":
        ov = cls()
        if df.empty:
            return ov
        for c in ("arvltime", "dprttime"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        # FIX 3: order by sqncnumb, the real stop sequence. coatrainsqnc is
        # constant here and sorting by it is a no-op.
        df = df.sort_values(["coatrainid", "sqncnumb"], kind="mergesort")

        for _, run in df.groupby("coatrainid", sort=False):
            rows = run.to_dict("records")
            for i, r in enumerate(rows):
                sid = CODE_TO_ID.get(str(r["station"]))
                if sid is None:
                    continue
                # MASK THE FUTURE. The row was selected because one of its
                # times is in the past; the others may not be.
                arr = _known(r["arvltime"], as_of)
                dep = _known(r["dprttime"], as_of)

                # Station dwell. If it arrived and has not departed, it is
                # STILL STANDING — occupy the station up to as_of, capped.
                if pd.notna(arr):
                    dwell_end = (dep if pd.notna(dep)
                                 else _open_end(arr, as_of, OPEN_DWELL_CAP_MIN))
                    if dwell_end is not None and dwell_end > arr:
                        ov.n_dwells += _add_interval(
                            ov.station_intervals, sid, arr, dwell_end)

                if i + 1 >= len(rows) or pd.isna(dep):
                    continue
                nxt = rows[i + 1]
                nsid = CODE_TO_ID.get(str(nxt["station"]))
                if nsid is None:
                    continue
                narr = _known(nxt["arvltime"], as_of)

                # FIX 2: departed but not yet arrived -> STILL IN SECTION.
                # The parent skips this case entirely, which hides exactly the
                # freight that is on the track right now.
                leg_end = (narr if pd.notna(narr)
                           else _open_end(dep, as_of, OPEN_TRANSIT_CAP_MIN))
                if leg_end is None or leg_end <= dep:
                    continue
                ov.n_legs += _add_interval(
                    ov.section_intervals, _sec_key(sid, nsid), dep, leg_end)
                _add_interval(ov.inbound_intervals, nsid, dep, leg_end)
        return ov


def _check_tsr_dates(rows, _seen=[]) -> None:
    """Fail loudly if the TSR window dates did not parse into something sane.

    On their database `todatetime` is a VARCHAR and we cast it. The cast is
    right only while the text is ISO-like. If it is ever dd/mm/yyyy the cast
    THROWS above the 13th but SILENTLY SWAPS day and month below it, so a
    restriction ending 05/09 reads as 9 May: already expired, quietly dropped,
    and the model believes the track is clear.

    Two things give that away without knowing the true format:
      * an end BEFORE its own start — impossible for a real window
      * a date nowhere near any plausible operating year
    Neither can happen with a correct parse, and both are cheap to test.
    Checked once per process; the table is small and its shape does not change.
    """
    if _seen:
        return
    _seen.append(True)

    bad_order = out_of_range = n = 0
    for row in rows:
        frm, to = row[2], row[3]
        if frm is None or to is None or pd.isna(frm) or pd.isna(to):
            continue
        n += 1
        f, t = pd.Timestamp(frm), pd.Timestamp(to)
        if t < f:
            bad_order += 1
        if not (pd.Timestamp("2000-01-01") <= f <= pd.Timestamp("2100-01-01")):
            out_of_range += 1

    if n and (bad_order or out_of_range):
        raise SystemExit("\n".join([
            f"TSR DATES DID NOT PARSE SANELY: of {n} windows, {bad_order} end"
            f" before they start and {out_of_range} fall outside 2000-2100.",
            f"  '{config.NAMES['tsr_to']}' is very likely the wrong cast for"
            f" this column's text format.",
            "  Override it, e.g.  IRPILOT_TBL_TSR_TO="
            "\"to_timestamp(todatetime,'DD/MM/YYYY HH24:MI')\"",
        ]))


class LiveDisruptionOverlay(DisruptionOverlay):
    """Blocks, failures and speed restrictions in force at a point in time."""

    @classmethod
    def from_db(cls, conn, as_of: pd.Timestamp, section_pairs=None,
                division: str = "JBP",
                lookback_hours: int = 72) -> "LiveDisruptionOverlay":
        ov = cls()
        lo = as_of - pd.Timedelta(hours=lookback_hours)

        with conn.cursor() as cur:
            # ── maintenance blocks ──────────────────────────────────────────
            cur.execute("""
                SELECT block_section, permitted_start_time, permitted_end_time,
                       actual_clear_time
                  FROM {mt}
                 WHERE permitted_start_time BETWEEN %s AND %s
            """.format(mt=config.NAMES["maintenance"]), (lo, as_of))
            for sec_s, start, p_end, cleared in cur.fetchall():
                sec = _parse_section_string(sec_s)
                if sec is None or pd.isna(start):
                    continue
                # FIX 4: a block with no clearance reported has NOT ended.
                # Using permitted_end would make an overrunning block vanish,
                # and 73% of blocks overrun.
                start = pd.Timestamp(start)
                cleared = _known(cleared, as_of)     # a clearance in the future
                end = (cleared if pd.notna(cleared)  # has not been reported yet
                       else _open_end(start, as_of, OPEN_BLOCK_CAP_MIN))
                if end is None:
                    continue
                ov._add(sec, start, end, severity=1.0)
                ov.n_blocks += 1

            # ── asset failures ──────────────────────────────────────────────
            cur.execute("""
                SELECT block_section, event_start_date, event_end_date,
                       affected_trains
                  FROM {af}
                 WHERE division = %s
                   AND event_start_date BETWEEN %s AND %s
            """.format(af=config.NAMES["asset_failure"]), (division, lo, as_of))
            for sec_s, start, end, n_aff in cur.fetchall():
                sec = _parse_section_string(sec_s)
                if sec is None or pd.isna(start):
                    continue
                # Same rule: an unclosed failure is still open.
                start = pd.Timestamp(start)
                end = _known(end, as_of)
                end = (end if pd.notna(end)
                       else _open_end(start, as_of, OPEN_FAILURE_CAP_MIN))
                if end is None:
                    continue
                ov._add(sec, start, end,
                        severity=min(1.0, 0.4 + 0.1 * float(n_aff or 0)))
                ov.n_failures += 1

            # ── temporary speed restrictions ────────────────────────────────
            cur.execute("""
                SELECT block_section, passenger_train_speed,
                       {tf} AS from_datetime, {tt} AS to_datetime
                  FROM {tsr}
            """.format(tf=config.NAMES["tsr_from"],
                       tt=config.NAMES["tsr_to"],
                       tsr=config.NAMES["tsr"]))
            tsr_rows = cur.fetchall()

        _check_tsr_dates(tsr_rows)

        def apply(key, spd):
            mult = min(4.0, 100.0 / spd)
            ov.tsr_multiplier[key] = max(ov.tsr_multiplier.get(key, 1.0), mult)
            ov.n_tsr += 1

        for sec_s, spd, frm, to in tsr_rows:
            if spd is None or float(spd) <= 0:
                continue
            # Respect a real end date once one exists. Every row in the current
            # extract carries no end, so this is a no-op today — but the moment
            # real end dates arrive, an expired restriction must stop applying.
            if to is not None and pd.notna(to) and pd.Timestamp(to) < as_of:
                continue
            if frm is not None and pd.notna(frm) and pd.Timestamp(frm) > as_of:
                continue
            spd = float(spd)
            sec = _parse_section_string(sec_s)
            if sec is not None:
                apply(_sec_key(CODE_TO_ID[sec[0]], CODE_TO_ID[sec[1]]), spd)
                continue
            code = str(sec_s).strip().upper()
            if code in CORRIDOR and section_pairs:
                sid = CODE_TO_ID[code]
                for a, b in section_pairs:
                    if sid in (a, b):
                        apply(_sec_key(a, b), spd)
        return ov
