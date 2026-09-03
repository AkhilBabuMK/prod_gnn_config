# -*- coding: utf-8 -*-
"""
Build corridor journeys FROM THE DATABASE, reusing the pipeline's own
build_journey(). This is the production journey builder: read schedule +
observation from Postgres, shape them into the DataFrame build_journey expects,
and call the trusted function. No merge (production has no old export — the
merge is baked into the seeded data).

The only new code here is "read the DB into the right shape". build_journey
itself is the exact code that produced data_cris/journeys/, so if the DB round
trip is faithful, the journeys are identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import psycopg2
import config

config.bootstrap_gnn_path()
from cris_pipeline.build_journeys import (          # noqa: E402
    build_journey, load_side_tables as load_side_tables_csv, TIME_COLS,
)

# The exact columns build_journey reads off each row, named as load_running names them.
_QUERY = """
    SELECT s.train_number              AS train_no,
           s.train_date                AS train_date,
           s.serial_number             AS serial_number,
           s.station_code              AS station_code,
           s.scheduled_arrival_time    AS scheduled_arrival_time,
           s.scheduled_departure_time  AS scheduled_departure_time,
           o.actual_arrival_time       AS actual_arrival_time,
           o.actual_departure_time     AS actual_departure_time,
           s.train_type                AS train_type,
           s.train_sub_type            AS train_sub_type,
           s.train_source_station      AS train_source_station,
           s.train_destination_station AS train_destination_station,
           s.wtt_stop_flag             AS wtt_stop_flag,
           s.ptt_stoppage_flag         AS ptt_stoppage_flag,
           s.traffic_allowance_seconds AS traffic_allowance_seconds,
           s.engineering_allowance_seconds AS engineering_allowance_seconds,
           s.distance_from_source_km   AS distance_from_source_km
      FROM schedule s
      LEFT JOIN observation o
        ON  o.train_number  = s.train_number
        AND o.train_date    = s.train_date
        AND o.serial_number = s.serial_number
"""

# What was KNOWN at a past instant, for replaying a historical window.
#
# `observation` is the LIVE table: in a replay it already holds the whole month,
# so joining it hands the model timestamps it could not possibly have had. The
# reveal rule is not re-invented here — feed_chunk already carries the exact
# moment CRIS reported each value, and this is the cumulative form of what the
# feed simulator drips into `observation`. Same rule, same result, no simulator.
_OBS_AS_OF = """(
        SELECT train_number, train_date, serial_number,
               max(actual_arrival_time)   AS actual_arrival_time,
               max(actual_departure_time) AS actual_departure_time
          FROM feed_chunk
         WHERE reveal_time <= %s
         GROUP BY train_number, train_date, serial_number
      ) o"""


def read_running_from_db(conn, trains=None, as_of=None) -> pd.DataFrame:
    """Reproduce load_running()'s output shape, but sourced from the DB.

    `trains` restricts to a set of (train_number, train_date). Pass the set that
    changed and the read drops from 624,180 rows to a few hundred.

    It filters WHICH TRAINS, never which of their stops: the entry features are
    computed from the whole upstream run, and 91% of a corridor train's stops
    are upstream of the corridor.

    `as_of` replays a past instant: observed values reported after it are not
    returned at all, so a journey built here holds exactly what was knowable
    then. Leave it None in production — the live `observation` table is already
    the present, and masking it would be a lie in the other direction.
    """
    sql, args = _QUERY, []
    if as_of is not None:
        sql = sql.replace("LEFT JOIN observation o", "LEFT JOIN " + _OBS_AS_OF)
        args.append(as_of)          # first %s in the text, so first in the tuple
    if trains:
        # BOTH SIDES MUST BE FILTERED. Filtering only `schedule` leaves the
        # LEFT JOIN scanning all 624,180 observation rows — the planner picks a
        # parallel hash join and the query takes minutes instead of a second.
        # Repeating the predicate on `observation` inside the ON clause is
        # safe: those rows could never match anyway, since the join keys equal
        # the schedule keys that are already restricted.
        sql = (sql.replace(
            "AND o.serial_number = s.serial_number",
            "AND o.serial_number = s.serial_number"
            " AND o.train_number = ANY(%s) AND o.train_date = ANY(%s)")
            + " WHERE s.train_number = ANY(%s) AND s.train_date = ANY(%s)")
        tns = sorted({t for t, _ in trains})
        dts = sorted({d for _, d in trains})
        # Both predicates sit on the grouping columns, so they push down into
        # the as-of aggregate instead of grouping the whole month first.
        args += [tns, dts, tns, dts]
    with conn.cursor() as cur:
        cur.execute(sql, args or None)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return _shape(pd.DataFrame(rows, columns=cols))


def changed_since_observation(conn, since) -> set:
    """(train_number, train_date) whose OBSERVATION rows were written after
    `since`. The feed simulator stamps ingested_at = now() on every reveal, so
    this is exactly the set the last window touched."""
    with conn.cursor() as cur:
        cur.execute("""SELECT DISTINCT train_number, train_date
                         FROM observation WHERE ingested_at > %s""", (since,))
        return {(r[0], r[1]) for r in cur.fetchall()}


def _shape(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce DB output into exactly what build_journey expects."""
    # Times: DB gives datetime already; normalise to pandas datetime to be safe.
    for c in TIME_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # Flags: load_running leaves them as read from CSV (float, NaN where blank),
    # and build_journey does bool(getattr(...)). Mirror that: NULL -> NaN (not
    # None), so bool(NaN)==True matches the offline behaviour exactly.
    for c in ("wtt_stop_flag", "ptt_stoppage_flag",
              "traffic_allowance_seconds", "engineering_allowance_seconds"):
        df[c] = pd.to_numeric(df[c], errors="coerce")   # <NA>/None -> NaN
    df["distance_from_source_km"] = pd.to_numeric(
        df["distance_from_source_km"], errors="coerce")
    df["serial_number"] = pd.to_numeric(df["serial_number"], errors="coerce")

    # TRAIN NUMBER MUST BE 5 CHARACTERS, ZERO-PADDED.
    # load_running() pads (build_journeys.py:85) and so do BOTH side tables
    # (:110 coaches, :118 platforms). Our seeded database happened to hold
    # padded values already, so this never mattered here — but their table
    # stores the bare number, '138' rather than '00138'.
    #
    # Unpadded, every lookup into coaches and platforms MISSES. Neither raises:
    # a dict lookup just returns the default, so coach count and platform
    # silently go blank for every train and the forecast quietly gets worse.
    # Padding here covers both profiles — it is a no-op on an already-padded
    # value, so the simulation is unaffected.
    df["train_no"] = df["train_no"].astype(str).str.strip().str.zfill(5)

    # Delay derived exactly as load_running.
    df["arr_delay"] = ((df.actual_arrival_time - df.scheduled_arrival_time)
                       .dt.total_seconds() / 60.0)
    df["dep_delay"] = ((df.actual_departure_time - df.scheduled_departure_time)
                       .dt.total_seconds() / 60.0)

    return df.sort_values(["train_no", "train_date", "serial_number"])


def build_all_from_db(conn) -> dict:
    """Return {(train_no, journey_date): journey_record} for every built journey.

    Keyed by journey_date (== train_date), which is UNIQUE per groupby group.
    corridor_date is NOT unique: two origin-dates can reach the corridor on the
    same day (the '#2' collision case), so keying by it would drop one.
    """
    df = read_running_from_db(conn)
    coaches, platforms = load_side_tables_csv()
    out = {}
    for (tn, date), g in df.groupby(["train_no", "train_date"], sort=True):
        rec = build_journey(g, coaches, platforms)
        if rec is None or rec.get("_rejected"):
            continue
        out[(rec["train_no"], rec["journey_date"])] = rec
    return out




if __name__ == "__main__":
    with psycopg2.connect(**config.DB) as c:
        js = build_all_from_db(c)
    print(f"built {len(js):,} journeys from the database")


# ── PRODUCTION: read the trigger-maintained main table ───────────────────────

_MAIN_QUERY = f"""
    SELECT train_number              AS train_no,
           {config.NAMES["train_date"]}   AS train_date,
           serial_number             AS serial_number,
           station_code              AS station_code,
           scheduled_arrival_time    AS scheduled_arrival_time,
           scheduled_departure_time  AS scheduled_departure_time,
           actual_arrival_time       AS actual_arrival_time,
           actual_departure_time     AS actual_departure_time,
           train_type                AS train_type,
           train_sub_type            AS train_sub_type,
           train_source_station      AS train_source_station,
           train_destination_station AS train_destination_station,
           wtt_stop_flag             AS wtt_stop_flag,
           ptt_stoppage_flag         AS ptt_stoppage_flag,
           traffic_allowance_seconds AS traffic_allowance_seconds,
           engineering_allowance_seconds AS engineering_allowance_seconds,
           distance_from_source_km   AS distance_from_source_km
      FROM {config.NAMES["main"]}
"""


def read_main_from_db(conn, trains=None) -> pd.DataFrame:
    """Same output shape as read_running_from_db, from the MAIN TABLE.

    One table instead of a join, because the incremental question — "which
    trains changed since I last looked?" — is an index seek here and a scan of
    both sides otherwise. That question is asked every three minutes forever.

    `trains` restricts to a set of (train_number, train_date). Pass the set
    that changed and we rebuild only those journeys instead of all ~100. Note
    it filters WHICH TRAINS, never which of their stops: the entry features are
    computed from the whole upstream run, and 91% of a corridor train's stops
    are upstream of the corridor.
    """
    sql, args = _MAIN_QUERY, []
    if trains:
        sql += " WHERE (train_number, train_date) IN %s"
        args = [tuple(sorted(trains))]
    with conn.cursor() as cur:
        cur.execute(sql, args or None)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return _shape(pd.DataFrame(rows, columns=cols))


def changed_trains(conn, since, until=None, by_event_time: bool = False) -> set:
    """(train_number, train_date) touched since `since`. The incremental read.

    PRODUCTION uses `changed_at` — when the trigger last wrote the row. That is
    the honest "what is new to me" question and it is an index seek.

    REPLAY of a historical window must use the OBSERVED EVENT TIMES instead.
    `changed_at` is wall-clock: on a replayed September day every row was
    written by the backfill months later, so `changed_at > since` matches
    everything at every window and the incremental read is never exercised.
    Selecting on when the event actually happened reproduces what the feed
    would have delivered in that window.
    """
    with conn.cursor() as cur:
        if by_event_time:
            cur.execute("""
                SELECT DISTINCT train_number, {td} AS train_date FROM {tbl}
                 WHERE (actual_arrival_time   > %s AND actual_arrival_time   <= %s)
                    OR (actual_departure_time > %s AND actual_departure_time <= %s)
            """.format(td=config.NAMES["train_date"], tbl=config.NAMES["main"]),
                (since, until, since, until))
        else:
            cur.execute("""SELECT DISTINCT train_number, {td} AS train_date
                             FROM {tbl} WHERE {ch} > %s""".format(
                                 td=config.NAMES["train_date"],
                                 tbl=config.NAMES["main"],
                                 ch=config.NAMES["changed_at"]), (since,))
        return {(r[0], r[1]) for r in cur.fetchall()}


def newest_event(conn, upto=None):
    """Newest OBSERVED event time held. Used for feed age during replay, where
    `changed_at` reflects when we loaded the data, not when it happened.

    FILTER AND REPORT ON THE SAME EXPRESSION. This used to select rows by
    COALESCE(departure, arrival) and then report GREATEST(arrival, departure).
    15 stops in this data depart BEFORE they arrive — train 15945 at BHGN
    departs 2025-09-26 14:47 and "arrives" 2025-09-28 10:25 — so such a row
    passed the already-happened test on its departure and then reported an
    arrival 43 hours ahead. The feed age came out permanently negative and
    pinned, which silently disabled STALE_WARN and STALE_CRIT: the one alarm
    that catches a feed that has stopped while the forecaster keeps producing
    confident output from a world that has moved on.
    """
    with conn.cursor() as cur:
        if upto is None:
            cur.execute("""SELECT max(GREATEST(actual_arrival_time,
                                               actual_departure_time))
                             FROM {tbl}""".format(tbl=config.NAMES["main"]))
        else:
            cur.execute("""
                SELECT max(t) FROM (
                    SELECT GREATEST(actual_arrival_time,
                                    actual_departure_time) AS t
                      FROM {tbl}) q
                 WHERE t <= %s""".format(tbl=config.NAMES["main"]), (upto,))
        return cur.fetchone()[0]


# ── COACH COUNTS AND PLATFORM ALLOCATIONS ────────────────────────────────────
# The pipeline reads these from two CSVs. That is right offline, where the CSVs
# ARE the source. In production it is not: the railway team holds both as live
# tables, and a shipped CSV is a snapshot that silently ages — a rake gets
# shorter, a platform moves, and we keep answering with last July's value.
#
# Three things make this more than a SELECT:
#
#   TRAIN NUMBERS ARE UNPADDED THERE. '1053', where every lookup key in the
#   pipeline is '01053'. An unpadded key misses on every train and returns
#   nothing: measured at 372 trains losing their coach count and 501 platform
#   allocations disappearing, with no error raised anywhere.
#
#   THERE IS MORE THAN ONE ROW PER TRAIN. Their coaches table carries
#   train_date, so a train appears once per day it ran, and the counts are not
#   always equal — a rake really does change. Take the most recent, not an
#   arbitrary row and not the maximum.
#
#   A NULL COUNT IS NOT A ZERO. Those rows are dropped, not read as
#   zero-length trains.
#
#   THE SOURCE GENUINELY CONFLICTS WITH ITSELF. 65 (train, station) pairs in
#   PF_INFO carry more than one platform, and 4 trains carry two coach counts.
#   The CSV reader resolves this by "last row in the file wins", which is file
#   order and nothing more. A table has no file order, so we resolve by recency
#   and break ties on ctid — the physical row position, which is insertion
#   order for an append-only load. Without that tie-break the answer is
#   whatever the planner felt like, and two runs of the same query could
#   disagree about which platform a train uses.

_RECENCY_COLS = ("train_date", "record_insert_time", "ingested_at")


def _recency_column(conn, table: str) -> str | None:
    """Which column says which row is newest.

    Their tables carry train_date and record_insert_time; ours carries
    ingested_at. Ask the catalogue rather than assume, so one code path serves
    both and a renamed column degrades to "no ordering" instead of an error.
    """
    schema, _, name = table.rpartition(".")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
             WHERE table_name = %s
               AND (%s = '' OR table_schema = %s)
               AND column_name = ANY(%s)""",
                    (name, schema, schema, list(_RECENCY_COLS)))
        found = {r[0] for r in cur.fetchall()}
    return next((c for c in _RECENCY_COLS if c in found), None)


def _load_coaches(conn) -> dict:
    tbl = config.NAMES["coaches"]
    order = _recency_column(conn, tbl)
    by = f"{order} DESC NULLS LAST, ctid DESC" if order else "ctid DESC"
    sql = (f"SELECT DISTINCT ON (train_number) train_number, no_of_coaches"
           f"  FROM {tbl} WHERE no_of_coaches IS NOT NULL"
           f" ORDER BY train_number, {by}")
    with conn.cursor() as cur:
        cur.execute(sql)
        return {str(tn).strip().zfill(5): int(n) for tn, n in cur.fetchall()}


def _load_platforms(conn) -> dict:
    tbl = config.NAMES["pf_info"]
    order = _recency_column(conn, tbl)
    by = f"{order} DESC NULLS LAST, ctid DESC" if order else "ctid DESC"
    sql = (f"SELECT DISTINCT ON (train_number, station_code)"
           f"       train_number, station_code, pf_number"
           f"  FROM {tbl} WHERE pf_number IS NOT NULL"
           f" ORDER BY train_number, station_code, {by}")
    with conn.cursor() as cur:
        cur.execute(sql)
        return {(str(tn).strip().zfill(5), str(st).strip()): str(pf)
                for tn, st, pf in cur.fetchall()}


def load_side_tables(conn, verbose: bool = False):
    """(coaches, platforms) from the database, falling back to the CSVs.

    Each falls back independently: a missing pf_info must not also cost us the
    coach counts. Raises if either ends up empty from BOTH sources — running on
    empty lookups is the silent failure this exists to prevent.
    """
    coaches = platforms = None
    src = {}
    for key, fn in (("coaches", _load_coaches), ("platforms", _load_platforms)):
        try:
            got = fn(conn)
            src[key] = f"{config.NAMES['coaches' if key == 'coaches' else 'pf_info']} ({len(got):,})"
        except Exception as exc:
            conn.rollback()
            got, src[key] = None, f"CSV — table unusable ({type(exc).__name__})"
        if key == "coaches":
            coaches = got
        else:
            platforms = got

    if not coaches or not platforms:
        c_csv, p_csv = load_side_tables_csv()
        if not coaches:
            coaches, src["coaches"] = c_csv, f"CSV ({len(c_csv):,})"
        if not platforms:
            platforms, src["platforms"] = p_csv, f"CSV ({len(p_csv):,})"

    if not coaches or not platforms:
        raise SystemExit("\n".join([
            "COACH OR PLATFORM DATA IS EMPTY from both the database and the"
            " CSVs.",
            "  Empty lookups cost about 14% of trains their coach count and"
            " raise nothing anywhere else.",
            f"  coaches: {src['coaches']}   platforms: {src['platforms']}",
        ]))

    if verbose:
        print(f"  coaches   <- {src['coaches']}")
        print(f"  platforms <- {src['platforms']}")
    return coaches, platforms
