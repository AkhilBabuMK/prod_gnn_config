# -*- coding: utf-8 -*-
"""
THE PRODUCTION LOOP.

    python run_prod.py                       # live, forever
    python run_prod.py --replay 2025-09-27 09:00 11:00   # score a past window

TWO MODES
---------
LIVE (default)
    The clock is the wall clock, shifted onto the day the data belongs to.

    At boot we measure one offset: how far the newest event in the table sits
    from right now. From then on the model clock is wall_clock + offset, so it
    advances in step with real time.

    Feed carrying today's events   -> offset is zero, and this is exactly the
                                      plain live behaviour.
    A historical day pushed in     -> offset is the gap to that day, and the
    real time (a demo)                clock runs across it at real speed.

    The offset is measured once and does not chase the feed. That is the point:
    if reports stop for ten minutes the clock keeps moving, because the trains
    kept moving. Unreported stops are filled with estimates, which is the
    normal path, not a fallback. A clock that froze with the feed would place
    every train where it stood ten minutes ago.

    Staleness is therefore measured separately: real minutes since the newest
    event last advanced. That is the alarm for a feed that has died.

REPLAY  (--replay DATE FROM TO)
    We drive the clock ourselves over a day already in the table, as fast as
    the machine allows — eight hours in about nine minutes. This is how the
    accuracy figures are produced. Observed values are masked to what had been
    reported at each simulated instant, so it cannot see the future.

--wall-clock forces the model clock to today and ignores the data's day. It
exists only to restore the old behaviour; there is no known reason to use it.

    every 5 min   read what changed since last time, rebuild those journeys,
                  refresh the freight and disruption overlays
    every 3 min   step the model one tick, then forecast 4 hours ahead

The 3-minute tick is not a choice — the model was trained on it. Corridor legs
run 6-8 minutes, so a 5-minute tick skips 13.9% of them. The 5-minute read
matches how often the data actually changes.

WHAT SURVIVES A RESTART
-----------------------
The three GRU memories, the station state and prev_delay are checkpointed to
`model_state` every tick and restored on boot. Without that a restart at 14:00
would forecast as though the day began then.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Late corrections. If a corrected timestamp arrives for something the memory has
already absorbed, the right repair is to rewind to a checkpoint and re-run
forward. The checkpoints exist for it; the rewind does not yet.
"""
from __future__ import annotations

import argparse
import datetime as dt
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd                                              # noqa: E402
import psycopg2                                                  # noqa: E402

import config                                                    # noqa: E402
from sim.state_engine import StateEngine                         # noqa: E402
from sim.build_journeys_db import (                               # noqa: E402
    changed_trains, newest_event,
)
from sim.runlock import RunLock                                  # noqa: E402

REAL_TYPES = {"SUF", "MEX", "VNDB", "DRNT"}

# Past this, the data is old enough that the forecast is describing a world
# that no longer exists. Normal worst case is one missed 5-minute window.
STALE_WARN_MIN = 12.0
STALE_CRIT_MIN = 30.0

_stop = False


def _on_signal(signum, frame):
    global _stop
    _stop = True
    print("\n  signal received — finishing this tick, then stopping.")


class Clock:
    """Real time, or a replay window. Same interface either way, so the loop
    code has no idea which it is running under."""

    def __init__(self, start=None, end=None, speed=0.0, accel=1.0):
        self.replay = start is not None
        self.t = start
        self.end = end
        self.speed = speed          # seconds of real sleep per simulated minute
        # Live only. Model minutes per real minute. 1.0 is real time; higher
        # compresses a day so the whole loop can be scored in minutes instead
        # of hours. It shortens the SLEEP between ticks, never the tick itself:
        # the model still steps 3 model-minutes each time, which is the cadence
        # it was trained on. Change that and the numbers stop being comparable.
        self.accel = max(0.0001, float(accel))
        self._next_wake = None      # accelerated mode: the real instant of the
                                    # next tick, so work time does not add on

    def now(self):
        return self.t if self.replay else dt.datetime.now()

    def advance(self, minutes):
        if self.replay:
            self.t += dt.timedelta(minutes=minutes)
            if self.speed:
                time.sleep(minutes * self.speed)
        else:
            nxt = self.t + dt.timedelta(minutes=minutes) if self.t else None
            self.t = nxt
            if self.accel != 1.0:
                # Compressed. SLEEP TO A FIXED SCHEDULE, not for a fixed
                # duration. Sleeping `interval` and then doing 2 s of work
                # makes each iteration interval+2 s long, and at 30x those two
                # seconds are a whole model minute: ticks drift to 4 model
                # minutes apart instead of 3. The model was trained on 3, so
                # that alone would move the numbers. Waking on a schedule
                # absorbs the work into the interval instead.
                interval = minutes * 60.0 / self.accel
                now = time.monotonic()
                if self._next_wake is None:
                    self._next_wake = now
                self._next_wake += interval
                gap = self._next_wake - now
                if gap > 0:
                    time.sleep(gap)
                elif -gap > interval:
                    # Work is outrunning the schedule; the compression is too
                    # aggressive to stay faithful. Say so rather than silently
                    # producing unevenly spaced ticks.
                    print(f"  ! cannot keep up at {self.accel:g}x "
                          f"(a tick needs more than {interval:.1f} s) — "
                          f"lower --accel")
                    self._next_wake = now
                return
            # sleep until the next tick boundary on the wall clock
            now = dt.datetime.now()
            target = now.replace(second=0, microsecond=0) + dt.timedelta(minutes=minutes)
            gap = (target - dt.datetime.now()).total_seconds()
            if gap > 0:
                time.sleep(gap)

    def running(self):
        if _stop:
            return False
        return self.t < self.end if self.replay else True


def feed_age_minutes(conn, now, replay: bool) -> float | None:
    """How old is the newest thing we hold?

    A stopped feed is the failure nothing else reveals: the forecaster keeps
    producing confident output from a world that has moved on, and the forecast
    itself looks completely normal.

    Live: measured from `changed_at` — how long since anything was written.
    Replay: measured from the newest OBSERVED EVENT at or before the replay
    clock, because `changed_at` is when we loaded the data, not when it
    happened, and would give a nonsense age of months."""
    if replay:
        mx = newest_event(conn, upto=now)
    else:
        with conn.cursor() as cur:
            cur.execute("SELECT max(changed_at) FROM main_table")
            mx = cur.fetchone()[0]
    if mx is None:
        return None
    ts = pd.Timestamp(mx)
    ts = ts.tz_localize(None) if ts.tz else ts
    return (pd.Timestamp(now) - ts).total_seconds() / 60.0


def _replay_is_honest(conn) -> str | None:
    """Replay masks the future through feed_chunk. If that table is not there,
    say so and stop — a silent fallback to the live table would produce a run
    that looks excellent and means nothing."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('feed_chunk')")
        if cur.fetchone()[0] is None:
            return "feed_chunk does not exist"
        cur.execute("SELECT count(*) FROM feed_chunk")
        if cur.fetchone()[0] == 0:
            return "feed_chunk is empty"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="corridor date to forecast "
                                   "(default: today, or the replay date)")
    ap.add_argument("--replay", nargs=3, metavar=("DATE", "FROM", "TO"),
                    help="replay a past window instead of running live")
    ap.add_argument("--accel", type=float, default=1.0,
                    help="live only: model minutes per real minute. 1 is real "
                         "time; 30 runs a day in 48 min. Ticks still step 3 "
                         "model-minutes, so results stay comparable.")
    ap.add_argument("--wall-clock", action="store_true",
                    help="force the model clock to today instead of the data's "
                         "day (almost never wanted; see the note in the header)")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="replay only: real seconds per simulated minute")
    ap.add_argument("--forecast-every", type=int, default=3)
    ap.add_argument("--no-resume", action="store_true",
                    help="start cold instead of restoring the saved memory")
    args = ap.parse_args()

    # ONE LIVE MODE, NOT TWO.
    # Anchoring the clock to the data's day is right whether the feed carries
    # today's events or a historical day being pushed in real time: when the
    # feed is live the offset comes out as zero and this is bit-for-bit the old
    # behaviour. --wall-clock exists only to force the old way back.
    follow = not (args.replay or args.wall_clock)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    if args.replay:
        d, f, t = args.replay
        day = dt.date.fromisoformat(d)
        hh = lambda s: dt.time(*map(int, s.split(":")))
        clock = Clock(dt.datetime.combine(day, hh(f)),
                      dt.datetime.combine(day, hh(t)), args.speed)
        date = args.date or d
    elif follow:
        # The loop wakes on the real clock; the corridor date comes from the
        # data, because the table holds a historical day, not today.
        clock = Clock(accel=args.accel)
        clock.t = dt.datetime.now()
        date = args.date                    # resolved from the DB below
    else:
        clock = Clock()
        clock.t = dt.datetime.now()
        date = args.date or dt.date.today().isoformat()

    conn = psycopg2.connect(**config.DB)

    # Follow mode takes the corridor date from the data. It MUST be resolved
    # before the banner and before the lock: the lock is scoped to the corridor
    # date, and locking on None is not a lock at all.
    if follow and date is None:
        seen = newest_event(conn)
        if seen is None:
            print()
            print("  NOTHING LOADED YET: the table holds no observed event.")
            print("  Start the loader first, or pass --date explicitly.")
            conn.close()
            return 4
        date = pd.Timestamp(seen).date().isoformat()

    mode = ("REPLAY " + args.replay[1] + "->" + args.replay[2] if args.replay
            else "LIVE")
    print("=" * 78)
    print(f"PRODUCTION LOOP   corridor date {date}   {mode}")
    print(f"  read every {config.FEED_MIN} min | tick every {config.TICK_MIN} min "
          f"| forecast every {args.forecast_every} min")
    print("=" * 78)

    if clock.replay:
        why = _replay_is_honest(conn)
        if why:
            print()
            print(f"  REFUSING TO REPLAY: {why}.")
            print("  Replay needs the reported-at times to hide the future.")
            print("  Seed it (sim/seed_db.py) or run live instead.")
            conn.close()
            return 2

    # One writer per corridor date. On its own connection, so the lock's life is
    # the process's life and not some transaction's.
    lock = RunLock(psycopg2.connect(**config.DB), date)
    if not lock.acquire():
        print()
        print(f"  REFUSING TO START: {lock.explain()}")
        conn.close()
        return 3

    # Replay reads through the reveal times; live reads the present as it is.
    as_of = (lambda t: t) if clock.replay else (lambda t: None)

    eng = StateEngine(conn, date, verbose=False)
    print(f"  engine ready | checkpoint {eng.ck_sha} | {eng.ds.num_services} services")

    if not args.no_resume:
        ts = eng.restore_state()
        print(f"  memory restored from {ts}" if ts else "  no checkpoint — cold start")

    # BOOT OVERLAYS MUST USE THE MODEL CLOCK, NOT THE WALL CLOCK.
    # In follow mode the wall clock is today and the data is a past day, so
    # building overlays against `clock.now()` searches a window with nothing in
    # it: freight, blocks and failures all come back empty. The loop repairs it
    # at the first feed window, but until then the model runs blind to
    # everything exogenous.
    boot_now = clock.now()
    offset = dt.timedelta(0)
    if follow:
        seen = newest_event(conn)
        if seen is not None:
            boot_now = pd.Timestamp(seen).to_pydatetime()
            # How far the data's day sits from today. Measured once; the clock
            # then advances on its own, in step with real time.
            offset = boot_now - clock.now()
            print(f"  clock offset {offset.total_seconds()/3600:+.1f} h "
                  f"| model time starts at {boot_now}"
                  + (f" | {clock.accel:g}x" if clock.accel != 1.0 else ""))

    n = eng.rebuild_journeys(as_of=as_of(boot_now))
    ov = eng.refresh_overlays(boot_now)
    # The read watermark is a real-world time: arrival_time is stamped by the
    # sender's clock, so it must be compared against the wall clock even when
    # the model is living in September.
    last_read = pd.Timestamp(clock.now())
    print(f"  journeys {n} | freight {ov['legs']} legs {ov['dwells']} dwells | "
          f"blocks {ov['blocks']} failures {ov['failures']} tsr {ov['tsr']}")
    print()
    print(f"  {'time':7}{'changed':>9}{'trains':>8}{'nodes':>7}{'fc':>6}"
          f"{'age':>7}   note")

    boot_wall = clock.now()          # real instant the loop started
    # Truncated to the minute. Ticks are whole minutes apart, so without this
    # the model clock inherits whatever seconds and microseconds the process
    # happened to start on, and every issued_at carries them —
    # 06:13:23.399146 instead of 06:13:00. That is not cosmetic: issued_at is
    # the key consumers join and compare on, and two runs of the same window
    # would then share no timestamps at all.
    boot_model = (boot_wall + offset).replace(second=0, microsecond=0)
    ticks = 1                        # model clock counts ticks, not seconds

    last_feed = boot_model      # model clock, matching the window gate
    last_fc = None
    total = 0
    last_seen = None        # follow mode: newest event we have observed
    last_moved = clock.now()   # ...and the REAL time it last advanced

    while clock.running():
        clock.advance(config.TICK_MIN)      # sleeps to the next real boundary
        wall = clock.now()                  # real time — drives the cadence
        now = wall                          # model time — replaced below
        note, n_changed = "", 0

        if follow:
            # THE CLOCK IS THE WALL CLOCK, SHIFTED ONTO THE DATA'S DAY.
            #
            # Taking the model clock straight from the newest event was wrong:
            # it FREEZES when the feed pauses. Ten quiet minutes and the model
            # would still place every train where it stood ten minutes ago,
            # while in reality they have all moved on. Position is what the
            # snapshot is built from, so that is not a small error.
            #
            # A fixed offset, measured once at boot, keeps both properties:
            # the clock advances with real time, so trains are positioned for
            # the moment we are actually in, and it stays on the data's day so
            # the journeys line up. Unreported stops are filled with estimates
            # exactly as they are in production — that path is not a fallback,
            # it is the normal one.
            # Exactly TICK_MIN of model time per tick. Deriving this from
            # elapsed real time instead would inherit every scheduling jitter,
            # multiplied by accel.
            now = boot_model + dt.timedelta(minutes=config.TICK_MIN * ticks)
            ticks += 1
            seen = newest_event(conn)
            if seen is not None and seen != last_seen:
                last_seen, last_moved = seen, wall

        # ── read the database ──────────────────────────────────────────────
        # Gate on the MODEL clock, always. It advances a fixed 3 minutes per
        # tick regardless of whether data arrived, so there is no deadlock to
        # avoid — an earlier version gated on the wall clock for that reason,
        # and under acceleration it then read only once per 5 REAL minutes
        # instead of once per 5 model minutes.
        if (now - last_feed).total_seconds() >= config.FEED_MIN * 60:
            # Look back further than the last read. Their arrival_time is the
            # SENDER's clock, so a row can commit here carrying a stamp older
            # than a read we have already done; without the overlap it would
            # never be returned. Re-reading is free — an unchanged journey
            # rebuilds to the identical journey.
            # The overlap is a REAL-time tolerance (sender clock skew,
            # transmission, out-of-order batches), so under compression it must
            # shrink with everything else. At 30x an unscaled 15 minutes covers
            # 450 model minutes and drags back every train in the table on
            # every window — which is both wrong and too slow to keep up.
            since = last_read - pd.Timedelta(
                minutes=config.FEED_LOOKBACK_MIN / clock.accel)
            changed = changed_trains(conn, since, until=now,
                                     by_event_time=clock.replay)
            n_changed = len(changed)
            # Both stamped on the clock they were measured against: arrival_time
            # is real-world, so the read watermark must be real-world too.
            # The watermark stays REAL-WORLD: arrival_time is stamped by the
            # sender's clock, so it can only be compared against wall time.
            last_read = pd.Timestamp(wall if follow else now)
            last_feed = now
            if changed:
                # Only the trains that reported need rebuilding; the rest are
                # kept as they stand. A full rebuild is 17.7 s against 0.9 s,
                # and produced identical journeys.
                n = eng.rebuild_journeys(changed, as_of=as_of(now))
            eng.refresh_overlays(now)
            note = "read+overlays"

        # ── model tick ─────────────────────────────────────────────────────
        r = eng.tick(now.hour * 60 + now.minute)
        eng.save_state(now)

        # ── forecast ───────────────────────────────────────────────────────
        nfc = 0
        if last_fc is None or (now - last_fc).total_seconds() >= args.forecast_every * 60:
            nfc = eng.forecast(now.hour * 60 + now.minute, now,
                               only_types=REAL_TYPES)
            last_fc = now
            total += nfc
            note = (note + " | FORECAST").strip(" |")

        # ── is the feed still alive? ───────────────────────────────────────
        # Follow mode needs a different question. The model clock IS the newest
        # event, so "how old is the newest event" is always zero and tells us
        # nothing. What matters is whether the loader is still delivering: how
        # long, in REAL minutes, since the newest event last moved forward.
        if follow:
            age = (wall - last_moved).total_seconds() / 60.0
        else:
            age = feed_age_minutes(conn, now, clock.replay)
        age_s = "--" if age is None else f"{age:.0f}m"
        if age is not None and age > STALE_CRIT_MIN:
            note += f"  *** FEED STALE {age:.0f} MIN ***"
        elif age is not None and age > STALE_WARN_MIN:
            note += f"  (feed {age:.0f} min old)"

        print(f"  {now.strftime('%H:%M'):7}{n_changed:>9}{len(eng.instances):>8}"
              f"{r['nodes']:>7}{nfc:>6}{age_s:>7}   {note}")

    print()
    print("=" * 78)
    print(f"  ticks {eng.ticks} | forecast rows written this run {total:,}")
    print("=" * 78)
    lock.release()
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
