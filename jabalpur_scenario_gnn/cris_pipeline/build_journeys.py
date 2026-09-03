# -*- coding: utf-8 -*-
"""
cris_pipeline/build_journeys.py
===============================
Converts the official CRIS running data into per-train-per-date journey JSON
files for the JBP-STA corridor.

Design decisions (each backed by a measurement — see cris_pipeline/README.md):

  1. TIME BASE — minutes since midnight of `corridor_date`, allowed to exceed
     1440. Two separate problems are fixed here:

       (a) The previous pipeline stored minutes-of-day, which goes backwards
           when a train crosses midnight inside the corridor. Measured: 9.2% of
           corridor train-days do exactly that.

       (b) `train_date` is the date the train left its ORIGIN, which for these
           long-distance services is usually a different day from the corridor
           traverse — 68% of journeys reach the corridor after midnight
           relative to their train_date. Snapshots must be grouped by the day
           the train is actually ON the corridor, so `corridor_date` (the date
           of the first corridor stop) is the grouping key and the time origin.
           `journey_date` is retained as metadata.

  2. DELAYS ARE DERIVED from (actual - scheduled) timestamps. There is no
     trustworthy pre-computed delay column, and ~12% of corridor stops have no
     actual time at all.

  3. NOMINAL-SCHEDULE SERVICES ARE KEPT AS ACTORS BUT NOT AS TARGETS.
     TOD/PEXP/PPSP/TRST/TRSF/INGL/AMTB have nominal paths rather than published
     timetables (median "delay" 260-1014 min). They still physically occupy the
     corridor, so removing them entirely would repeat the old mistake of an
     under-populated network. They are emitted with is_target_eligible=False.

  4. INHERITED DELAY. Corridor trains run far beyond the corridor (full
     journeys span 60 divisions). The out-of-corridor prefix is collapsed into
     entry features so the model can see the delay a train arrives WITH,
     which was previously unobservable.

Usage:
    python -m cris_pipeline.build_journeys
    python -m cris_pipeline.build_journeys --limit 50     # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import (
    RUNNING_CSV, COACHES_CSV, PF_INFO_CSV,
    CORRIDOR, CORRIDOR_ORDER, BLOCK_POSTS, NOMINAL_SCHEDULE_TYPES,
    JOURNEYS_DIR, TOPOLOGY_PATH,
)
from .build_topology import (
    TRAIN_TYPE_MAP, CLASS_PROPERTIES, CLASS_TO_GROUP,
)

CODE_TO_ID = {c: i for i, c in enumerate(CORRIDOR_ORDER)}

# Largest believable delay change across one corridor leg (see the guard in
# build_journey for the measurement this is chosen from).
MAX_LEG_DELTA_MIN = 180.0

TIME_COLS = [
    "scheduled_arrival_time", "actual_arrival_time",
    "scheduled_departure_time", "actual_departure_time",
]


# ── Loading ───────────────────────────────────────────────────────────────────

def load_running() -> pd.DataFrame:
    df = pd.read_csv(RUNNING_CSV, low_memory=False)

    # train_number arrives as a float with leading zeros stripped. The previous
    # pipeline kept the stripped form ('1053'), CRIS elsewhere uses '01053';
    # normalising is what makes the coach/platform joins land.
    df = df[df.train_number.notna()].copy()
    df["train_no"] = df.train_number.astype(int).astype(str).str.zfill(5)

    for c in TIME_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # Fill corridor gaps from the reference export BEFORE deriving delays, so a
    # recovered schedule produces a real delay rather than a dropped stop.
    # CRIS values are never overwritten; see reference_merge for the checks.
    from .reference_merge import ReferenceIndex
    _ref = ReferenceIndex.load()
    df = _ref.fill(df)
    _ref.report()

    df["arr_delay"] = ((df.actual_arrival_time - df.scheduled_arrival_time)
                       .dt.total_seconds() / 60.0)
    df["dep_delay"] = ((df.actual_departure_time - df.scheduled_departure_time)
                       .dt.total_seconds() / 60.0)
    return df.sort_values(["train_no", "train_date", "serial_number"])


def load_side_tables() -> tuple[dict, dict]:
    """Returns (coaches_by_train, platform_by_train_station)."""
    coaches: dict[str, int] = {}
    try:
        cdf = pd.read_csv(COACHES_CSV, low_memory=False)
        cdf["tn"] = cdf.TRAIN_NUMBER.astype(str).str.strip().str.zfill(5)
        coaches = dict(zip(cdf.tn, cdf.NO_OF_COACHES.fillna(0).astype(int)))
    except Exception as exc:                                   # pragma: no cover
        print(f"  ! coaches table unavailable ({exc})")

    platforms: dict[tuple[str, str], str] = {}
    try:
        pdf = pd.read_csv(PF_INFO_CSV, low_memory=False)
        pdf["tn"] = pdf.TRAIN_NUMBER.astype(str).str.strip().str.zfill(5)
        for r in pdf.itertuples():
            platforms[(r.tn, str(r.STATION_CODE).strip())] = str(r.PF_NUMBER)
    except Exception as exc:                                   # pragma: no cover
        print(f"  ! PF_INFO table unavailable ({exc})")

    return coaches, platforms


# ── Per-journey conversion ────────────────────────────────────────────────────

def _rel_minutes(ts, day_midnight) -> float | None:
    """Minutes from midnight of journey_date. May exceed 1440 (next-day arrival)."""
    if pd.isna(ts):
        return None
    return round((ts - day_midnight).total_seconds() / 60.0, 2)


def _entry_features(prefix: pd.DataFrame) -> dict:
    """Collapse the out-of-corridor prefix into inherited-delay context.

    `prefix` is every stop the train made BEFORE reaching the corridor, in
    order. Everything here is knowable at corridor entry — no leakage.
    """
    obs = prefix[prefix.arr_delay.notna()]
    if obs.empty:
        return {
            "upstream_stops_observed":  0,
            "entry_delay_min":          0.0,
            "upstream_delay_trend_min": 0.0,
            "upstream_max_delay_min":   0.0,
            "km_before_corridor":       float(prefix.distance_from_source_km.max())
                                        if len(prefix) else 0.0,
            "has_upstream":             False,
            "upstream_volatility":      0.0,
            "upstream_recovered_min":   0.0,
            "upstream_rate5":           0.0,
        }

    delays = obs.arr_delay.to_numpy(dtype=float)
    entry  = float(delays[-1])
    # Trend over the last 3 observed upstream stops: is it building or easing?
    trend  = float(delays[-1] - delays[max(0, len(delays) - 4)])

    # UPSTREAM SHAPE, not just level and trend.
    #
    # A train reaches this corridor having already run a median of 685 km over
    # 76 observed stops, and we were collapsing all of it into entry delay plus
    # a 3-stop trend. Measured on the training split with a gradient-boosted
    # tree grouped by train, predicting the MAX RISE inside our corridor:
    #
    #     entry + trend only        R^2  -0.024   (worse than guessing)
    #     + the shape below         R^2  +0.108
    #
    # and permutation importance puts VOLATILITY far ahead of everything else:
    #     upstream_volatility +0.553, stops observed +0.259, entry +0.221,
    #     recovered +0.161, km +0.075, rate5 +0.050
    #
    # So the question that matters is not "how late is it" but "how ERRATIC has
    # it been". A train that has been swinging for 700 km keeps swinging here;
    # a steady one, however late, stays steady. We had no version of that.
    #
    # This is a ONE-STEP input improvement, which is where the leverage is:
    # rollout error is flat across horizon given sound inputs (2.89 min at hop
    # 1, 1.92 at hop 8), and compounding is near-optimal random walk, so a
    # minute cut here is worth about sqrt(8) ~ 2.8 min at the 4 h horizon.
    #
    # Every value is observed strictly before the train reaches the corridor.
    steps = delays[1:] - delays[:-1] if len(delays) > 1 else delays[:0]
    if len(steps):
        mean_s = float(steps.mean())
        vol = float((sum((s - mean_s) ** 2 for s in steps) / len(steps)) ** 0.5)
        recovered = float(-sum(s for s in steps if s < 0))
        rate5 = float(steps[-5:].mean())
    else:
        vol = recovered = rate5 = 0.0

    return {
        "upstream_stops_observed":  int(len(obs)),
        "entry_delay_min":          round(entry, 2),
        "upstream_delay_trend_min": round(trend, 2),
        "upstream_max_delay_min":   round(float(delays.max()), 2),
        "km_before_corridor":       round(float(prefix.distance_from_source_km.max()), 2),
        "has_upstream":             True,
        "upstream_volatility":      round(vol, 2),
        "upstream_recovered_min":   round(recovered, 2),
        "upstream_rate5":           round(rate5, 2),
    }


def build_journey(g: pd.DataFrame, coaches: dict, platforms: dict) -> dict | None:
    """Convert one (train_no, train_date) group into a journey record."""
    train_no   = g.train_no.iloc[0]
    date_str   = str(g.train_date.iloc[0])
    train_type = str(g.train_type.iloc[0])

    in_corr = g.station_code.isin(CORRIDOR).to_numpy()
    if not in_corr.any():
        return None

    first, last = int(np.argmax(in_corr)), int(len(in_corr) - 1 - np.argmax(in_corr[::-1]))
    corr   = g.iloc[first:last + 1]
    corr   = corr[corr.station_code.isin(CORRIDOR)]
    prefix = g.iloc[:first]

    if len(corr) < 2:
        return None

    # Time origin = midnight of the day the train actually reaches the corridor,
    # not the day it left its origin. Prefer the first observed timestamp;
    # fall back to schedule when the train never reported.
    anchor = corr.actual_arrival_time.dropna()
    if anchor.empty:
        anchor = corr.scheduled_arrival_time.dropna()
    if anchor.empty:
        anchor = corr.scheduled_departure_time.dropna()
    if anchor.empty:
        return None
    corridor_date = anchor.iloc[0].normalize()
    day_midnight  = corridor_date

    stops = []
    for r in corr.itertuples():
        code = str(r.station_code)
        sa = _rel_minutes(r.scheduled_arrival_time,   day_midnight)
        sd = _rel_minutes(r.scheduled_departure_time, day_midnight)
        aa = _rel_minutes(r.actual_arrival_time,      day_midnight)
        ad = _rel_minutes(r.actual_departure_time,    day_midnight)
        if sa is None and sd is None:
            continue
        if sa is None:
            sa = sd
        if sd is None:
            sd = sa

        stops.append({
            "station_code":     code,
            "station_id":       CODE_TO_ID[code],
            "sched_arr_min":    sa,
            "sched_dep_min":    sd,
            "actual_arr_min":   aa,
            "actual_dep_min":   ad,
            "arr_delay_min":    (None if pd.isna(r.arr_delay)
                                 else round(float(r.arr_delay), 2)),
            "dep_delay_min":    (None if pd.isna(r.dep_delay)
                                 else round(float(r.dep_delay), 2)),
            # A target is set at EVERY station the train moves through, halt or
            # pass-through — we forecast the whole movement, not just the stops.
            # Only block posts are excluded: they report an actual time on 0.0%
            # of rows, so they can never be supervised.
            "is_division_stop": code not in BLOCK_POSTS,
            "has_actual":       aa is not None,
            # Halt vs pass-through is a FEATURE, never a target filter.
            # Measured: only 23.7% of corridor stop-rows are booked halts. A
            # passing train occupies the running line, not a platform, and has
            # no dwell — the old pipeline conflated the two.
            "is_booked_halt":   bool(getattr(r, "wtt_stop_flag", 0)),
            "is_ptt_halt":      bool(getattr(r, "ptt_stoppage_flag", 0)),
            # Slack the TIMETABLE itself carries at this stop: pathing time
            # plus engineering allowance, both in the working timetable and so
            # known before the train runs -- no leakage.
            #
            # This is where recovery physically comes from, and the model has
            # never been shown it. Measured on the corridor: mean delay change
            # at a stop is +1.24 min with no allowance and -17.47 min with 10+
            # minutes, monotonic across the range, and the gap holds inside
            # every incoming-delay band (-3.94 on time, -6.59 for 5-30 late,
            # -7.99 for 30+ late), so it is not "late trains get allowance".
            #
            # Not redundant with scheduled run time, which the model already
            # has: corr = 0.584. Scheduled time INCLUDES the padding, so
            # without this column the model cannot separate a genuinely long
            # section from a padded one.
            #
            # Corridor coverage: 16.5% of stops, mean 5.2 min where present,
            # concentrated at JBP (73.8% of stops, 6.64 min), STA (73.3%,
            # 4.44) and KTE (56.4%) -- the junctions where recovery happens.
            "sched_allowance_min": round(
                (float(getattr(r, "traffic_allowance_seconds", 0) or 0.0)
                 + float(getattr(r, "engineering_allowance_seconds", 0) or 0.0))
                / 60.0, 2),
            "platform":         platforms.get((train_no, code)),
            "distance_km":      (None if pd.isna(r.distance_from_source_km)
                                 else round(float(r.distance_from_source_km), 2)),
        })

    if len(stops) < 2:
        return None

    # ── Physical-plausibility guard ───────────────────────────────────────────
    # Corridor legs are scheduled at 2.5-9.4 minutes (longest section 14.98 km).
    # A delay that jumps by more than MAX_LEG_DELTA_MIN between two consecutive
    # reported stops is a mis-dated record, not an operational event — e.g.
    # train 11705 on 2025-09-28 "departs JBP 1 min late" then "arrives SHR 811
    # late", which is 14 hours to cover 44 km.
    #
    # The measured distribution has a clean break here: the count of h1 deltas
    # above 120 / 180 / 400 min is 174 / 150 / 139, i.e. a discrete cluster of
    # bad records far above a smooth genuine tail. Those records account for
    # only 0.18% of train-days but 38% of total absolute error, and their
    # timestamps also place the train in the graph at the wrong hour, so they
    # are dropped whole rather than target-masked.
    # Check the combined ARRIVAL-or-DEPARTURE observed sequence: for 11705 the
    # arrival chain alone is self-consistent (811 -> 786 -> 789 ...) and the
    # contradiction only shows up against JBP's observed DEPARTURE of +1 min.
    obs = []
    for s in stops:
        d = s["arr_delay_min"]
        if d is None:
            d = s["dep_delay_min"]
        if d is not None:
            obs.append((s["station_code"], float(d)))
    for (ca, da), (cb, db) in zip(obs, obs[1:]):
        if abs(db - da) > MAX_LEG_DELTA_MIN:
            return {"_rejected": True, "train_no": train_no,
                    "corridor_date": corridor_date.strftime("%Y-%m-%d"),
                    "reason": (f"implausible leg delta {ca}->{cb}: "
                               f"{da:.0f} -> {db:.0f} min")}

    # Direction from corridor index movement (STA=0 ... JBP=21).
    direction = "DN" if stops[-1]["station_id"] > stops[0]["station_id"] else "UP"

    cls = TRAIN_TYPE_MAP.get(train_type, "mail_express")
    props = CLASS_PROPERTIES[cls]

    return {
        "train_no":            train_no,
        # Day the corridor traverse happens — the snapshot grouping key.
        "corridor_date":       corridor_date.strftime("%Y-%m-%d"),
        # Day the train left its origin — metadata only.
        "journey_date":        date_str,
        "train_type":          train_type,
        "train_sub_type":      str(g.train_sub_type.iloc[0]),
        "train_class":         cls,
        "train_group":         CLASS_TO_GROUP[cls],
        "priority":            props["priority"],
        "speed_factor":        props["speed_factor"],
        "source":              str(g.train_source_station.iloc[0]),
        "destination":         str(g.train_destination_station.iloc[0]),
        "direction":           direction,
        "n_coaches":           coaches.get(train_no, 0),
        # Nominal-schedule services stay in the graph as occupancy actors but
        # are excluded from supervision.
        "is_target_eligible":  train_type not in NOMINAL_SCHEDULE_TYPES,
        "entry":               _entry_features(prefix),
        "data_source":         "CRIS",
        "stops":               stops,
    }


# ── Driver ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(JOURNEYS_DIR))
    ap.add_argument("--limit", type=int, default=0,
                    help="only process N train-days (smoke test)")
    ap.add_argument("--keep-existing", action="store_true",
                    help="do NOT wipe the output directory first. Almost "
                         "never what you want — see the guard below.")
    ap.add_argument("--clean", action="store_true",
                    help="deprecated; wiping is now the default")
    args = ap.parse_args()

    out_dir = pd.io.common.stringify_path(args.out)
    from pathlib import Path
    out = Path(out_dir)

    # Wipe by default. This used to require --clean, and forgetting it was
    # silently catastrophic rather than noisy: a rebuild does not overwrite,
    # it writes `<date>#2.json` beside the stale `<date>.json` (see the
    # collision branch below). _load_journeys then reads BOTH, so every train
    # is duplicated in every snapshot, and because sorted order puts '#'
    # (0x23) before '.' (0x2e) the STALE copy is processed last and wins.
    #
    # That is exactly what happened on 2025-07-30: a config change making
    # TOD/TRST/AMTB target-eligible was reverted by July-28 files, and the
    # resulting dataset had 6,183 duplicate train instances per day and zero
    # TOD targets. Both audit_features and audit_causality passed on it —
    # they check feature sanity and causality, not data integrity.
    if out.exists() and not args.keep_existing:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading CRIS running data ...")
    df = load_running()
    coaches, platforms = load_side_tables()
    print(f"  {len(df):,} rows | {df.train_no.nunique()} services")

    # Restrict to train-days that touch the corridor at all.
    touch = df[df.station_code.isin(CORRIDOR)]
    keys  = set(zip(touch.train_no, touch.train_date))
    print(f"  {len(keys):,} corridor train-days")

    written  = 0
    skipped  = 0
    rejected: list[str] = []
    stats    = defaultdict(int)
    services: dict[str, dict] = {}

    for (tn, date), g in df.groupby(["train_no", "train_date"], sort=True):
        if (tn, date) not in keys:
            continue
        if args.limit and written >= args.limit:
            break

        rec = build_journey(g, coaches, platforms)
        if rec is None:
            skipped += 1
            continue
        if rec.get("_rejected"):
            stats["rejected_implausible"] += 1
            rejected.append(f"{rec['train_no']} {rec['corridor_date']}: "
                            f"{rec['reason']}")
            continue

        stats["target_eligible" if rec["is_target_eligible"]
              else "actor_only"] += 1
        stats["with_upstream"] += int(rec["entry"]["has_upstream"])
        stats["stops"] += len(rec["stops"])
        stats["stops_with_actual"] += sum(s["has_actual"] for s in rec["stops"])

        services.setdefault(tn, {
            "train_no":     tn,
            "train_class":  rec["train_class"],
            "train_group":  rec["train_group"],
            "priority":     rec["priority"],
            "speed_factor": rec["speed_factor"],
            "train_type":   rec["train_type"],
            "n_coaches":    rec["n_coaches"],
            "is_target_eligible": rec["is_target_eligible"],
        })

        # Filed under corridor_date so the dataset builder groups each journey
        # onto the day it is actually on the corridor.
        # Filed under corridor_date. Consecutive origin-dates can map to the
        # same corridor_date (a service that traverses either side of midnight),
        # so suffix rather than overwrite — the dataset builder groups on the
        # `corridor_date` field, not the filename, so both are picked up.
        d = out / tn
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{rec['corridor_date']}.json"
        if path.exists():
            stats["corridor_date_collision"] += 1
            k = 2
            while (d / f"{rec['corridor_date']}#{k}.json").exists():
                k += 1
            path = d / f"{rec['corridor_date']}#{k}.json"
        path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        written += 1

    # Register the service list on the topology so embeddings have a stable index.
    if TOPOLOGY_PATH.exists() and not args.limit:
        topo = json.loads(TOPOLOGY_PATH.read_text(encoding="utf-8"))
        topo["trains"] = [services[k] for k in sorted(services)]
        topo["meta"]["n_trains"] = len(services)
        TOPOLOGY_PATH.write_text(json.dumps(topo, indent=1), encoding="utf-8")
        print(f"  registered {len(services)} services on the topology")

    print(f"\nWrote {written:,} journey files to {out}")
    print(f"  skipped (<2 corridor stops) : {skipped:,}")
    print(f"  target-eligible journeys    : {stats['target_eligible']:,}")
    print(f"  actor-only (nominal sched)  : {stats['actor_only']:,}")
    print(f"  with upstream context       : {stats['with_upstream']:,}")
    print(f"  corridor stops              : {stats['stops']:,} "
          f"({100 * stats['stops_with_actual'] / max(stats['stops'], 1):.1f}% "
          f"with an actual time)")
    if stats["rejected_implausible"]:
        print(f"  rejected (implausible leg) : {stats['rejected_implausible']:,}")
        for r in rejected:
            print(f"      {r}")
    if stats["corridor_date_collision"]:
        print(f"  ! corridor_date collisions  : "
              f"{stats['corridor_date_collision']:,} (kept side by side as "
              f"<date>#N.json — both are loaded as separate instances)")


if __name__ == "__main__":
    main()
