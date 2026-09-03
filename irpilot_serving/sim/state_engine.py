# -*- coding: utf-8 -*-
"""
STEP 6 — the state engine.

The production loop, joined together:

    every 5 min   feed window revealed -> observation UPSERT -> rebuild journeys
    every 3 min   build snapshot -> model(update_memory=True) -> back up memory
    on schedule   clone memory -> 4h rollout -> restore -> write forecast rows

The three carried things live in RAM and are checkpointed to `model_state`:
the 3 GRU memories, _StationState, and prev_delay. Nothing else survives a tick.

Reads ONLY the live `observation` table, so it can never see the future.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import psycopg2
import psycopg2.extras
import torch

import config

config.bootstrap_gnn_path()

import cris_pipeline.build_dataset as BD                       # noqa: E402
from cris_pipeline.dataset_cris import (                        # noqa: E402
    CrisDataset, match_checkpoint_features,
)
from cris_pipeline.model_cris import (                          # noqa: E402
    CorridorNextEventGNN, load_weights, check_feature_dims,
)
from cris_pipeline.overlays import (                            # noqa: E402
    FreightOverlay, DisruptionOverlay,
)
from sim.build_journeys_db import (                             # noqa: E402
    read_running_from_db, read_main_from_db, load_side_tables,
)
from cris_pipeline.build_journeys import build_journey           # noqa: E402


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class StateEngine:
    """Long-running forecaster over the live database."""

    def __init__(self, conn, corridor_date: str, verbose: bool = True):
        self.conn = conn
        self.date = corridor_date
        self.verbose = verbose

        # ── model + topology (static, loaded once) ──────────────────────────
        ck = torch.load(config.CHECKPOINT, map_location="cpu",
                        weights_only=False)
        self.ck_sha = _sha(config.CHECKPOINT)
        match_checkpoint_features(ck["config"])

        topo = json.loads(Path(config.TOPOLOGY).read_text(encoding="utf-8"))
        BD._init_topology(topo)
        BD.HONEST_STANDING_DELAY = bool(ck.get("honest_standing_delay", False))

        # CrisDataset needs a day-file directory to bootstrap its feature
        # widths and service index; it is NOT used as a data source here.
        self.ds = CrisDataset(str(config.DATASET_DIR),
                              str(config.TOPOLOGY), lazy=True, verbose=False)
        self.model = CorridorNextEventGNN(**ck["config"])
        load_weights(self.model, ck["model"])
        check_feature_dims(ck["config"], self.ds)
        self.model.eval()

        # ROSTER GUARD: service_idx is positional. If the roster the model was
        # trained on differs from the one loaded now, every train silently maps
        # to the wrong embedding row. Assert instead of failing quietly.
        n_ck = int(ck["config"]["num_services"])
        if self.ds.num_services != n_ck:
            raise SystemExit(
                f"ROSTER MISMATCH: checkpoint expects {n_ck} services, "
                f"topology has {self.ds.num_services}. Embeddings would be wrong.")

        # Overlays are REBUILT every feed window by refresh_overlays(), not
        # loaded once here. Building them at boot and never again meant a block
        # declared after startup stayed invisible all day — measured, severity
        # 1.0 vs 0.0 for the same section and minute, decided purely by when
        # the process happened to start.
        self.pairs = {(int(s["from_id"]), int(s["to_id"]))
                      for s in BD._SECTIONS.values()}
        self.freight = None
        self.disrupt = None
        self.overlay_as_of = None

        # day_idx must match how build_dataset numbered days, so absolute
        # minutes line up with the topology/overlay lookups.
        self.day_idx = 0
        d = dt.date.fromisoformat(corridor_date)
        self.day_meta = {"day_of_week": d.weekday()}

        # ── the three carried things ────────────────────────────────────────
        self.model.reset_memory()
        self.st_state = BD._StationState()
        self.prev_delay: dict[str, float] = {}

        # (train_number, train_date) -> instance. Kept between windows so only
        # the trains that changed are rebuilt.
        self._jcache: dict[tuple, dict] = {}
        self._side = None            # coaches + platforms, read once

        self.instances: list[dict] = []
        self.by_iid: dict[str, dict] = {}
        self.routes: dict[str, list] = {}
        self.ticks = 0
        self.forecasts_written = 0

    # ── overlays: rebuild from the LIVE tables, at the current clock ────────
    def refresh_overlays(self, as_of) -> dict:
        """Rebuild both overlays as they stand at `as_of`.

        Must be called on every feed window. Freight moves and blocks open and
        close continuously; an overlay built at 09:00 describes 09:00 and
        nothing else. `as_of` also bounds every OPEN event — a freight that has
        departed and not arrived, a block with no clearance — so those run to
        now rather than being dropped or guessed at.
        """
        import os
        ts = pd.Timestamp(as_of)
        if os.getenv("IRPILOT_LEGACY_OVERLAYS") == "1":
            # A/B SWITCH, for measuring what the causal fixes cost.
            # These are the ORIGINAL archive overlays: they read the whole CSV,
            # so past T0 they answer with what actually happened — future
            # breakdowns, real block clearances, observed future freight.
            # Never use this in production; it exists to price the fix.
            self.freight = FreightOverlay.load()
            self.disrupt = DisruptionOverlay.load(self.pairs)
        else:
            from live.overlays_live import (
                LiveFreightOverlay, LiveDisruptionOverlay,
            )
            self.freight = LiveFreightOverlay.from_db(self.conn, ts)
            self.disrupt = LiveDisruptionOverlay.from_db(self.conn, ts, self.pairs)
        self.overlay_as_of = ts
        return {"legs": self.freight.n_legs, "dwells": self.freight.n_dwells,
                "blocks": self.disrupt.n_blocks,
                "failures": self.disrupt.n_failures,
                "tsr": self.disrupt.n_tsr}

    def causal_overlays(self, t0_ts, lookback_hours: int = 72):
        """Overlays for ONE forecast, valid at `t0_ts`.

        The tick overlays describe now. A rollout walks 80 ticks INTO THE
        FUTURE and asks the overlay what is on the track at each one, so it
        needs a different object: observed up to T0, PLANNED beyond it.

        What changes past T0:
          * freight standing or running at T0 is carried forward on its typical
            duration, and goods BOOKED to arrive inside the horizon are counted
            on their booked path — a scheduled goods train still takes capacity
          * blocks run to their PERMITTED end, not the real clearance
          * asset failures that had not started by T0 are excluded entirely,
            because a failure is unplanned and cannot be foreseen
        """
        from live.overlays_causal import (
            CausalFreightOverlay, CausalDisruptionOverlay,
        )
        # Two forecasts at the same instant get the same overlays; rebuilding
        # them cost 0.48 s each.
        if getattr(self, "_causal_at", None) == t0_ts:
            return self._causal
        lo = t0_ts - pd.Timedelta(hours=lookback_hours)
        hi = t0_ts + pd.Timedelta(minutes=240)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT r.coatrainid, r.sqncnumb, r.station,
                       r.arvltime, r.dprttime, s.schdarvl, s.schddprt
                  FROM {gr} r
                  LEFT JOIN {gs} s
                    ON s.coatrainid = r.coatrainid AND s.sqncnumb = r.sqncnumb
                 WHERE r.station = ANY(%s)
                   AND (r.arvltime BETWEEN %s AND %s
                     OR r.dprttime BETWEEN %s AND %s
                     OR s.schdarvl BETWEEN %s AND %s)
            """.format(gr=config.NAMES["goods_running"],
                       gs=config.NAMES["goods_schedule"]),
                (list(BD._CODE_TO_ID), lo, t0_ts, lo, t0_ts, t0_ts, hi))
            gdf = pd.DataFrame(cur.fetchall(),
                               columns=["coatrainid", "sqncnumb", "station",
                                        "arvltime", "dprttime",
                                        "schdarvl", "schddprt"])
            cur.execute("""SELECT block_section, permitted_start_time,
                                  permitted_end_time, actual_clear_time
                             FROM {mt}
                            WHERE permitted_start_time BETWEEN %s AND %s""".format(
                                mt=config.NAMES["maintenance"]),
                        (lo, hi))
            blocks = cur.fetchall()
            cur.execute("""SELECT block_section, event_start_date,
                                  event_end_date, affected_trains
                             FROM {af}
                            WHERE division = 'JBP'
                              AND event_start_date BETWEEN %s AND %s""".format(
                                af=config.NAMES["asset_failure"]),
                        (lo, t0_ts))     # never past T0 — failures are unplanned
            failures = cur.fetchall()
            cur.execute("""SELECT block_section, passenger_train_speed,
                                  {tf} AS from_datetime, {tt} AS to_datetime
                             FROM {tsr}""".format(
                                 tf=config.NAMES["tsr_from"],
                                 tt=config.NAMES["tsr_to"],
                                 tsr=config.NAMES["tsr"]))
            tsrs = cur.fetchall()

        self._causal = (CausalFreightOverlay.at_t0(gdf, t0_ts),
                        CausalDisruptionOverlay.at_t0(blocks, failures, tsrs,
                                                      t0_ts, self.pairs))
        self._causal_at = t0_ts
        return self._causal

    # ── journeys: rebuild from the LIVE observation table ───────────────────
    def rebuild_journeys(self, changed=None, as_of=None) -> int:
        """Rebuild instances for this corridor date.

        `changed` is a set of (train_number, train_date). Pass it and ONLY those
        trains are re-read and rebuilt; every other journey is kept as it stands
        in memory. Pass None (or call it the first time) and everything is built.

        WHY THIS MATTERS. Rebuilding all 99 trains means reading the whole
        624,180-row table and running build_journey ~3,000 times: 22.5 seconds.
        Doing that on every 5-minute window made the feed windows 84% of a
        replay's runtime, against 16% for the forecasts themselves. A window
        typically touches ~75 trains, usually one stop each.

        The stages after this are unchanged — journey -> instance (every NULL
        filled with schedule + carried delay) -> snapshot -> tensors. Only the
        set of journeys being rebuilt is narrowed.

        `as_of` is for replaying a past window against a database that already
        holds the whole month. It restricts the read to values CRIS had actually
        reported by then. Production leaves it None: the live table is the
        present, and there is nothing to mask.
        """
        # The coach and platform tables are two CSVs that do not change during a
        # run; re-reading them on every window was pure waste.
        if self._side is None:
            self._side = load_side_tables(self.conn, verbose=self.verbose)
        coaches, platforms = self._side

        full = changed is None or not self._jcache
        sel = None if full else changed

        if as_of is None:
            # LIVE: read the merged table. The trigger has already done the
            # schedule-to-observation join on write, so redoing it on every
            # read is work for nothing. It is also more precise: the old join
            # filtered train_number and train_date as two separate lists, so
            # asking for 69 train-days dragged back 134 — the cross product —
            # and rebuilt 65 journeys that had not changed. Verified: all 2,604
            # journeys come out byte-identical either way.
            df = read_main_from_db(self.conn, sel)
        else:
            # REPLAY: main_table carries no reveal times, so it cannot answer
            # "what was known at 09:00" — it holds the whole month. Only
            # feed_chunk can, so a replay must keep going through the masked
            # schedule/observation path.
            df = read_running_from_db(self.conn, sel, as_of=as_of)

        for (tn, date), g in df.groupby(["train_no", "train_date"], sort=True):
            rec = build_journey(g, coaches, platforms)
            if rec is None or rec.get("_rejected") or rec["corridor_date"] != self.date:
                # It may have been valid before and not now — drop the stale one
                # rather than leaving a journey the data no longer supports.
                self._jcache.pop((tn, date), None)
                continue
            inst = BD._build_instance(rec, self.day_idx)
            if inst:
                inst["detention"] = {}
                self._jcache[(tn, date)] = inst
            else:
                self._jcache.pop((tn, date), None)

        instances = sorted(self._jcache.values(), key=lambda i: i["instance_id"])
        self.instances = instances
        self.by_iid = {i["instance_id"]: i for i in instances}
        self.routes = {i["instance_id"]: i["route"] for i in instances}
        return len(instances)

    # ── one 3-minute tick ──────────────────────────────────────────────────
    def tick(self, minute_of_day: int) -> dict:
        if self.freight is None or self.disrupt is None:
            raise RuntimeError(
                "refresh_overlays() has not been called. Ticking without it "
                "would run the model on no freight and no disruptions at all.")
        t = self.day_idx * 1440 + minute_of_day
        snap = BD.build_snapshot(self.day_idx, t, self.instances, self.st_state,
                                 self.day_meta, self.prev_delay,
                                 self.freight, self.disrupt, self.date)
        n_nodes = len(snap["train_nodes"])
        if not n_nodes:
            self.ticks += 1
            return {"nodes": 0, "ran": False}

        data = self.ds.materialise(snap, self.by_iid, self.routes)
        if data is None or not data["train"].x.numel():
            self.ticks += 1
            return {"nodes": n_nodes, "ran": False}

        with torch.no_grad():
            self.model(data, update_memory=True)     # <- memory advances here
        self.ticks += 1
        return {"nodes": n_nodes, "ran": True}

    # ── memory checkpoint ──────────────────────────────────────────────────
    def save_state(self, tick_ts: dt.datetime) -> int:
        def blob(t):
            b = io.BytesIO(); torch.save(t, b); return psycopg2.Binary(b.getvalue())
        st = {"cong": dict(self.st_state.cong),
              "crit": {str(k): bool(v) for k, v in self.st_state.crit.items()},
              "steps": {str(k): int(v) for k, v in self.st_state.steps.items()}}
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_state (tick_ts, station_memory, service_memory,
                                         section_memory, station_state,
                                         prev_delay, checkpoint_sha)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (tick_ts) DO UPDATE SET
                    station_memory=EXCLUDED.station_memory,
                    service_memory=EXCLUDED.service_memory,
                    section_memory=EXCLUDED.section_memory,
                    station_state =EXCLUDED.station_state,
                    prev_delay    =EXCLUDED.prev_delay,
                    checkpoint_sha=EXCLUDED.checkpoint_sha
            """, (tick_ts, blob(self.model.station_memory),
                  blob(self.model.service_memory), blob(self.model.section_memory),
                  psycopg2.extras.Json(st),
                  psycopg2.extras.Json({k: float(v) for k, v in self.prev_delay.items()}),
                  self.ck_sha))
        self.conn.commit()
        return 1

    def restore_state(self) -> dt.datetime | None:
        """Load the newest checkpoint. Refuses a checkpoint from other weights."""
        with self.conn.cursor() as cur:
            cur.execute("""SELECT tick_ts, station_memory, service_memory,
                                  section_memory, station_state, prev_delay,
                                  checkpoint_sha
                             FROM model_state ORDER BY tick_ts DESC LIMIT 1""")
            r = cur.fetchone()
        if not r:
            return None
        ts, sm, vm, cm, st, pd_, sha = r
        if sha != self.ck_sha:
            raise SystemExit(f"STATE/WEIGHTS MISMATCH: saved state belongs to "
                             f"checkpoint {sha}, running {self.ck_sha}")
        ld = lambda b: torch.load(io.BytesIO(bytes(b)), weights_only=False)
        self.model.station_memory = ld(sm)
        self.model.service_memory = ld(vm)
        self.model.section_memory = ld(cm)
        self.st_state = BD._StationState()
        for k, v in (st.get("cong") or {}).items():
            self.st_state.cong[int(k)] = float(v)
        for k, v in (st.get("crit") or {}).items():
            self.st_state.crit[int(k)] = bool(v)
        for k, v in (st.get("steps") or {}).items():
            self.st_state.steps[int(k)] = int(v)
        self.prev_delay = {k: float(v) for k, v in (pd_ or {}).items()}
        return ts

    # ── forecast: fork the memory, roll forward, restore ───────────────────
    def forecast(self, minute_of_day: int, issued_at: dt.datetime,
                 only_types=None) -> int:
        from cris_pipeline.rollout_cris import Rollout, _state_copy

        t0 = self.day_idx * 1440 + minute_of_day
        saved = (self.model.station_memory.clone(),
                 self.model.service_memory.clone(),
                 self.model.section_memory.clone())
        # CAUSAL overlays, not the tick ones. The rollout queries the overlay at
        # minutes AFTER t0; the tick overlays answer those from observation,
        # which at t0 is the future. See causal_overlays().
        import os
        if os.getenv("IRPILOT_LEGACY_OVERLAYS") == "1":
            c_freight, c_disrupt = self.freight, self.disrupt   # archive, leaky
        else:
            c_freight, c_disrupt = self.causal_overlays(pd.Timestamp(issued_at))
        ro = Rollout(self.model, self.ds, self.day_idx, self.date, self.day_meta,
                     c_freight, c_disrupt, self.by_iid, None, "absolute",
                     standing_dep="booked")
        # The rollout WRITES its predictions into this deep copy's stops. Its
        # returned `records` are for SCORING only and are gated on ground truth
        # existing (`has_actual`), which in live operation the future does not
        # have — so read the forecast off the mutated journeys instead.
        world = copy.deepcopy(self.instances)
        ro.run(t0, world, _state_copy(self.st_state), dict(self.prev_delay))
        (self.model.station_memory, self.model.service_memory,
         self.model.section_memory) = saved            # <- memory untouched

        horizon_end = t0 + 240                          # 4 h, as the rollout uses
        recs = []
        for inst in world:
            iid = inst["instance_id"]
            truth = self.by_iid.get(iid)
            if truth is None:
                continue
            if only_types and truth.get("train_type") not in only_types:
                continue
            pos = BD._find_position(truth, t0)
            if pos is None:
                continue                                # not running at T0
            k0 = pos["stop_idx"]
            for k in range(k0 + 1, len(inst["stops"])):
                s = inst["stops"][k]
                arr = s.get("actual_arr_abs")
                sa = s.get("sched_arr_abs")
                if arr is None or sa is None:
                    continue
                if arr > horizon_end:                   # beyond the 4 h horizon
                    break
                recs.append((issued_at, iid, int(s["station"]), k - k0,
                             float(arr) - t0, float(arr) - float(sa),
                             None, None, "v13b_ep18"))
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, """
                INSERT INTO forecast (issued_at, instance_id, station_id, hop,
                                      lead_min, pred_delay_min, lo80, hi80,
                                      model_version)
                VALUES %s
                ON CONFLICT (issued_at, instance_id, station_id) DO NOTHING
            """, recs)
        self.conn.commit()
        self.forecasts_written += len(recs)
        return len(recs)
