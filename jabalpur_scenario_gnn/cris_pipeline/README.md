# CRIS Pipeline — official JBP-STA corridor data

Rebuilds the training data from the **official CRIS September-2025 extract**
(`data_standardized_timestamp_all/`), replacing the self-scraped railradar data
that every model up to `nextevent_v2.pt` was trained on.

```
python -m cris_pipeline.build_topology     # infra tables -> topology
python -m cris_pipeline.build_journeys     # running CSV  -> per-train-day journeys
python -m cris_pipeline.build_dataset      # journeys     -> day snapshots   [TODO]
```

Outputs land in `data_cris/`.

---

## Why the old feature set had to be re-derived

Measured over 56,470 train nodes / 175,500 station nodes of `dataset_v12` —
the data `nextevent_v2` was actually trained on — by running the live
`dataset.py` pipeline and taking per-dim statistics:

| Channel | Finding |
|---|---|
| Section (9 dims) | `current_multiplier` **constant 1.0**; `is_disrupted` **constant 0**; `from_is_jct`/`to_is_jct` **constant 0**; `trains_in_section` **99.4% zero**. Only 4 of 9 dims alive. |
| Station (18 dims) | Whole congestion channel **~95% zero** (occupancy 95.5%, incoming_pressure 95.6%, delayed_density 96.3%, critical_flag 98%). |
| Train (41 dims) | `eta_second_jct` **std 0.0064**, `eta_nearest_jct` std 0.048 — saturated at the 180-min clamp. Conflict/pressure block 76–94% zero. |

**This is the real cause of the diagnosed "platform blocking unmodeled"**
(congested-vs-free station cosine similarity 0.973). It was never a model
failure: 117 stations were fed ~30–50 passenger trains a day and **zero
freight**, so the platforms genuinely were always free in the input. The model
became a delay-persistence machine because that was the only live signal —
which is why it beats persistence by only 0.7–3.0 min.

---

## Bugs found and fixed

1. **`from_is_jct` / `to_is_jct` always false** — `build_dataset.py` writes them
   onto the snapshot *edge dict*, but `dataset.py::_section_features` read them
   from `topo_sections_by_key`, built from the topology, which never carried
   those keys. Every section on every snapshot got the default `False`.
   Fixed in `dataset.py` by deriving the flags from station identity
   (`junction_ids`); verified — now 8.2% non-zero on the division graph.

2. **Duplicate aggregate sections → 4 phantom junctions.** CRIS lists both the
   subdivided legs through a block post *and* an aggregate section spanning it.
   Exact distance arithmetic proves the duplication:

   ```
   BUU-GNWA  6.24 + GNWA-UDR 6.42 = 12.66  ==  "BUU-UDR"  12.66
   KTES-MDRR 3.93 + MDRR-NWR  7.85 = 11.78  ==  "KTES-NWR" 11.79
   NWR-SNRR  7.64 + SNRR-SBD  6.96 = 14.60  ==  "NWR-SBD"  14.60
   ```

   The old topology loaded all 24, which double-counted track capacity and gave
   BUU/UDR/NWR/SBD a phantom third edge — promoting them to "degree>=3
   junctions". CRIS `MACJUNCTION` says only **STA, KTE, KTES, JBP** are
   junctions. This matters because `junction_ids` drives the entire
   conflict-edge builder in `scenario_events.py`.
   Fixed: 21 sections, 4 junctions.

3. **Midnight rollover corrupted the time base.** The old journeys stored
   minutes-of-day, which goes backwards when a train crosses midnight inside
   the corridor. Measured: **9.2% of corridor train-days**. (This was partly
   mis-attributed to "scraper artifacts"; `_find_position` "tolerating" it meant
   those trains were positioned wrongly in the simulator.)
   Fixed: journeys now store minutes since midnight of `journey_date`, allowed
   to exceed 1440. Verified **0% non-monotonic**, down from 9.2%.

---

## Data decisions

**Nominal-schedule services are actors, not targets.** Median corridor arrival
delay by type:

```
VNDB  -2 | DRNT 17 | MEX 18 | SUF 20 | MEMU 25      <- real timetable
AMTB  59 | TRST 97 | TRSF 154 | TOD 260 | PEXP 985 | PPSP 1014
```

`TOD/PEXP/PPSP/TRST/TRSF/INGL/AMTB` run to nominal paths, so their "delay" is
meaningless — including them takes corridor mean delay from 32 to 80 min and
p95 from 120 to 384. But they *physically occupy the corridor*, so dropping
them outright would repeat the old mistake of an under-populated network. They
are emitted with `is_target_eligible=False`: present in the graph, excluded
from supervision. Freight is treated the same way (median "delay" 324 min).

**Block posts stay as pass-through nodes.** GNWA / MDRR / SNRR are
`MACHALTFLAG='Y'`, class D, zero platform lines, and report actual times on
**0.0%** of their stop-rows. They subdivide sections and carry occupancy, but
can never be supervised.

**`is_single_track` is dropped for this corridor.** All 21 sections are
`MANNUMBLINES=2`. Keeping it would reintroduce exactly the constant-dim problem
being fixed here.

**Junction-ETA clamp 180 -> 90 min.** The 180-min clamp saturated on the
division graph. The corridor is 238 km with 4 junctions, so 90 min covers the
realistic approach window without pinning to 1.0.

**`train_number` normalisation.** It arrives as a float with leading zeros
stripped. The old pipeline kept `'1053'` while CRIS elsewhere uses `'01053'`;
without `zfill(5)` roughly half the coach/platform join is silently lost.

---

## Built so far

`build_topology.py` — 22 stations (19 halts + 3 block posts), 21 sections,
4 junctions, real `MANNUMBLINES` / `MANMAXSPEED` / `MANINTRDIST`, real platform
counts and `MANNUMBTRAINALLWD` holding capacity, PSR time-loss per section.

`build_journeys.py` — 2,606 journey files (2,223 target-eligible + 383
actor-only), 47,818 corridor stops at 80.7% actual-time coverage.

Validated:

```
scheduled non-monotonic : 0.00%      (was 9.2%)
delays (eligible)       : median 18 | mean 32 | p95 120 | 31.1% spikes >30min
supervised chain length : median 18 stops | 93% of journeys >=8
                          32,470 supervised stops | 30,264 hop transitions
inherited delay         : 26.6% enter the corridor already >30 min late
                          29.0% arrive with a worsening upstream trend
```

For comparison, the old `dataset_v12` capped at **4 valid hops, with 60% of
train-samples having zero valid targets**. Half the calendar days, ~4.5x the
usable supervision per journey — and the inherited-delay context was previously
unobservable entirely.

---

## Remaining

- `build_dataset.py` — snapshot generation with freight occupancy overlay,
  disruption overlay (maintenance blocks / asset failures / TSR), and the
  re-derived feature spec.
- **Acceptance gate**: re-run the feature audit on the new dataset — no dim may
  ship constant or >98% zero.
- Retrain and compare against the `nextevent_v2` simulator baseline:
  `8.0 / 12.0 / 23.8 min` at 0-30m / 30-60m / 1-2h, spike recall 25/30/13%.
