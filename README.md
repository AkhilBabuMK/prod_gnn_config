# Corridor Delay Forecaster — JBP to STA

Predicts train arrival delays up to 4 hours ahead, every 3 minutes, from the
data already in your database.

It **reads** six of your tables and never writes to them. It creates four
objects of its own plus two read views.

---

## 1. Install

CPU only. There is no GPU anywhere in this.

    python 3.11 or newer
    pip install torch pandas numpy psycopg2-binary

## 2. Configure

Open `env.sh`, fill in the five database lines, then:

    source env.sh

Nothing else needs editing. Every environment-specific value is in that file.

If one of your tables is named differently from what we expect, uncomment the
matching `IRPILOT_TBL_...` line instead of changing code.

## 3. Set up and check

    cd irpilot_serving
    python setup_prod.py

This creates our objects and then runs 16 checks against your database. It is
safe to re-run at any time and it changes nothing of yours.

**Read the output.** Every check corresponds to a real failure we hit during
integration, and most of them were silent — the service kept running and the
forecasts quietly got worse. If anything says FAIL, fix it before step 4.

## 4. Run

    python run_prod.py

Reads what is new every 5 minutes, steps the model every 3, forecasts 4 hours
ahead. Runs on the real clock and does not stop. Safe to restart at any moment:
it checkpoints its memory every 3 minutes and restores on boot.

## 5. Read the forecasts

    SELECT train_number, start_date, station_code, block_section,
           pred_delay_min, lead_min, scheduled_arrival, predicted_arrival, actual_arrival
      FROM forecast_latest
     ORDER BY predicted_arrival;

`forecast_latest` is the most recent issue. `forecast_read` is the full history
in the same shape.

---

## Rehearsing first (recommended)

Before pointing at the live table, you can drive the whole thing from a
stand-in feed built in your own database:

    python demo/feed_loader.py --from 06:00 --hours 8 --speed 30
    bash demo/run_forecaster.sh --accel 30

That compresses a whole day into about 16 minutes. The model still steps 3
minutes at a time, so the result is comparable to running it for real.

---

## What to watch when it is running

One line per tick:

    time     changed  trains  nodes    fc    age   note
    09:08          0      99     14    99     0m   FORECAST
    09:11        114      99     14    96     0m   read+overlays | FORECAST

- **time** — the moment the model believes it is. Should advance 3 minutes
  every line, always.
- **trains** — journeys held, around 99. Zero all day means the corridor date
  is wrong.
- **changed** — trains that reported since the last read, normally 70-120.
  Zero on every line means the feed is not arriving.
- **age** — minutes since the feed last delivered. Over 12 warns; over 30 is
  critical and means the feed has stopped while the model keeps producing
  confident output.

At the first boot also confirm:

- `301 services` — the roster guard passed. It refuses to start on a mismatch,
  because the service index is positional and every train would otherwise take
  another train's embedding, silently.
- `checkpoint f941fb83a91105dc` — a different value means a different model
  than the one our figures came from.
- **No line saying `! coaches table unavailable` or `! PF_INFO table
  unavailable`.** If you see one, stop. The service will keep running with
  empty lookups; measured cost is 14% of trains losing their coach count, with
  no error anywhere.

---

## Scoring it on your own data

After a day has run:

    python tests/score_forecasts.py

On our data this gives 9.15 min mean error overall and 77% recall on delays
over 30 minutes. Yours will differ — different trains, different days — but the
shape should hold: roughly 5 minutes of error at a half-hour horizon, rising to
about 15 minutes at four hours.

---

## Two things worth confirming on your side

**Is the stop key unique?** We need one row per
`(train_number, train_date, serial_number)`. Two rows for the same stop would
build a journey from both and nothing would raise. `setup_prod.py` tests this
and refuses to pass if it finds duplicates.

**What format is `todatetime` on the TSR table?** It is text, not a timestamp,
so we cast it. That is correct while the text is ISO-like. If it is
`dd/mm/yyyy` the cast throws above the 13th and, worse, silently swaps day and
month below it — a restriction ending 05/09 reads as 9 May, already expired, so
it is dropped and the model believes the track is clear. A guard catches this
at startup by looking for windows that end before they start. If it fires, set
`IRPILOT_TBL_TSR_TO` as shown in `env.sh`.

---

## One thing we already handle

Your `train_number` has no leading zeros — `1053` where our reference tables use
`01053`. Left alone, every coach and platform lookup misses and returns nothing,
with no error: 372 trains lose their coach count and 501 platform assignments
disappear. We pad on read, so there is nothing for you to do. It is mentioned
only so it is not a surprise later.

---

## What gets created in your database

    forecast          one row per (issue time, train, station)
    model_state       the carried model memory, one row per 3-minute tick
    infra_topology    the corridor graph, so any past forecast can be matched
                      to the graph that produced it
    station_ref       station id -> code and name
    forecast_read     the forecast in readable terms
    forecast_latest   just the most recent issue

## What gets read, and never written

    gnn_input_table              goods_train_running
    maintenance_block            goods_train_schedule
    asset_failure                temporary_speed_restriction
