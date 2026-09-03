# -*- coding: utf-8 -*-
"""
cris_pipeline/audit_splits.py
=============================
IS THE TRAIN/VAL/TEST SPLIT ABLE TO ANSWER THE QUESTIONS WE ASK OF IT?

A split can be methodologically correct and still be the wrong instrument.
Ours is chronological, which is the only honest choice for forecasting -- a
shuffled split would leak tomorrow into today. That part is right and must not
change. What this module measures is whether 29 days, cut 19/5/5, can support
the conclusions we keep drawing from it.

Reports:

  1  CALENDAR COVERAGE   which weekdays each split contains. Weekly services
                         exist on this corridor, so a missing weekday means a
                         whole service is structurally absent from evaluation.
  2  DIFFICULTY          spike/recovery rate and delay spread per split. If
                         test is harder than train, every headline number is
                         conservative -- worth knowing before comparing to
                         anything historical.
  3  SERVICE EXPOSURE    how many training days each service appears on, and
                         how much TEST supervision rests on services the model
                         barely saw or never saw. `nn.Embedding(num_services)`
                         gives every service its own row; a service absent from
                         the training split keeps its RANDOM INIT.
  4  EMBEDDING REALITY   (with --checkpoint) the learned norm of those rows,
                         grouped by exposure. This turns "the embedding is
                         untrained" from an argument into a measurement.

    python -m cris_pipeline.audit_splits
    python -m cris_pipeline.audit_splits --checkpoint data_cris/corridor_v3_ep21.pt
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cris_pipeline import build_dataset as BD                       # noqa: E402
from cris_pipeline.config import JOURNEYS_DIR, TOPOLOGY_PATH        # noqa: E402

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SPLITS = ["train", "validation", "test"]


def _bucket(n: int) -> str:
    if n == 0:
        return "0 NEVER"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    if n <= 14:
        return "6-14"
    return "15-19"


BUCKETS = ["0 NEVER", "1-2", "3-5", "6-14", "15-19"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journeys", default=str(JOURNEYS_DIR))
    ap.add_argument("--checkpoint", default=None,
                    help="also report learned service-embedding norms")
    args = ap.parse_args()

    by_date = BD._load_journeys(Path(args.journeys))
    dates = BD._drop_partial_days(by_date, sorted(by_date))
    split = BD._assign_splits(dates)

    days = collections.defaultdict(list)
    for d in dates:
        days[split[d]].append(d)

    # ── 1  calendar coverage ─────────────────────────────────────────────────
    print("=" * 78)
    print("1  CALENDAR COVERAGE")
    print("=" * 78)
    print(f"  {len(dates)} usable days: {dates[0]} .. {dates[-1]}\n")
    for k in SPLITS:
        ds = days[k]
        wd = collections.Counter(dt.date.fromisoformat(d).strftime("%a") for d in ds)
        missing = [w for w in WEEKDAYS if not wd.get(w)]
        print(f"  {k:11} n={len(ds):2d}  {ds[0]} .. {ds[-1]}")
        print(f"  {'':11} " + "  ".join(f"{w}:{wd.get(w, 0)}" for w in WEEKDAYS)
              + (f"   MISSING {'/'.join(missing)}" if missing else ""))

    # ── 2  difficulty ────────────────────────────────────────────────────────
    stats = collections.defaultdict(lambda: [0, 0, 0, 0, []])
    exposure = collections.defaultdict(collections.Counter)
    tgt = collections.defaultdict(collections.Counter)
    ttype: dict[str, str] = {}

    for d in dates:
        k = split[d]
        for rec in by_date[d]:
            tno = str(rec["train_no"])
            ttype[tno] = rec.get("train_type", "?")
            exposure[tno][k] += 1
            if not rec.get("is_target_eligible"):
                continue
            tgt[tno][k] += sum(1 for s in rec["stops"] if s.get("has_actual"))
            ds = [s["arr_delay_min"] for s in rec["stops"]
                  if s.get("arr_delay_min") is not None]
            if len(ds) < 2:
                continue
            deltas = [float(b) - float(a) for a, b in zip(ds, ds[1:])]
            a = stats[k]
            a[0] += 1
            a[1] += len(deltas)
            a[2] += sum(1 for x in deltas if x > 30)
            a[3] += sum(1 for x in deltas if x < -30)
            a[4].extend(deltas)

    print("\n" + "=" * 78)
    print("2  DIFFICULTY — is test the same problem as train?")
    print("=" * 78)
    print(f"  {'split':11} {'journeys':>9} {'targets':>8} {'spike%':>8} "
          f"{'recov%':>8} {'mean d':>8} {'sd':>7}")
    for k in SPLITS:
        j, n, s, r, dl = stats[k]
        if not n:
            continue
        print(f"  {k:11} {j:>9,} {n:>8,} {100*s/n:>7.2f}% {100*r/n:>7.2f}% "
              f"{st.mean(dl):>8.2f} {st.pstdev(dl):>7.2f}")

    # ── 3  service exposure ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("3  SERVICE EXPOSURE — how much TEST supervision rests on unseen trains")
    print("=" * 78)
    n_svc = collections.Counter()
    n_rows = collections.Counter()
    for tno in exposure:
        b = _bucket(exposure[tno]["train"])
        n_svc[b] += 1
        n_rows[b] += tgt[tno]["test"]
    total_rows = sum(n_rows.values())
    print(f"  {'training days':>16} {'services':>9} {'test stops':>12} {'share':>8}")
    for b in BUCKETS:
        if n_svc[b]:
            print(f"  {b:>16} {n_svc[b]:>9} {n_rows[b]:>12,} "
                  f"{100*n_rows[b]/max(total_rows,1):>7.2f}%")
    print(f"  {'TOTAL':>16} {sum(n_svc.values()):>9} {total_rows:>12,}")

    never = [t for t in exposure
             if exposure[t]["train"] == 0 and tgt[t]["test"] > 0]
    if never:
        c = collections.Counter(ttype[t] for t in never)
        print(f"\n  services with TEST targets but ZERO training exposure: "
              f"{len(never)}")
        print(f"     by type: {dict(c)}")
        print(f"     examples: {sorted(never)[:12]}")

    # ── 4  embedding reality ─────────────────────────────────────────────────
    if args.checkpoint:
        import torch
        topo = json.loads(Path(TOPOLOGY_PATH).read_text(encoding="utf-8"))
        services = sorted({t["train_no"] for t in topo.get("trains", [])})
        s2i = {t: i for i, t in enumerate(services)}
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = ck.get("model") or ck.get("state_dict") or ck
        W = sd["service_emb.weight"]

        print("\n" + "=" * 78)
        print("4  EMBEDDING REALITY — were those rows ever trained?")
        print("=" * 78)
        g = collections.defaultdict(list)
        for t, i in s2i.items():
            g[_bucket(exposure.get(str(t), collections.Counter())["train"])].append(i)
        print(f"  {'training days':>16} {'services':>9} {'mean row norm':>15} "
              f"{'row sd':>9}")
        for b in BUCKETS:
            idx = g.get(b)
            if idx:
                R = W[idx]
                print(f"  {b:>16} {len(idx):>9} {R.norm(dim=1).mean():>15.4f} "
                      f"{R.std():>9.4f}")
        print("\n  A '0 NEVER' norm far below the trained buckets means those rows\n"
              "  still hold their random init: the model has no learned identity\n"
              "  for those services. Init is SMALL, so they contribute a near-null\n"
              "  vector -- 'no information', not adversarial noise.")

    print()


if __name__ == "__main__":
    main()
