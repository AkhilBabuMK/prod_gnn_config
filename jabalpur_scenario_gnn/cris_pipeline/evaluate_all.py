# -*- coding: utf-8 -*-
"""
cris_pipeline/evaluate_all.py
=============================
Runs the whole evaluation suite against one checkpoint, in the order the
results should be READ — cheapest and least trustworthy first, so the honest
numbers land last and are the ones you remember.

  1. audit_features    are the inputs alive, and is any of them too good?
  2. audit_causality   does any input read the future? (with the pending-
                       departure perturbation that caught three leaks)
  3. diagnose_model    shapes, gradients, message passing, distinguishability
  4. eval_cris         TEACHER-FORCED. A training diagnostic, NOT a forecast.
                       It cannot detect a leak, because every input it hands
                       the model is ground truth.
  5. ablate_edges      which parts of the graph earn their place
  6. rollout_cris      THE PRODUCTION NUMBER. Autoregressive, nothing observed
                       after T0, compared against the old system's
                       8.0 / 12.0 / 23.8 min by lead time.

Quote the rollout STRICT block. Everything above it is diagnosis.

    python -m cris_pipeline.evaluate_all
    python -m cris_pipeline.evaluate_all --quick     (skip the slow audits)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cris_pipeline.config import OUT_DIR                       # noqa: E402

STEPS = [
    ("FEATURE AUDIT", ["-m", "cris_pipeline.audit_features", "--max-days", "6"], False),
    ("CAUSALITY AUDIT", ["-m", "cris_pipeline.audit_causality", "--days", "2",
                         "--ticks", "10"], True),
    ("MODEL DIAGNOSTICS", ["-m", "cris_pipeline.diagnose_model", "--max-days", "2"], False),
    ("TEACHER-FORCED EVAL (diagnostic only)",
     ["-m", "cris_pipeline.eval_cris"], False),
    ("EDGE ABLATION", ["-m", "cris_pipeline.ablate_edges"], False),
    ("ROLLOUT — THE PRODUCTION METRIC",
     ["-m", "cris_pipeline.rollout_cris", "--every", "120",
      "--check-state", "--show", "6"], False),
]

NEEDS_CKPT = {"MODEL DIAGNOSTICS", "TEACHER-FORCED EVAL (diagnostic only)",
              "EDGE ABLATION", "ROLLOUT — THE PRODUCTION METRIC"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUT_DIR / "corridor_nextevent.pt"))
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow audits (2 and 5)")
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"no checkpoint at {args.checkpoint}")

    skip = {"CAUSALITY AUDIT", "EDGE ABLATION"} if args.quick else set()
    failures = []

    for name, cmd, slow in STEPS:
        if name in skip:
            print(f"\n{'#' * 78}\n# SKIPPED (--quick): {name}\n{'#' * 78}")
            continue
        full = [sys.executable] + cmd
        if name in NEEDS_CKPT:
            full += ["--checkpoint", args.checkpoint]
        print(f"\n\n{'#' * 78}")
        print(f"# {name}")
        print(f"# {' '.join(cmd)}")
        print(f"{'#' * 78}", flush=True)
        t0 = time.time()
        r = subprocess.run(full, cwd=str(Path(__file__).resolve().parent.parent))
        dt = time.time() - t0
        if r.returncode != 0:
            failures.append(f"{name} (exit {r.returncode})")
            print(f"\n  !! {name} FAILED (exit {r.returncode}) after {dt:.0f}s")
        else:
            print(f"\n  {name} ok ({dt:.0f}s)")

    print(f"\n\n{'=' * 78}")
    if failures:
        print("SUITE FINISHED WITH FAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SUITE PASSED — quote the rollout STRICT block, not the "
          "teacher-forced eval.")


if __name__ == "__main__":
    main()
