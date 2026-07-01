"""
evaluate_timer_sweep.py
=======================
Thin wrapper around run_baseline_timer_comparison that accepts:
  --timers T1 T2 T3 ...   (space-separated, positional-style via nargs='+')
  --n-eval N              (total eval episodes per timer; split as N seeds x 1 episode)
  --output-prefix PREFIX  (default: timer_sweep_summary)
  --models M1,M2,...      (default: all five)

All other heavy lifting is delegated to run_baseline_timer_comparison.
"""
from __future__ import annotations

import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SRC_DIR, "..")))

import src.run_baseline_timer_comparison as sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Timer sweep evaluation wrapper.")
    parser.add_argument("--timers", nargs="+", type=int, required=True,
                        help="Space-separated timer values (env steps).")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="Number of evaluation episodes per (model, timer) pair.")
    parser.add_argument("--models", default=",".join(sweep.MODELS),
                        help="Comma-separated model names.")
    parser.add_argument("--output-prefix", default="timer_sweep_summary",
                        help="Filename prefix for output CSV/JSON/plot.")
    parser.add_argument("--base-seed", type=int, default=None)
    args = parser.parse_args()

    # Map --n-eval to seeds x episodes (N seeds, 1 episode each → N unique contexts).
    seeds = args.n_eval
    episodes_per_seed = 1

    timers_str = ",".join(str(t) for t in sorted(set(args.timers)))

    # Rebuild argv so run_baseline_timer_comparison.main() parses correctly.
    sys.argv = [
        "run_baseline_timer_comparison.py",
        "--models", args.models,
        "--timers", timers_str,
        "--seeds", str(seeds),
        "--episodes-per-seed", str(episodes_per_seed),
        "--output-prefix", args.output_prefix,
    ]
    if args.base_seed is not None:
        sys.argv += ["--base-seed", str(args.base_seed)]

    sweep.main()


if __name__ == "__main__":
    main()
