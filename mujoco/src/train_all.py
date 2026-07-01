"""
train_all.py
============
Convenience script that trains all five algorithms sequentially.

Usage
-----
  # Full training run
  python src/train_all.py

  # Quick smoke test (100 episodes, 200 steps)
  python src/train_all.py --episodes 100 --max-steps 200

  # Skip specific algorithms
  python src/train_all.py --skip ppo ppo_lag

  # Custom device
  python src/train_all.py --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_SRC = Path(__file__).parent
_MJ_ROOT = _SRC.parent
sys.path.insert(0, str(_MJ_ROOT))
sys.path.insert(0, str(_SRC))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _train_one(algo: str, args: argparse.Namespace) -> None:
    """Import and run the training function for a single algorithm."""
    results_dir = str(Path(args.results_dir))

    if algo == "cddp":
        from src.train_cddp import train, _parse as _p
        a = argparse.Namespace(
            episodes=args.episodes,
            pretrain_episodes=max(1, args.episodes // 5),
            max_steps=args.max_steps,
            device=args.device,
            results_dir=results_dir,
        )
        train(a)

    elif algo == "ddpg":
        from src.train_ddpg import train, _parse as _p
        a = argparse.Namespace(
            episodes=args.episodes, max_steps=args.max_steps,
            device=args.device, results_dir=results_dir,
        )
        train(a)

    elif algo == "ppo":
        from src.train_ppo import train
        a = argparse.Namespace(
            episodes=args.episodes, max_steps=args.max_steps,
            device=args.device, results_dir=results_dir,
        )
        train(a)

    elif algo == "strict_cbf":
        from src.train_strict_cbf import train
        a = argparse.Namespace(
            episodes=args.episodes, max_steps=args.max_steps,
            device=args.device, results_dir=results_dir,
        )
        train(a)

    elif algo == "ppo_lag":
        from src.train_ppo_lagrangian import train
        a = argparse.Namespace(
            episodes=args.episodes, max_steps=args.max_steps,
            device=args.device, results_dir=results_dir,
        )
        train(a)

    else:
        log.warning("Unknown algorithm: %s", algo)


ALL_ALGOS = ["cddp", "ddpg", "ppo", "strict_cbf", "ppo_lag"]


def main() -> None:
    p = argparse.ArgumentParser(description="Train all CDDP-MuJoCo algorithms")
    p.add_argument("--episodes",    type=int,    default=5_000)
    p.add_argument("--max-steps",   type=int,    default=500)
    p.add_argument("--device",      type=str,    default="cpu")
    p.add_argument("--skip",        nargs="*",   default=[],
                   help="Algorithm names to skip")
    p.add_argument("--only",        nargs="*",   default=None,
                   help="Only train these algorithms")
    p.add_argument("--results-dir", type=str,
                   default=str(Path(__file__).parent.parent / "results"))
    args = p.parse_args()

    algos = ALL_ALGOS if args.only is None else args.only
    algos = [a for a in algos if a not in (args.skip or [])]

    log.info("Training plan: %s", algos)
    log.info("Episodes=%d  max_steps=%d  device=%s", args.episodes, args.max_steps, args.device)

    timings: dict[str, float] = {}
    for algo in algos:
        log.info("\n" + "="*60)
        log.info("TRAINING: %s", algo.upper())
        log.info("="*60)
        t0 = time.time()
        try:
            _train_one(algo, args)
        except Exception as e:
            log.error("Training %s failed: %s", algo, e)
            import traceback; traceback.print_exc()
        timings[algo] = time.time() - t0
        log.info("Finished %s in %.1f s", algo, timings[algo])

    log.info("\n=== Training Summary ===")
    for algo, elapsed in timings.items():
        log.info("  %-15s %.1f s", algo, elapsed)
    log.info("All done. Checkpoints in: %s", args.results_dir)


if __name__ == "__main__":
    main()
