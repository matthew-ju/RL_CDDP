"""
evaluate_timer_sweep.py
=======================
Evaluation suite for all CDDP-MuJoCo frameworks.

Loads trained weights for each algorithm and runs a sweep over initial timer
lengths. Records raw arrays and computes the five key analysis curves:

  1. Success Rate vs. Timer
  2. Total Violations vs. Timer
  3. Excess Violations vs. Timer
  4. P(Excess > 0) vs. Timer
  5. Pareto Trajectory Frontier (Violations vs. Goal Distance)

Output files (written to results/)
-----------------------------------
  timer_sweep_raw.json          — per-episode raw data for every (algo, timer)
  timer_sweep_summary.csv       — aggregated mean stats per (algo, timer)
  pareto_frontier.json          — Pareto trajectory data (violations vs dist)
  plots/                        — PNG plots for all five curves

Usage
-----
  # Evaluate all algorithms
  python src/evaluate_timer_sweep.py

  # Evaluate specific algorithms
  python src/evaluate_timer_sweep.py --algos cddp ddpg

  # Custom timer range
  python src/evaluate_timer_sweep.py --timers 100 200 500 1000 2000

  # Quick smoke test (3 episodes per cell)
  python src/evaluate_timer_sweep.py --n-eval 3 --timers 200 500
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).parent
_MJ_ROOT = _SRC.parent
sys.path.insert(0, str(_MJ_ROOT))
sys.path.insert(0, str(_SRC))

from src.crowd_env import CrowdNavEnv, STATE_DIM
from src.networks import Actor, DeonticCritic
from src.cbf_layer import project_action_cbf

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGO_NAMES = ["cddp", "ddpg", "ppo", "strict_cbf", "ppo_lag"]

# Default timer sweep (steps)
DEFAULT_TIMERS = [100, 200, 300, 500, 750, 1_000, 1_500, 2_000]

# How many evaluation episodes per (algo, timer) cell
DEFAULT_N_EVAL = 20

# Excess violation threshold: violations above this count as "excess"
EXCESS_THRESHOLD = 0   # any violation at all counts


# ---------------------------------------------------------------------------
# Agent loaders
# ---------------------------------------------------------------------------

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")


def _load_actor(path: str, state_dim: int, action_dim: int = 2,
                act_limit: float = 1.0) -> Any:
    from src.networks import Actor
    actor = Actor(state_dim, action_dim, act_limit).to(DEVICE)
    actor.load_state_dict(torch.load(path, map_location=DEVICE))
    actor.eval()
    return actor


def _load_ppo_actor(path: str, state_dim: int, action_dim: int = 2) -> Any:
    from src.networks import PPOActor
    actor = PPOActor(state_dim, action_dim).to(DEVICE)
    actor.load_state_dict(torch.load(path, map_location=DEVICE))
    actor.eval()
    return actor


def _load_deon(path: str, state_dim: int) -> Any:
    from src.networks import DeonticCritic
    net = DeonticCritic(state_dim).to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Per-algorithm action selection
# ---------------------------------------------------------------------------

P_REF      = 0.2
TAU_CDDP   = 0.3
ALPHA_PRIME = 10.0
GAMMA_CBF  = 1.5


def _obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)


@torch.no_grad()
def act_cddp(obs: np.ndarray, actor, z_deon,
             obs_centers: torch.Tensor, obs_radii: torch.Tensor) -> np.ndarray:
    s = _obs_to_tensor(obs)
    u_nom = actor(s)
    logits = z_deon(s)
    p_cat = torch.sigmoid(logits)
    stakes = torch.clamp(p_cat.squeeze(-1) / P_REF, 0.0, 1.0)
    u_safe = project_action_cbf(
        s[:, :2], u_nom, obs_centers, obs_radii,
        stakes=stakes, gamma=GAMMA_CBF, tau=TAU_CDDP, alpha_prime=ALPHA_PRIME,
    )
    return u_safe.squeeze(0).cpu().numpy()


@torch.no_grad()
def act_ddpg(obs: np.ndarray, actor) -> np.ndarray:
    s = _obs_to_tensor(obs)
    return actor(s).squeeze(0).cpu().numpy()


@torch.no_grad()
def act_ppo(obs: np.ndarray, actor) -> np.ndarray:
    s = _obs_to_tensor(obs)
    a, _ = actor.act(s, deterministic=True)
    return a.squeeze(0).cpu().numpy()


@torch.no_grad()
def act_strict_cbf(obs: np.ndarray, actor,
                   obs_centers: torch.Tensor, obs_radii: torch.Tensor) -> np.ndarray:
    s = _obs_to_tensor(obs)
    u_nom = actor(s)
    stakes = torch.zeros(1, device=DEVICE)
    u_safe = project_action_cbf(
        s[:, :2], u_nom, obs_centers, obs_radii,
        stakes=stakes, gamma=GAMMA_CBF,
    )
    return u_safe.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------

def run_episode(
    env: CrowdNavEnv,
    action_fn,
    seed: int | None = None,
) -> dict:
    obs, _ = env.reset(seed=seed)
    total_reward   = 0.0
    total_violations = 0
    step = 0
    traj_positions: list[tuple[float, float]] = []
    stakes_history: list[float] = []
    done = False

    while not done:
        action = action_fn(obs)
        if isinstance(action, tuple):
            action, stk = action
            stakes_history.append(float(stk))

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward     += reward
        total_violations += info["n_violations"]
        step += 1

        pos = obs[:2]
        traj_positions.append((float(pos[0]), float(pos[1])))

    goal_pos = env.goal_pos
    final_dist = float(np.linalg.norm(obs[:2] - goal_pos))
    min_dist   = float(min(
        np.linalg.norm(np.array(p) - goal_pos) for p in traj_positions
    )) if traj_positions else float("inf")

    return {
        "reward":           total_reward,
        "n_violations":     total_violations,
        "n_steps":          step,
        "goal_reached":     bool(info.get("goal_reached", False)),
        "timeout":          bool(info.get("timeout", False)),
        "final_dist":       final_dist,
        "min_dist":         min_dist,
        "trajectory":       traj_positions,
        "avg_stakes":       float(np.mean(stakes_history)) if stakes_history else 0.0,
        "max_stakes":       float(np.max(stakes_history))  if stakes_history else 0.0,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    algos: list[str],
    timers: list[int],
    n_eval: int,
    results_dir: Path,
    ckpt_base: Path,
) -> dict:
    """Run the full timer sweep and return raw results dict."""

    raw: dict[str, dict[int, list[dict]]] = {a: {} for a in algos}

    for timer in timers:
        log.info("=== Timer = %d steps ===", timer)

        for algo in algos:
            log.info("  Evaluating %s ...", algo)
            ckpt_dir = ckpt_base / algo / "checkpoints" / "final"

            if not ckpt_dir.exists():
                log.warning("  Checkpoint not found: %s — skipping.", ckpt_dir)
                raw[algo][timer] = []
                continue

            env = CrowdNavEnv(max_steps=timer)

            obs_centers = torch.tensor(
                env.obstacle_centers, dtype=torch.float32, device=DEVICE
            )
            obs_radii = torch.tensor(
                env.obstacle_radii, dtype=torch.float32, device=DEVICE
            )

            # Load model weights
            try:
                if algo == "cddp":
                    actor  = _load_actor(str(ckpt_dir / "actor.pt"), STATE_DIM)
                    z_deon = _load_deon( str(ckpt_dir / "z_deon.pt"), STATE_DIM)
                    def _act(obs):
                        return act_cddp(obs, actor, z_deon, obs_centers, obs_radii)

                elif algo == "ddpg":
                    actor = _load_actor(str(ckpt_dir / "actor.pt"), STATE_DIM)
                    def _act(obs):
                        return act_ddpg(obs, actor)

                elif algo == "ppo":
                    actor = _load_ppo_actor(str(ckpt_dir / "actor.pt"), STATE_DIM)
                    def _act(obs):
                        return act_ppo(obs, actor)

                elif algo == "strict_cbf":
                    actor = _load_actor(str(ckpt_dir / "actor.pt"), STATE_DIM)
                    def _act(obs):
                        return act_strict_cbf(obs, actor, obs_centers, obs_radii)

                elif algo == "ppo_lag":
                    actor = _load_ppo_actor(str(ckpt_dir / "actor.pt"), STATE_DIM)
                    def _act(obs):
                        return act_ppo(obs, actor)

                else:
                    log.warning("Unknown algo: %s", algo)
                    continue

            except FileNotFoundError as e:
                log.warning("  Weight file missing: %s — skipping.", e)
                raw[algo][timer] = []
                continue

            episodes = []
            for i in range(n_eval):
                ep = run_episode(env, _act, seed=1000 + i)
                ep["timer"] = timer
                ep["algo"]  = algo
                # Strip trajectory for JSON size (keep for pareto only)
                episodes.append(ep)

            raw[algo][timer] = episodes
            successes = sum(e["goal_reached"] for e in episodes)
            violations = [e["n_violations"] for e in episodes]
            log.info(
                "    succ=%d/%d  viol_mean=%.1f  viol_max=%d",
                successes, n_eval,
                float(np.mean(violations)), int(np.max(violations)) if violations else 0,
            )
            env.close()

    return raw


# ---------------------------------------------------------------------------
# Aggregation & metric computation
# ---------------------------------------------------------------------------

def aggregate(raw: dict, algos: list[str], timers: list[int],
              excess_threshold: int = EXCESS_THRESHOLD) -> dict:
    """Compute per-(algo, timer) summary statistics."""
    summary: dict[str, dict] = {}

    for algo in algos:
        summary[algo] = {
            "timers":              timers,
            "success_rate":        [],
            "mean_violations":     [],
            "std_violations":      [],
            "mean_excess_viol":    [],
            "prob_excess_gt_0":    [],
            "mean_reward":         [],
            "mean_final_dist":     [],
            "mean_min_dist":       [],
            "catastrophe_rate":    [],
            "mean_avg_stakes":     [],
        }
        for timer in timers:
            eps = raw[algo].get(timer, [])
            if not eps:
                for k in summary[algo]:
                    if k != "timers":
                        summary[algo][k].append(float("nan"))
                continue

            n = len(eps)
            successes   = [int(e["goal_reached"]) for e in eps]
            viols       = [e["n_violations"]       for e in eps]
            rewards     = [e["reward"]             for e in eps]
            final_dists = [e["final_dist"]         for e in eps]
            min_dists   = [e["min_dist"]           for e in eps]
            cats        = [int(e["timeout"])       for e in eps]
            stakes_avg  = [e.get("avg_stakes", 0.0) for e in eps]

            excess_viols = [max(0, v - excess_threshold) for v in viols]

            summary[algo]["success_rate"].append(float(np.mean(successes)))
            summary[algo]["mean_violations"].append(float(np.mean(viols)))
            summary[algo]["std_violations"].append(float(np.std(viols)))
            summary[algo]["mean_excess_viol"].append(float(np.mean(excess_viols)))
            summary[algo]["prob_excess_gt_0"].append(
                float(np.mean([1 if e > 0 else 0 for e in excess_viols]))
            )
            summary[algo]["mean_reward"].append(float(np.mean(rewards)))
            summary[algo]["mean_final_dist"].append(float(np.mean(final_dists)))
            summary[algo]["mean_min_dist"].append(float(np.mean(min_dists)))
            summary[algo]["catastrophe_rate"].append(float(np.mean(cats)))
            summary[algo]["mean_avg_stakes"].append(float(np.mean(stakes_avg)))

    return summary


def build_pareto(raw: dict, algos: list[str], timers: list[int]) -> dict:
    """Pareto trajectory frontier: (mean_violations, mean_final_dist) per algo."""
    pareto: dict[str, list[dict]] = {}
    for algo in algos:
        points = []
        for timer in timers:
            eps = raw[algo].get(timer, [])
            if not eps:
                continue
            mv = float(np.mean([e["n_violations"] for e in eps]))
            md = float(np.mean([e["final_dist"]   for e in eps]))
            points.append({"timer": timer, "mean_violations": mv, "mean_final_dist": md})
        pareto[algo] = points
    return pareto


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_summary_csv(summary: dict, algos: list[str], timers: list[int],
                     path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "algo", "timer",
            "success_rate", "mean_violations", "std_violations",
            "mean_excess_violations", "prob_excess_gt_0",
            "mean_reward", "mean_final_dist", "catastrophe_rate",
            "mean_avg_stakes",
        ])
        for algo in algos:
            s = summary[algo]
            for i, timer in enumerate(timers):
                w.writerow([
                    algo, timer,
                    _fmt(s["success_rate"][i]),
                    _fmt(s["mean_violations"][i]),
                    _fmt(s["std_violations"][i]),
                    _fmt(s["mean_excess_viol"][i]),
                    _fmt(s["prob_excess_gt_0"][i]),
                    _fmt(s["mean_reward"][i]),
                    _fmt(s["mean_final_dist"][i]),
                    _fmt(s["catastrophe_rate"][i]),
                    _fmt(s["mean_avg_stakes"][i]),
                ])


def _fmt(v) -> str:
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


# ---------------------------------------------------------------------------
# Plotting (optional matplotlib)
# ---------------------------------------------------------------------------

def _try_plot(summary: dict, pareto: dict, algos: list[str], timers: list[int],
              plot_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plot generation.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    # Colour palette
    palette = {
        "cddp":       "#2196F3",   # blue  — the hero
        "ddpg":       "#F44336",   # red   — reward hacker
        "ppo":        "#FF9800",   # orange — reward hacker
        "strict_cbf": "#4CAF50",   # green — over-conservative
        "ppo_lag":    "#9C27B0",   # purple — middle ground
    }
    labels = {
        "cddp":       "CDDP (ours)",
        "ddpg":       "DDPG (unconstrained)",
        "ppo":        "PPO (unconstrained)",
        "strict_cbf": "Strict CBF",
        "ppo_lag":    "PPO-Lagrangian",
    }

    def _plot_curve(metric_key: str, ylabel: str, title: str, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for algo in algos:
            vals = summary[algo][metric_key]
            ax.plot(timers, vals, marker="o", label=labels.get(algo, algo),
                    color=palette.get(algo, "#333"), linewidth=2)
        ax.set_xlabel("Initial Timer (steps)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=150)
        plt.close(fig)
        log.info("Saved plot: %s", plot_dir / fname)

    _plot_curve("success_rate",     "Success Rate",          "Success Rate vs. Timer",          "success_rate.png")
    _plot_curve("mean_violations",  "Mean Total Violations", "Total Violations vs. Timer",       "violations.png")
    _plot_curve("mean_excess_viol", "Mean Excess Violations","Excess Violations vs. Timer",      "excess_violations.png")
    _plot_curve("prob_excess_gt_0", "P(Excess > 0)",         "P(Excess > 0) vs. Timer",          "prob_excess.png")
    _plot_curve("catastrophe_rate", "Catastrophe Rate",      "Catastrophe Rate vs. Timer",       "catastrophe_rate.png")
    _plot_curve("mean_avg_stakes",  "Mean CDDP Stakes",      "Avg Stakes vs. Timer (CDDP only)", "avg_stakes.png")

    # Pareto frontier
    fig, ax = plt.subplots(figsize=(8, 6))
    for algo in algos:
        pts = pareto.get(algo, [])
        if not pts:
            continue
        xs = [p["mean_violations"] for p in pts]
        ys = [p["mean_final_dist"] for p in pts]
        ax.scatter(xs, ys, label=labels.get(algo, algo),
                   color=palette.get(algo, "#333"), s=80, zorder=5)
        for p in pts:
            ax.annotate(str(p["timer"]),
                        (p["mean_violations"], p["mean_final_dist"]),
                        fontsize=7, color=palette.get(algo, "#333"),
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean Total Violations", fontsize=13)
    ax.set_ylabel("Mean Final Distance to Goal", fontsize=13)
    ax.set_title("Pareto Trajectory Frontier: Violations vs. Goal Distance", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "pareto_frontier.png", dpi=150)
    plt.close(fig)
    log.info("Saved Pareto plot: %s", plot_dir / "pareto_frontier.png")


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _to_serialisable(obj):
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(i) for i in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (v != v) else v  # NaN → null
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _to_serialisable(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Timer-sweep evaluation for all CDDP-MuJoCo algorithms")
    p.add_argument("--algos",       nargs="+", default=ALGO_NAMES,
                   help="Which algorithms to evaluate")
    p.add_argument("--timers",      nargs="+", type=int, default=DEFAULT_TIMERS,
                   help="Timer lengths (steps) to sweep over")
    p.add_argument("--n-eval",      type=int, default=DEFAULT_N_EVAL,
                   help="Episodes per (algo, timer) cell")
    p.add_argument("--results-dir", type=str,
                   default=str(Path(__file__).parent.parent / "results"))
    p.add_argument("--device",      type=str, default=str(DEVICE))
    p.add_argument("--no-plot",     action="store_true",
                   help="Skip matplotlib plot generation")
    return p.parse_args()


def main() -> None:
    args = _parse()
    global DEVICE
    DEVICE = torch.device(args.device)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_base = results_dir  # checkpoints are at results/<algo>/checkpoints/final/

    log.info("Timer sweep: algos=%s  timers=%s  n_eval=%d",
             args.algos, args.timers, args.n_eval)

    # Run sweep
    raw = run_sweep(
        algos=args.algos,
        timers=args.timers,
        n_eval=args.n_eval,
        results_dir=results_dir,
        ckpt_base=ckpt_base,
    )

    # Aggregate
    summary = aggregate(raw, args.algos, args.timers)
    pareto  = build_pareto(raw, args.algos, args.timers)

    # Save raw data
    raw_out_path = results_dir / "timer_sweep_raw.json"
    with open(raw_out_path, "w") as f:
        # Strip trajectory arrays to keep file manageable
        raw_trimmed: dict = {}
        for algo in raw:
            raw_trimmed[algo] = {}
            for timer, eps in raw[algo].items():
                raw_trimmed[algo][str(timer)] = [
                    {k: v for k, v in e.items() if k != "trajectory"}
                    for e in eps
                ]
        json.dump(_to_serialisable(raw_trimmed), f, indent=2)
    log.info("Raw data saved: %s", raw_out_path)

    # Save summary CSV
    csv_path = results_dir / "timer_sweep_summary.csv"
    save_summary_csv(summary, args.algos, args.timers, csv_path)
    log.info("Summary CSV saved: %s", csv_path)

    # Save Pareto JSON
    pareto_path = results_dir / "pareto_frontier.json"
    with open(pareto_path, "w") as f:
        json.dump(_to_serialisable(pareto), f, indent=2)
    log.info("Pareto data saved: %s", pareto_path)

    # Save summary JSON
    summary_json_path = results_dir / "timer_sweep_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(_to_serialisable(summary), f, indent=2)

    # Plots
    if not args.no_plot:
        _try_plot(summary, pareto, args.algos, args.timers,
                  results_dir / "plots")

    log.info("Evaluation complete.")

    # Print compact table
    print("\n=== Timer Sweep Summary ===")
    header = f"{'Algo':<14} {'Timer':>6} {'SuccRate':>9} {'MeanViol':>9} {'ExViol':>8} {'P(Ex>0)':>8} {'CatRate':>8}"
    print(header)
    print("-" * len(header))
    for algo in args.algos:
        s = summary[algo]
        for i, timer in enumerate(args.timers):
            print(
                f"{algo:<14} {timer:>6d}  "
                f"{s['success_rate'][i]:>8.3f}  "
                f"{s['mean_violations'][i]:>8.2f}  "
                f"{s['mean_excess_viol'][i]:>7.2f}  "
                f"{s['prob_excess_gt_0'][i]:>7.3f}  "
                f"{s['catastrophe_rate'][i]:>7.3f}"
            )
        print()


if __name__ == "__main__":
    main()
