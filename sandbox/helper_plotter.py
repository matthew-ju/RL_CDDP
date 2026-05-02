"""Standalone Phase 5 plotter for time-aware CDDP checkpoints.

Run from repo root or sandbox/:
  python sandbox/helper_plotter.py
"""

from __future__ import annotations

import os
import sys
import importlib

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/private", "tmp", "cddp_rl_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT_DIR, ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ROOT_DIR)


PLOT_PATH = os.path.join(ROOT_DIR, "cddp_trajectories.png")


def _train_module():
    required = ("CDDPAgent", "make_env", "CHECKPOINT_DIR", "DEVICE")
    for module_name in ("__main__", "train_sandbox_cddp", "sandbox.train_sandbox_cddp"):
        module = sys.modules.get(module_name)
        if module is not None and all(hasattr(module, name) for name in required):
            return module
    return importlib.import_module("train_sandbox_cddp")


def load_best_agent() -> tuple[object, object]:
    cddp = _train_module()
    env = cddp.make_env()
    agent = cddp.CDDPAgent(env)

    try:
        agent.actor.load_state_dict(
            torch.load(os.path.join(cddp.CHECKPOINT_DIR, "cddp_actor_best.pt"), map_location=cddp.DEVICE)
        )
        agent.z_free.load_state_dict(
            torch.load(os.path.join(cddp.CHECKPOINT_DIR, "cddp_z_free_best.pt"), map_location=cddp.DEVICE)
        )
        agent.z_deon.load_state_dict(
            torch.load(os.path.join(cddp.CHECKPOINT_DIR, "cddp_z_deon_best.pt"), map_location=cddp.DEVICE)
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Best checkpoints are incompatible with the 5D time-augmented "
            "architecture. Run train_sandbox_cddp.py to regenerate them."
        ) from exc
    agent.actor.eval()
    agent.z_free.eval()
    agent.z_deon.eval()
    return agent, env


@torch.no_grad()
def rollout(
    agent: object,
    env: object,
    clock_max_steps: int,
    rollout_cap: int,
) -> tuple[list[tuple[float, float]], list[float]]:
    cddp = _train_module()
    start_state = cddp.EVAL_START_STATE
    s0 = torch.tensor([start_state], dtype=torch.float32, device=cddp.DEVICE)
    env.reset(s0, max_steps=clock_max_steps)

    traj: list[tuple[float, float]] = [tuple(env.state[0, :2].detach().cpu().tolist())]
    stakes: list[float] = []

    for _ in range(rollout_cap):
        s = env.state.detach()
        u_nom = agent.actor(s)
        stake = agent.stakes_from_deon(s, detach=True)
        u_safe = agent.safe_action(s, u_nom, stakes=stake)

        out = env.step(u_safe)
        stakes.append(stake.item())
        traj.append(tuple(out["state"][0, :2].detach().cpu().tolist()))

        if out["goal_reached"].item():
            break
        if env.current_step >= clock_max_steps:
            break

    return traj, stakes


def add_colored_path(
    ax: plt.Axes,
    traj: list[tuple[float, float]],
    stakes: list[float],
    label: str,
    linestyle: str,
    linewidth: float,
    cmap_name: str = "plasma",
) -> LineCollection | None:
    if len(traj) < 2:
        return None

    segments = [[traj[i], traj[i + 1]] for i in range(len(traj) - 1)]
    lc = LineCollection(
        segments,
        cmap=plt.get_cmap(cmap_name),
        norm=plt.Normalize(0.0, 1.0),
        linewidth=linewidth,
        linestyles=linestyle,
        label=label,
        alpha=0.8,
    )
    lc.set_array(torch.tensor(stakes, dtype=torch.float32).numpy())
    ax.add_collection(lc)
    return lc


def trajectory_midpoint(traj: list[tuple[float, float]]) -> tuple[float, float]:
    if not traj:
        start_state = _train_module().EVAL_START_STATE
        return (start_state[0], start_state[1])
    return traj[len(traj) // 2]


def decorate_axes(ax: plt.Axes) -> None:
    cddp = _train_module()
    start_state = cddp.EVAL_START_STATE

    obstacle = plt.Circle(tuple(cddp.OBS_C.tolist()), cddp.OBS_R, color="red", alpha=0.28)
    ax.add_patch(obstacle)

    # Plot goal as a circle matching the exact physics hit-box.
    goal_hitbox = plt.Circle(tuple(cddp.GOAL.tolist()), cddp.GOAL_TOL, color="green", alpha=0.4)
    ax.add_patch(goal_hitbox)

    ax.plot(start_state[0], start_state[1], "ks", markersize=7)

    # Expand view to see out-of-bounds maneuvers.
    ax.set_xlim(-2.0, 6.0)
    ax.set_ylim(-2.0, 6.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)


def plot_best() -> None:
    cddp = _train_module()
    agent, env = load_best_agent()
    min_steps, tight_steps = cddp.eval_clock_config()

    long_traj, long_stakes = rollout(
        agent,
        env,
        clock_max_steps=cddp.LONG_CLOCK_STEPS,
        rollout_cap=cddp.LONG_CLOCK_STEPS,
    )
    tight_traj, tight_stakes = rollout(
        agent,
        env,
        clock_max_steps=tight_steps,
        rollout_cap=tight_steps,
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    decorate_axes(ax)

    long_lc = add_colored_path(
        ax,
        long_traj,
        long_stakes,
        "",
        "dashed",
        2.6,
        cmap_name="Blues",
    )
    tight_lc = add_colored_path(
        ax,
        tight_traj,
        tight_stakes,
        "",
        "solid",
        3.0,
        cmap_name="Reds",
    )
    if long_lc is not None:
        cbar_long = plt.colorbar(long_lc, ax=ax, fraction=0.046, pad=0.04)
        cbar_long.set_label("Stakes S(s) - Long Clock")

    if tight_lc is not None:
        cbar_tight = plt.colorbar(tight_lc, ax=ax, fraction=0.046, pad=0.04)
        cbar_tight.set_label("Stakes S(s) - Tight Clock")

    ax.set_title(f"CDDP Spatio-Temporal Adaptation (min={min_steps}, tight={tight_steps})")

    # Add direct text labels to the plot instead of a legend.
    if len(long_traj) > 5:
        ax.text(
            long_traj[5][0],
            long_traj[5][1] + 0.3,
            "Long Clock",
            color="blue",
            fontsize=11,
            fontweight="bold",
        )
    if len(tight_traj) > 5:
        ax.text(
            tight_traj[5][0],
            tight_traj[5][1] - 0.3,
            "Tight Clock",
            color="red",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {PLOT_PATH}")


if __name__ == "__main__":
    plot_best()
