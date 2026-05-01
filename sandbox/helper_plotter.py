"""Standalone Phase 5 plotter for time-aware CDDP checkpoints.

Run from repo root or sandbox/:
  python sandbox/helper_plotter.py
"""

from __future__ import annotations

import os
import sys

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

from train_sandbox_cddp import (  # noqa: E402
    ALPHA_CDDP,
    CHECKPOINT_DIR,
    DEVICE,
    EVAL_START_STATE,
    GAMMA_CBF,
    GOAL,
    LONG_CLOCK_STEPS,
    OBS_C,
    OBS_R,
    TAU_CDDP,
    CDDPAgent,
    eval_clock_config,
    make_env,
)


PLOT_PATH = os.path.join(ROOT_DIR, "cddp_trajectories.png")
START_STATE = EVAL_START_STATE


def load_best_agent() -> tuple[CDDPAgent, object]:
    env = make_env()
    agent = CDDPAgent(env)

    try:
        agent.actor.load_state_dict(
            torch.load(os.path.join(CHECKPOINT_DIR, "cddp_actor_best.pt"), map_location=DEVICE)
        )
        agent.z_free.load_state_dict(
            torch.load(os.path.join(CHECKPOINT_DIR, "cddp_z_free_best.pt"), map_location=DEVICE)
        )
        agent.z_deon.load_state_dict(
            torch.load(os.path.join(CHECKPOINT_DIR, "cddp_z_deon_best.pt"), map_location=DEVICE)
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
    agent: CDDPAgent,
    env: object,
    clock_max_steps: int,
    rollout_cap: int,
) -> tuple[list[tuple[float, float]], list[float]]:
    s0 = torch.tensor([START_STATE], dtype=torch.float32, device=DEVICE)
    env.reset(s0, max_steps=clock_max_steps)

    traj: list[tuple[float, float]] = [tuple(env.state[0, :2].detach().cpu().tolist())]
    stakes: list[float] = []

    for _ in range(rollout_cap):
        s = env.state.detach()
        u_nom = agent.actor(s)
        stake = agent.stakes_from_deon(s, detach=True)
        u_safe = env.project_action_cbf(
            s,
            u_nom,
            stakes=stake,
            gamma=GAMMA_CBF,
            tau=TAU_CDDP,
            alpha_prime=ALPHA_CDDP,
        )

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
) -> LineCollection | None:
    if len(traj) < 2:
        return None

    segments = [[traj[i], traj[i + 1]] for i in range(len(traj) - 1)]
    lc = LineCollection(
        segments,
        cmap=plt.cm.plasma,
        norm=plt.Normalize(0.0, 1.0),
        linewidth=linewidth,
        linestyles=linestyle,
        label=label,
        alpha=0.6,
    )
    lc.set_array(torch.tensor(stakes, dtype=torch.float32).numpy())
    ax.add_collection(lc)
    return lc


def trajectory_midpoint(traj: list[tuple[float, float]]) -> tuple[float, float]:
    if not traj:
        return (START_STATE[0], START_STATE[1])
    return traj[len(traj) // 2]


def decorate_axes(ax: plt.Axes) -> None:
    obstacle = plt.Circle(tuple(OBS_C.tolist()), OBS_R, color="red", alpha=0.28, label="Friction zone")
    ax.add_patch(obstacle)
    ax.plot(GOAL[0].item(), GOAL[1].item(), "g*", markersize=16, label="Goal")
    ax.plot(START_STATE[0], START_STATE[1], "ks", markersize=7, label="Start")
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 5.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

def plot_best() -> None:
    agent, env = load_best_agent()
    min_steps, tight_steps = eval_clock_config()

    long_traj, long_stakes = rollout(
        agent,
        env,
        clock_max_steps=LONG_CLOCK_STEPS,
        rollout_cap=LONG_CLOCK_STEPS,
    )
    tight_traj, tight_stakes = rollout(
        agent,
        env,
        clock_max_steps=tight_steps,
        rollout_cap=tight_steps,
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    decorate_axes(ax)

    long_lc = add_colored_path(
        ax,
        long_traj,
        long_stakes,
        f"Long clock max_steps={LONG_CLOCK_STEPS}",
        "dashed",
        2.6,
    )
    tight_lc = add_colored_path(
        ax,
        tight_traj,
        tight_stakes,
        f"Tight clock max_steps={tight_steps}",
        "solid",
        3.0,
    )
    colorbar_source = tight_lc if tight_lc is not None else long_lc
    if colorbar_source is not None:
        plt.colorbar(colorbar_source, ax=ax, label="Stakes S(s)")

    long_mid = trajectory_midpoint(long_traj)
    tight_mid = trajectory_midpoint(tight_traj)
    ax.annotate(
        "Long Clock",
        xy=long_mid,
        xytext=(max(0.15, long_mid[0] - 0.9), min(4.85, long_mid[1] + 0.8)),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.4},
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "black", "alpha": 0.85},
    )
    ax.annotate(
        "Tight Clock",
        xy=tight_mid,
        xytext=(min(4.65, tight_mid[0] + 0.45), max(0.15, tight_mid[1] - 0.65)),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.4},
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "black", "alpha": 0.85},
    )

    ax.set_title(f"CDDP Spatio-Temporal Adaptation (min={min_steps}, tight={tight_steps})")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Annotated plot saved to {PLOT_PATH}")


if __name__ == "__main__":
    plot_best()
