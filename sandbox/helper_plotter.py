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
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import torch

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    # Inward-facing ticks on all four sides with minor ticks enabled
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
})

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT_DIR, ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ROOT_DIR)


PLOT_PATH = os.path.join(ROOT_DIR, "cddp_trajectories.png")
CLUSTER_SIZE = 10
START_NOISE_STD = 0.03
STAKE_CMAP = "viridis"
STAKE_NORM = plt.Normalize(0.0, 1.0)


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
def make_start_cluster() -> torch.Tensor:
    cddp = _train_module()
    starts = torch.tensor(
        [cddp.EVAL_START_STATE] * CLUSTER_SIZE,
        dtype=torch.float32,
        device=cddp.DEVICE,
    )
    starts[1:, :2] += torch.randn(CLUSTER_SIZE - 1, 2, device=cddp.DEVICE) * START_NOISE_STD
    return starts


@torch.no_grad()
def rollout(
    agent: object,
    env: object,
    start_states: torch.Tensor,
    clock_max_steps: int,
    rollout_cap: int,
) -> tuple[list[list[tuple[float, float]]], list[list[float]]]:
    env.reset(start_states, max_steps=clock_max_steps)

    batch_size = int(start_states.shape[0])
    trajs: list[list[tuple[float, float]]] = [
        [tuple(env.state[i, :2].detach().cpu().tolist())] for i in range(batch_size)
    ]
    stakes_by_env: list[list[float]] = [[] for _ in range(batch_size)]
    active = torch.ones(batch_size, dtype=torch.bool, device=start_states.device)

    for _ in range(rollout_cap):
        s = env.state.detach()
        u_nom = agent.actor(s)
        stake = agent.stakes_from_deon(s, detach=True)
        u_safe = agent.safe_action(s, u_nom, stakes=stake)
        u_safe[~active] = 0.0

        out = env.step(u_safe)
        next_state = out["state"].detach().clone()
        next_state[~active] = s[~active]
        env._state = next_state

        state_cpu = next_state[:, :2].detach().cpu()
        stake_cpu = stake.detach().cpu()

        for i in range(batch_size):
            stakes_by_env[i].append(float(stake_cpu[i].item()))
            trajs[i].append(tuple(state_cpu[i].tolist()))

        active &= ~out["goal_reached"].detach()
        if not active.any():
            break
        if env.current_step >= clock_max_steps:
            break

    return trajs, stakes_by_env


def add_colored_path(
    ax: plt.Axes,
    traj: list[tuple[float, float]],
    stakes: list[float],
    label: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> LineCollection | None:
    if len(traj) < 2:
        return None

    segments = [[traj[i], traj[i + 1]] for i in range(len(traj) - 1)]
    lc = LineCollection(
        segments,
        cmap=plt.get_cmap(STAKE_CMAP),
        norm=STAKE_NORM,
        linewidth=linewidth,
        linestyles=linestyle,
        label=label,
        alpha=alpha,
        zorder=zorder,
    )
    lc.set_array(torch.tensor(stakes, dtype=torch.float32).numpy())
    ax.add_collection(lc)
    return lc


def add_direction_arrow(
    ax: plt.Axes,
    traj: list[tuple[float, float]],
    frac: float,
    color: str,
    zorder: int = 14,
) -> None:
    """Draw a small arrowhead at *frac* of the way along *traj*."""
    idx = max(0, min(int(frac * (len(traj) - 2)), len(traj) - 2))
    x0, y0 = traj[idx]
    x1, y1 = traj[idx + 1]
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=1.8,
            mutation_scale=16,
        ),
        zorder=zorder,
    )


def trajectory_midpoint(traj: list[tuple[float, float]]) -> tuple[float, float]:
    if not traj:
        start_state = _train_module().EVAL_START_STATE
        return (start_state[0], start_state[1])
    return traj[len(traj) // 2]


def final_distance_to_goal(traj: list[tuple[float, float]]) -> float:
    cddp = _train_module()
    if not traj:
        return float("inf")
    goal_x, goal_y = cddp.GOAL.detach().cpu().tolist()
    final_x, final_y = traj[-1]
    return ((final_x - goal_x) ** 2 + (final_y - goal_y) ** 2) ** 0.5


def best_run_index(trajs: list[list[tuple[float, float]]]) -> int:
    return min(range(len(trajs)), key=lambda idx: final_distance_to_goal(trajs[idx]))


def decorate_axes(ax: plt.Axes) -> None:
    cddp = _train_module()
    start_state = cddp.EVAL_START_STATE
    obs_c = tuple(cddp.OBS_C.tolist())

    # ── Layered constraint zone (outermost → innermost, so inner sits on top) ──
    # Layer 3 (outermost): extra -0.05 penalty
    ring3 = plt.Circle(obs_c, cddp.LAYER_3_R, color="#e74c3c", alpha=0.08, zorder=1)
    ring3_edge = plt.Circle(obs_c, cddp.LAYER_3_R, fill=False,
                            edgecolor="#e74c3c", linewidth=0.8,
                            linestyle=(0, (3, 4)), alpha=0.55, zorder=2)
    # Layer 2 (middle): extra -0.10 penalty
    ring2 = plt.Circle(obs_c, cddp.LAYER_2_R, color="#e74c3c", alpha=0.12, zorder=2)
    ring2_edge = plt.Circle(obs_c, cddp.LAYER_2_R, fill=False,
                            edgecolor="#c0392b", linewidth=0.9,
                            linestyle=(0, (4, 3)), alpha=0.65, zorder=3)
    # Layer 1 (second-deepest): extra -0.25 penalty
    ring1 = plt.Circle(obs_c, cddp.LAYER_1_R, color="#e74c3c", alpha=0.18, zorder=3)
    ring1_edge = plt.Circle(obs_c, cddp.LAYER_1_R, fill=False,
                            edgecolor="#a93226", linewidth=1.1,
                            linestyle=(0, (5, 3)), alpha=0.80, zorder=4)
    # Hard CBF boundary: extra -0.4 penalty + hatch
    hard = plt.Circle(obs_c, cddp.OBS_R, color="#e74c3c", alpha=0.22, zorder=4)
    hard_hatch = plt.Circle(obs_c, cddp.OBS_R, facecolor="none",
                            edgecolor="#922b21", linewidth=0.0,
                            hatch="//////", alpha=0.30, zorder=5)
    hard_edge = plt.Circle(obs_c, cddp.OBS_R, fill=False,
                           edgecolor="#7b241c", linewidth=1.8,
                           linestyle=(0, (5, 2)), zorder=6)
    for patch in (ring3, ring3_edge, ring2, ring2_edge,
                  ring1, ring1_edge, hard, hard_hatch, hard_edge):
        ax.add_patch(patch)

    # Goal circle matching the physics hit-box
    goal = tuple(cddp.GOAL.tolist())
    goal_hitbox = plt.Circle(goal, cddp.GOAL_TOL, color="green", alpha=0.4, zorder=7)
    ax.add_patch(goal_hitbox)
    ax.plot(goal[0], goal[1], marker="*", color="green",
            markeredgecolor="darkgreen", markersize=11, zorder=9)

    ax.plot(start_state[0], start_state[1], "ks", markersize=7, zorder=8)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$ (position)")
    ax.set_ylabel(r"$x_2$ (position)")
    ax.grid(True, alpha=0.2)


def draw_agent_panel(
    ax: plt.Axes,
    trajs: list[list[tuple[float, float]]],
    stakes_by_env: list[list[float]],
    title: str,
    linestyle: str,
) -> None:
    decorate_axes(ax)
    best_idx = best_run_index(trajs)

    for idx, (traj, stakes) in enumerate(zip(trajs, stakes_by_env)):
        if idx == best_idx:
            continue
        add_colored_path(ax, traj, stakes, "", linestyle, 1.0, alpha=0.3, zorder=2)

    best_traj = trajs[best_idx]
    best_stakes = stakes_by_env[best_idx]
    add_colored_path(ax, best_traj, best_stakes, "", "solid", 2.5, alpha=1.0, zorder=10)

    ax.set_title(title)


def add_global_legend(fig: plt.Figure) -> None:
    cddp = _train_module()
    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="black",
               markeredgecolor="black", markersize=7, label="Start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="green",
               markeredgecolor="darkgreen", markersize=11, label="Goal"),
        Patch(facecolor="#e74c3c", edgecolor="#7b241c", hatch="//////",
              alpha=0.22, label="Hard Boundary (-0.4)"),
        Patch(facecolor="#e74c3c", edgecolor="#a93226", linestyle=(0, (5, 3)),
              linewidth=1.1, alpha=0.18, label=f"Layer 1 ({cddp.LAYER_1_PENALTY:g})"),
        Patch(facecolor="#e74c3c", edgecolor="#c0392b", linestyle=(0, (4, 3)),
              linewidth=0.9, alpha=0.12, label=f"Layer 2 ({cddp.LAYER_2_PENALTY:g})"),
        Patch(facecolor="#e74c3c", edgecolor="#e74c3c", linestyle=(0, (3, 4)),
              linewidth=0.8, alpha=0.08, label=f"Layer 3 ({cddp.LAYER_3_PENALTY:g})"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.05),
        frameon=False,
    )


def draw_cluster_plot(
    fig: plt.Figure,
    ax1: plt.Axes,
    ax2: plt.Axes,
    long_trajs: list[list[tuple[float, float]]],
    long_stakes: list[list[float]],
    tight_trajs: list[list[tuple[float, float]]],
    tight_stakes: list[list[float]],
) -> None:
    draw_agent_panel(
        ax1,
        tight_trajs,
        tight_stakes,
        "Tight Clock Agent",
        "solid",
    )
    draw_agent_panel(
        ax2,
        long_trajs,
        long_stakes,
        "Long Clock Agent",
        "dashed",
    )

    fig.subplots_adjust(bottom=0.22, right=0.86, wspace=0.28)

    cbar = fig.colorbar(
        ScalarMappable(norm=STAKE_NORM, cmap=plt.get_cmap(STAKE_CMAP)),
        ax=[ax1, ax2],
        fraction=0.03,
        pad=0.08,
    )
    cbar.set_label(r"Stakes $S(s)$")

    fig.suptitle(r"CDDP Spatio-Temporal Adaptation")
    add_global_legend(fig)


def plot_best() -> None:
    cddp = _train_module()
    agent, env = load_best_agent()
    _, tight_steps = cddp.eval_clock_config()

    long_trajs, long_stakes = rollout(
        agent,
        env,
        make_start_cluster(),
        clock_max_steps=cddp.LONG_CLOCK_STEPS,
        rollout_cap=cddp.LONG_CLOCK_STEPS,
    )
    tight_trajs, tight_stakes = rollout(
        agent,
        env,
        make_start_cluster(),
        clock_max_steps=tight_steps,
        rollout_cap=tight_steps,
    )

    def render_and_save() -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        draw_cluster_plot(
            fig,
            ax1,
            ax2,
            long_trajs,
            long_stakes,
            tight_trajs,
            tight_stakes,
        )
        fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
        plt.close(fig)

    try:
        render_and_save()
    except RuntimeError as exc:
        if "latex could not be found" not in str(exc):
            raise
        plt.close("all")
        plt.rcParams.update({"text.usetex": False})
        print("Warning: LaTeX executable not found; falling back to Matplotlib text rendering.")
        render_and_save()

    print(f"Plot saved to {PLOT_PATH}")


if __name__ == "__main__":
    plot_best()
