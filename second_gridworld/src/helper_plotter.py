"""Standalone CDDP evaluation plotter.

Run from repo root or sandbox/:
  python sandbox/helper_plotter.py
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/private", "tmp", "cddp_rl_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import torch

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT_DIR, ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ROOT_DIR)

RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
PLOT_PATH = os.path.join(RESULTS_ROOT, "cddp", "cddp_trajectories.png")
STAKE_CMAP = "viridis"
STAKE_NORM = plt.Normalize(0.0, 1.0)
BASELINE_ALGOS = {"ppo", "strict", "lagrangian", "cpo"}
HARD_BOUNDARY_HATCH = "/////////"


def _train_module():
    required = ("CDDPAgent", "make_env", "CHECKPOINT_DIR", "DEVICE")
    for module_name in ("__main__", "train_sandbox_cddp", "src.train_sandbox_cddp", "sandbox.train_sandbox_cddp"):
        module = sys.modules.get(module_name)
        if module is not None and all(hasattr(module, name) for name in required):
            return module
    return importlib.import_module("src.train_sandbox_cddp")


def _baseline_module():
    required = ("ActorCritic", "CHECKPOINT_DIR", "DEVICE")
    for module_name in ("__main__", "train_baselines_ppo", "src.train_baselines_ppo", "sandbox.train_baselines_ppo"):
        module = sys.modules.get(module_name)
        if module is not None and all(hasattr(module, name) for name in required):
            return module
    return importlib.import_module("src.train_baselines_ppo")


def _checkpoint_paths(cddp: object, suffix: str) -> tuple[str, str, str]:
    return (
        os.path.join(cddp.CHECKPOINT_DIR, f"cddp_actor{suffix}.pt"),
        os.path.join(cddp.CHECKPOINT_DIR, f"cddp_z_free{suffix}.pt"),
        os.path.join(cddp.CHECKPOINT_DIR, f"cddp_z_deon{suffix}.pt"),
    )


def _load_checkpoint_set(agent: object, cddp: object, suffix: str) -> bool:
    actor_path, z_free_path, z_deon_path = _checkpoint_paths(cddp, suffix)
    if not all(os.path.exists(path) for path in (actor_path, z_free_path, z_deon_path)):
        return False
    try:
        agent.actor.load_state_dict(torch.load(actor_path, map_location=cddp.DEVICE))
        agent.z_free.load_state_dict(torch.load(z_free_path, map_location=cddp.DEVICE))
        agent.z_deon.load_state_dict(torch.load(z_deon_path, map_location=cddp.DEVICE))
    except RuntimeError:
        return False
    return True


def load_best_agent(algo: str = "cddp") -> tuple[object, object]:
    cddp = _train_module()
    env = cddp.make_env()

    if algo == "cddp":
        agent = cddp.CDDPAgent(env)
        if not (_load_checkpoint_set(agent, cddp, "_best") or _load_checkpoint_set(agent, cddp, "")):
            raise RuntimeError(
                "Checkpoints are incompatible with the current 5D time-augmented "
                "architecture. Run train_sandbox_cddp.py to regenerate them."
            )
        agent.actor.eval()
        agent.z_free.eval()
        agent.z_deon.eval()
        return agent, env

    if algo not in BASELINE_ALGOS:
        raise ValueError(f"Unsupported algo '{algo}'. Expected cddp or one of {sorted(BASELINE_ALGOS)}.")

    baseline = _baseline_module()
    agent = baseline.ActorCritic(cddp.MODEL_HIDDEN_DIM, cddp.MODEL_HIDDEN_LAYERS).to(cddp.DEVICE)
    if hasattr(baseline, "checkpoint_path_for_algo"):
        checkpoint_path = baseline.checkpoint_path_for_algo(algo)
    else:
        checkpoint_path = os.path.join(baseline.CHECKPOINT_DIR, f"{algo}_actor_critic.pt")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            f"Missing baseline checkpoint for '{algo}': {checkpoint_path}. "
            f"Run train_baselines_ppo.py --algo {algo} first."
        )
    try:
        agent.load_state_dict(torch.load(checkpoint_path, map_location=cddp.DEVICE))
    except RuntimeError as exc:
        raise RuntimeError(
            f"Baseline checkpoint for '{algo}' is incompatible with the current ActorCritic: {exc}"
        ) from exc
    agent.eval()
    return agent, env


@torch.no_grad()
def make_eval_context(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    cddp = _train_module()
    start_states, goals = cddp.sample_episode_context(1, seed=seed)
    return start_states[0].detach().clone(), goals[0].detach().clone()


def add_colored_path(
    ax: plt.Axes,
    traj: list[tuple[float, float]],
    stakes: list[float],
    linewidth: float,
    alpha: float,
    zorder: int,
) -> LineCollection | None:
    if len(traj) < 2:
        return None

    segments = [[traj[i], traj[i + 1]] for i in range(len(traj) - 1)]
    stake_values = stakes[: len(segments)]
    if not stake_values:
        stake_values = [0.0] * len(segments)
    elif len(stake_values) < len(segments):
        stake_values.extend([stake_values[-1]] * (len(segments) - len(stake_values)))

    lc = LineCollection(
        segments,
        cmap=plt.get_cmap(STAKE_CMAP),
        norm=STAKE_NORM,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    lc.set_array(torch.tensor(stake_values, dtype=torch.float32).numpy())
    ax.add_collection(lc)
    return lc


def decorate_axes(
    ax: plt.Axes,
    env: object,
    start_xys: list[tuple[float, float]],
    goal_xys: list[tuple[float, float]],
    best_idx: int = 0,
) -> None:
    """Draw obstacles, goal zone, start zone, and per-rollout start markers.

    start_xys : one entry per rollout, in the same order as the rollouts list.
    best_idx  : index of the highlighted (best-reward) rollout.
    """
    cddp = _train_module()
    centers = cddp.obstacle_centers_at(timestep=0).detach().cpu()
    core_r = cddp.OBS_R.detach().cpu()
    layer_1_r = cddp.LAYER_1_R.detach().cpu()
    layer_2_r = cddp.LAYER_2_R.detach().cpu()
    layer_3_r = cddp.LAYER_3_R.detach().cpu()

    for center, r_core, r_l1, r_l2, r_l3 in zip(centers, core_r, layer_1_r, layer_2_r, layer_3_r):
        x, y = float(center[0].item()), float(center[1].item())
        # Filled layers — outermost first so inner rings paint on top
        ax.add_patch(plt.Circle((x, y), float(r_l3.item()), color="#ef9a9a", alpha=0.10, zorder=1))
        ax.add_patch(plt.Circle((x, y), float(r_l2.item()), color="#ef5350", alpha=0.12, zorder=2))
        ax.add_patch(plt.Circle((x, y), float(r_l1.item()), color="#e53935", alpha=0.14, zorder=3))
        ax.add_patch(plt.Circle((x, y), float(r_core.item()), color="#b71c1c", alpha=0.30, zorder=4))
        # Hard-boundary hatch
        ax.add_patch(
            plt.Circle(
                (x, y),
                float(r_core.item()),
                facecolor="none",
                edgecolor="#7b241c",
                hatch=HARD_BOUNDARY_HATCH,
                alpha=0.14,
                zorder=5,
            )
        )
        arrow_half_width = max(0.18, 0.70 * float(r_core.item()))
        ax.annotate(
            "",
            xy=(x + arrow_half_width, y),
            xytext=(x - arrow_half_width, y),
            arrowprops={
                "arrowstyle": "<->",
                "color": "#7b241c",
                "lw": 1.0,
                "alpha": 0.45,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=8,
        )
        # Dashed ring borders matching the static aesthetic
        for r_val, ec, lw in (
            (float(r_l3.item()), "#e57373", 0.7),
            (float(r_l2.item()), "#e53935", 0.8),
            (float(r_l1.item()), "#c62828", 1.0),
            (float(r_core.item()), "#7b241c", 1.5),
        ):
            ax.add_patch(
                plt.Circle(
                    (x, y), r_val,
                    fill=False,
                    edgecolor=ec,
                    linewidth=lw,
                    linestyle=(0, (4, 3)),
                    alpha=0.55,
                    zorder=6,
                )
            )

    # Goal sampling zone is not the default terminal condition.  Success is
    # the sampled-goal tolerance circle unless terminate-on-goal-zone is set.
    goal_low = cddp.GOAL_LOW.detach().cpu()
    goal_high = cddp.GOAL_HIGH.detach().cpu()
    goal_x = float(goal_low[0].item())
    goal_y = float(goal_low[1].item())
    goal_w = float((goal_high[0] - goal_low[0]).item())
    goal_h = float((goal_high[1] - goal_low[1]).item())
    ax.add_patch(
        plt.Rectangle(
            (goal_x, goal_y),
            goal_w,
            goal_h,
            facecolor="#4caf50",
            edgecolor="#388e3c",
            linewidth=1.0,
            alpha=0.07,
            zorder=7,
        )
    )
    ax.text(
        goal_x - 0.75,
        goal_y + 0.18,
        "goal sample zone",
        color="#1b5e20",
        fontsize=8,
        alpha=0.85,
        zorder=16,
    )
    if goal_xys:
        for i, gxy in enumerate(goal_xys):
            highlighted = i == best_idx
            ax.add_patch(
                plt.Circle(
                    gxy,
                    float(cddp.GOAL_TOL),
                    fill=False,
                    edgecolor="#0b6b2b",
                    linewidth=1.4 if highlighted else 0.8,
                    linestyle="-" if highlighted else (0, (3, 2)),
                    alpha=0.95 if highlighted else 0.28,
                    zorder=13 if highlighted else 12,
                )
            )
            ax.plot(
                gxy[0],
                gxy[1],
                marker="x",
                color="#0b6b2b",
                markersize=5 if highlighted else 3.5,
                markeredgewidth=1.0,
                alpha=0.90 if highlighted else 0.25,
                zorder=14 if highlighted else 12,
                clip_on=False,
            )
        best_goal = goal_xys[best_idx]
        label_x = max(-0.2, min(best_goal[0] - 1.65, 8.15))
        label_y = max(8.25, min(best_goal[1] + float(cddp.GOAL_TOL) + 0.18, 10.35))
        ax.text(
            label_x,
            label_y,
            "terminal tolerance",
            color="#0b6b2b",
            fontsize=8,
            alpha=0.90,
            zorder=16,
        )

    # ── Start zone: very faint box showing where the start can be sampled ────
    start_low = cddp.START_LOW.detach().cpu()
    start_high = cddp.START_HIGH.detach().cpu()
    start_x = float(start_low[0].item())
    start_y = float(start_low[1].item())
    start_w = float((start_high[0] - start_low[0]).item())
    start_h = float((start_high[1] - start_low[1]).item())
    ax.add_patch(
        plt.Rectangle(
            (start_x, start_y),
            start_w,
            start_h,
            facecolor="black",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.05,
            zorder=8,
        )
    )

    if start_xys:
        # Individual start markers — dim for non-best, highlighted for best
        for i, sxy in enumerate(start_xys):
            if i == best_idx:
                continue  # best start re-drawn at full opacity in draw_agent_panel
            ax.plot(
                sxy[0], sxy[1],
                marker="s",
                color="black",
                markersize=4,
                alpha=0.25,
                zorder=9,
                clip_on=False,
            )
    ax.set_xlim(-0.5, 11.0)
    ax.set_ylim(-0.5, 11.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$ (position)")
    ax.set_ylabel(r"$x_2$ (position)")
    ax.minorticks_on()
    ax.grid(True, linestyle=":", color="lightgray")


@torch.no_grad()
def rollout_episode(
    agent: object,
    env: object,
    start_state: torch.Tensor,
    goal: torch.Tensor,
    clock_max_steps: int,
    agent_profile: str,
    seed: int,
    context_id: str,
    algo: str = "cddp",
) -> tuple[list[tuple[float, float]], list[float], dict[str, object]]:
    cddp = _train_module()
    start_batch = start_state.reshape(1, 5).to(cddp.DEVICE).clone()
    goal_batch = goal.reshape(1, 2).to(cddp.DEVICE).clone()
    start_batch[:, 4] = float(clock_max_steps) / float(cddp.LONG_CLOCK_STEPS)

    env.reset(start_batch, max_steps=clock_max_steps)
    env.set_goal(goal_batch)
    if env.elapsed_steps is not None:
        env.elapsed_steps[:] = 0.0

    start_xy = tuple(float(v) for v in start_batch[0, :2].detach().cpu().tolist())
    goal_xy = tuple(float(v) for v in goal_batch[0].detach().cpu().tolist())
    traj = [start_xy]
    stakes_by_step: list[float] = []
    constraint_violations: list[dict[str, object]] = []
    total_reward = 0.0
    episode_success = False

    for step_idx in range(clock_max_steps):
        s = env.state.detach()
        prev_pos = s[:, :2].clone()
        u_nom = agent.actor(s)
        if algo == "cddp":
            stakes = agent.stakes_from_deon(s, detach=True)
            action = agent.safe_action(s, u_nom, stakes=stakes)
        else:
            stakes = torch.zeros(s.shape[0], dtype=s.dtype, device=s.device)
            if algo == "strict":
                action = env.project_action_cbf(
                    s,
                    u_nom,
                    stakes=stakes,
                    gamma=cddp.GAMMA_CBF,
                    tau=cddp.TAU_CDDP,
                    alpha_prime=cddp.ALPHA_CDDP,
                )
            else:
                action = u_nom.clamp(-cddp.U_MAX, cddp.U_MAX)
        out = env.step(action)

        next_state = out["state"].detach().clone()
        remaining_steps = max(0, clock_max_steps - (step_idx + 1))
        next_state[:, 4] = float(remaining_steps) / float(cddp.LONG_CLOCK_STEPS)
        env._state = next_state
        if env.elapsed_steps is not None:
            env.elapsed_steps[:] = float(step_idx + 1)

        reached = cddp.success_mask(next_state[:, :2], goal_batch, out["goal_reached"].detach())
        episode_success = episode_success or bool(reached.detach().cpu().item())
        timeout = torch.tensor(
            [step_idx + 1 >= clock_max_steps],
            dtype=torch.bool,
            device=cddp.DEVICE,
        ) & ~reached
        reward = cddp.compute_reward(
            next_state[:, :2],
            prev_pos,
            goal_batch,
            reached,
            out["unsafe"].detach(),
            timeout,
            timestep=torch.tensor([step_idx + 1], dtype=torch.float32, device=cddp.DEVICE),
        )
        total_reward += float(reward.detach().cpu().item())

        stakes_by_step.append(float(stakes.detach().cpu().item()))
        next_xy = tuple(float(v) for v in next_state[0, :2].detach().cpu().tolist())
        traj.append(next_xy)

        violations = out["constraint_violations"][0].detach().cpu()
        distances = out["obstacle_distances"][0].detach().cpu()
        for obs_idx in torch.where(violations)[0].tolist():
            constraint_violations.append({
                "step": step_idx + 1,
                "obstacle": int(obs_idx + 1),
                "x": next_xy[0],
                "y": next_xy[1],
                "distance": float(distances[obs_idx].item()),
                "core_radius": float(cddp.OBS_R[obs_idx].item()),
            })

        if bool((reached | timeout).item()):
            break

    record = {
        "seed": int(seed),
        "context_id": context_id,
        "agent_profile": agent_profile,
        "timer": int(clock_max_steps),
        "max_steps": int(clock_max_steps),
        "start_x": start_xy[0],
        "start_y": start_xy[1],
        "goal_x": goal_xy[0],
        "goal_y": goal_xy[1],
        "total_reward": total_reward,
        "constraint_violations": constraint_violations,
    }
    record.update(cddp.compute_goal_capture_diagnostics(
        traj,
        goal_xy,
        clock_max_steps,
        success=episode_success,
        total_violations=len(constraint_violations),
        stakes_by_step=stakes_by_step,
    ))
    record["success"] = bool(record["success"])
    record["total_steps"] = int(record["total_steps"])
    record["total_violations"] = int(record["total_violations"])
    record["constraint_violations"] = constraint_violations
    return traj, stakes_by_step, record


def record_goal_xy(record: dict[str, object]) -> tuple[float, float]:
    return (float(record["goal_x"]), float(record["goal_y"]))


def terminal_status(record: dict[str, object]) -> str:
    if record.get("constraint_violations") and not bool(record.get("success", False)):
        return "crash"
    if bool(record.get("success", False)):
        return "success"
    return "timeout"


def add_terminal_marker(
    ax: plt.Axes,
    traj: list[tuple[float, float]],
    record: dict[str, object],
    *,
    highlighted: bool,
) -> None:
    if not traj:
        return

    x, y = traj[-1]
    status = terminal_status(record)
    # Collision runs: intentionally not plotted (clutter removal)
    if status == "crash":
        return
    alpha = 1.0 if highlighted else 0.30
    zorder = 14 if highlighted else 11
    if status == "success":
        ax.plot(
            x,
            y,
            marker="*",
            linestyle="None",
            color="#1b8e3e",
            markeredgecolor="#0b5d1e",
            markersize=13 if highlighted else 10,
            markeredgewidth=1.1,
            alpha=alpha,
            zorder=zorder,
            clip_on=False,
        )
    elif status == "timeout":
        ax.plot(
            x,
            y,
            marker="s",
            linestyle="None",
            markerfacecolor="#c62828",
            markeredgecolor="#7f0000",
            markersize=8 if highlighted else 6.5,
            markeredgewidth=1.1,
            alpha=alpha,
            zorder=zorder,
            clip_on=False,
        )


def draw_agent_panel(
    ax: plt.Axes,
    env: object,
    rollouts: list[tuple[list[tuple[float, float]], list[float], dict[str, object]]],
    title: str,
) -> None:
    if not rollouts:
        raise ValueError("draw_agent_panel requires at least one rollout.")

    best_idx = max(range(len(rollouts)), key=lambda idx: float(rollouts[idx][2]["total_reward"]))

    start_xys = [
        (float(record["start_x"]), float(record["start_y"]))
        for _traj, _stakes, record in rollouts
    ]
    goal_xys = [record_goal_xy(record) for _traj, _stakes, record in rollouts]
    decorate_axes(ax, env, start_xys, goal_xys, best_idx=best_idx)

    for idx, (traj, stakes, record) in enumerate(rollouts):
        if idx == best_idx:
            continue
        add_colored_path(ax, traj, stakes, linewidth=1.0, alpha=0.25, zorder=5)
        add_terminal_marker(ax, traj, record, highlighted=False)

    best_traj, best_stakes, best_record = rollouts[best_idx]
    add_colored_path(ax, best_traj, best_stakes, linewidth=2.5, alpha=1.0, zorder=10)
    add_terminal_marker(ax, best_traj, best_record, highlighted=True)

    # Best-run start marker at full opacity, drawn on top of dim non-best starts
    best_start = start_xys[best_idx]
    ax.plot(
        best_start[0], best_start[1],
        marker="s",
        color="black",
        markersize=7,
        alpha=1.0,
        zorder=15,
        clip_on=False,
    )

    ax.set_title(title)


def add_global_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D(
            [0], [0],
            marker="s",
            color="none",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=7,
            label="Start",
        ),
        Patch(
            facecolor="black",
            edgecolor="black",
            alpha=0.10,
            label="Start zone",
        ),
        Patch(
            facecolor="#4caf50",
            edgecolor="#388e3c",
            alpha=0.20,
            label="Goal sampling zone",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="#0b6b2b",
            markerfacecolor="none",
            linestyle="None",
            markersize=8,
            label="Actual terminal tolerance",
        ),
        Patch(
            facecolor="#b71c1c",
            edgecolor="#7b241c",
            hatch=HARD_BOUNDARY_HATCH,
            alpha=0.30,
            label="Hard Boundary (-0.4)",
        ),
        Patch(
            facecolor="#e53935",
            edgecolor="#e53935",
            alpha=0.12,
            label="Penalty buffer",
        ),
        Line2D(
            [0], [0],
            marker="*",
            color="#1b8e3e",
            markeredgecolor="#0b5d1e",
            linestyle="None",
            markersize=11,
            label="Success",
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="#c62828",
            markerfacecolor="#c62828",
            markeredgecolor="#7f0000",
            linestyle="None",
            markersize=7,
            label="Timeout",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.035),
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.0,
    )


def render_evaluation_figure(
    env: object,
    tight_rollouts: list[tuple[list[tuple[float, float]], list[float], dict[str, object]]],
    long_rollouts: list[tuple[list[tuple[float, float]], list[float], dict[str, object]]],
    tight_steps: int,
    long_steps: int,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    draw_agent_panel(
        ax1,
        env,
        tight_rollouts,
        f"Tight Clock Agent ({tight_steps} Max Steps)",
    )
    draw_agent_panel(
        ax2,
        env,
        long_rollouts,
        f"Long Clock Agent ({long_steps} Max Steps)",
    )
    fig.subplots_adjust(bottom=0.18, right=0.88, top=0.92, wspace=0.25)
    cbar = fig.colorbar(
        ScalarMappable(norm=STAKE_NORM, cmap=plt.get_cmap(STAKE_CMAP)),
        ax=[ax1, ax2],
        fraction=0.03,
        pad=0.04,
    )
    cbar.set_label(r"Stakes $S(s)$")
    cbar.ax.minorticks_on()
    fig.text(
        0.5,
        0.125,
        "The green square is only the goal sampling zone; default success requires entering the sampled-goal tolerance circle.",
        ha="center",
        va="center",
        fontsize=9,
        color="#1b5e20",
    )
    add_global_legend(fig)
    return fig


def plot_path_for_algo(algo: str) -> str:
    if algo == "cddp":
        return PLOT_PATH
    return os.path.join(RESULTS_ROOT, algo, f"{algo}_trajectories.png")


def _algo_label(algo: str) -> str:
    return {
        "cddp": "CDDP",
        "ppo": "PPO",
        "strict": "Strict CMDP",
        "lagrangian": "PPO-Lagrangian",
        "cpo": "TRPO-Lagrangian",
    }[algo]


def plot_best(algo: str = "cddp") -> None:
    algo = algo.lower()
    if algo != "cddp" and algo not in BASELINE_ALGOS:
        raise ValueError(f"Unsupported algo '{algo}'. Expected cddp or one of {sorted(BASELINE_ALGOS)}.")

    cddp = _train_module()
    agent, env = load_best_agent(algo)
    tight_steps, long_steps = cddp.eval_clock_config()
    label = _algo_label(algo)

    tight_rollouts: list[tuple[list[tuple[float, float]], list[float], dict[str, object]]] = []
    long_rollouts: list[tuple[list[tuple[float, float]], list[float], dict[str, object]]] = []
    for i in range(10):
        seed = cddp.EVAL_SEED + i
        start_state, goal = make_eval_context(seed)
        context_id = f"seed-{seed}"
        tight_rollouts.append(
            rollout_episode(
                agent,
                env,
                start_state,
                goal,
                clock_max_steps=tight_steps,
                agent_profile=("Tight Clock Agent" if algo == "cddp" else f"{label} Tight Clock Agent"),
                seed=seed,
                context_id=context_id,
                algo=algo,
            )
        )
        long_rollouts.append(
            rollout_episode(
                agent,
                env,
                start_state,
                goal,
                clock_max_steps=long_steps,
                agent_profile=("Long Clock Agent" if algo == "cddp" else f"{label} Long Clock Agent"),
                seed=seed,
                context_id=context_id,
                algo=algo,
            )
        )

    fig = render_evaluation_figure(
        env,
        tight_rollouts,
        long_rollouts,
        tight_steps,
        long_steps,
    )
    output_plot_path = plot_path_for_algo(algo)
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    fig.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if algo == "cddp":
        cddp.save_evaluation_logs([record for _traj, _stakes, record in tight_rollouts + long_rollouts])
        json_path = cddp.EVAL_LOG_JSON_PATH
        csv_path = cddp.EVAL_LOG_CSV_PATH
    else:
        baseline = _baseline_module()
        json_path, csv_path = baseline.save_evaluation_logs(
            algo,
            [record for _traj, _stakes, record in tight_rollouts + long_rollouts],
        )
    print(f"Plot saved to {output_plot_path}")
    print(f"Evaluation logs saved to {json_path} and {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CDDP or PPO-family baseline trajectories.")
    parser.add_argument("--algo", choices=["cddp", "ppo", "strict", "lagrangian", "cpo"], default="cddp")
    parser.add_argument("--run-name", default=None, help="Optional CDDP ablation subdirectory under sandbox/results/cddp.")
    parser.add_argument("--goal-tol", type=float, default=None)
    parser.add_argument("--terminate-on-goal-zone", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cddp = _train_module()
    if args.run_name:
        cddp.configure_results_dir(args.run_name)
    if args.goal_tol is not None:
        cddp.GOAL_TOL = float(args.goal_tol)
    if args.terminate_on_goal_zone:
        cddp.TERMINATE_ON_GOAL_ZONE = True
    plot_best(args.algo)
