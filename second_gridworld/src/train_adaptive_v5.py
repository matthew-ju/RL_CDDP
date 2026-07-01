"""
train_adaptive_v5.py
====================
Adaptive CDDP v5 — four targeted fixes for the tight-clock dead zone observed in v4:

  1. Stakes as actor input (dim 6 = [x,y,vx,vy,clock,stakes]).  The actor now
     receives stakes directly so the gradient signal for "output aggressive
     actions when stakes is high" flows without being entangled with the clock
     inference.  Critics (Z_free, Z_deon) still use the plain 5-dim state.

  2. Risk-aligned reward bonus.  A tiny per-step bonus
       +RISK_ALIGN_ALPHA * stakes * obstacle_proximity
     is added to utilitarian episodes.  This breaks the gradient dead zone by
     giving the actor something to optimise for outside the CBF safe set when
     stakes are high, without requiring the critic to have already seen those
     states succeed.

  3. Shortened pre-CDDP phase (PRETRAIN_EPISODES = 2000, was 10000).  The v4
     log showed mean_P_cat=1.000 throughout all 13k pretraining/seed episodes —
     Z_deon learned everything is catastrophic before CDDP even started.
     Cutting pretrain to 2k starts the bifurcated (strict vs relaxed) data
     collection much earlier so Z_deon can calibrate.

  4. Bootstrap aggressive-policy episodes (BOOTSTRAP_CDDP_EPISODES = 2000).
     During the first N CDDP episodes, every third episode uses a naive
     straight-to-goal policy that ignores the CBF.  These transitions are
     stored in the tight_util_buffer so Z_free has successful direct-path
     trajectories to calibrate on before the actor can discover them on its own.

Same OOM-resilient structure as v4: periodic snapshots, --resume auto, MPS
cache flush each episode, GC every 500 episodes.

Same optimisation tricks as v4: TD3-style twin critics, vectorised envs,
split tight/loose replay buffers, compression timer curriculum, OU noise,
goal-seeking blend at high stakes.

Run:
    python3 src/train_adaptive_v5.py --run-name adaptive_v5 [OPTIONS]
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import random
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/private", "tmp", "cddp_rl_matplotlib"))

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

from src.analytical_sandbox import DifferentiablePointMassEnv
from src.helper_plotter import plot_best


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NUM_ENVS = 128

# Environment
PLANE_MIN = 0.0
PLANE_MAX = 10.0
START_LOW = torch.tensor([0.0, 0.0], dtype=torch.float32)
START_HIGH = torch.tensor([1.0, 1.0], dtype=torch.float32)
GOAL_LOW = torch.tensor([8.0, 8.0], dtype=torch.float32)
GOAL_HIGH = torch.tensor([10.0, 10.0], dtype=torch.float32)
GOAL = torch.tensor([9.0, 9.0], dtype=torch.float32)
DT = 0.02
V_MAX_ENV = 4.0
GOAL_TOL = 0.4
TERMINATE_ON_GOAL_ZONE = False
TIGHT_CLOCK_STEPS = 420
LONG_CLOCK_STEPS = 2500

OBS_C = torch.tensor(
    [
        [5.0, 5.0],
        [1.5, 5.0],
        [8.0, 2.0],
        [6.5, 8.5],
        [4.0, 2.0],
    ],
    dtype=torch.float32,
)
OBS_R = torch.tensor(
    [
        math.sqrt(0.5),
        math.sqrt(0.2),
        math.sqrt(0.2),
        math.sqrt(0.2),
        math.sqrt(0.2),
    ],
    dtype=torch.float32,
)
OBS_OUTER_R = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
OBS_DELTA_R = (OBS_OUTER_R - OBS_R) / 3.0
LAYER_1_R = OBS_R + OBS_DELTA_R
LAYER_2_R = OBS_R + 2.0 * OBS_DELTA_R
LAYER_3_R = OBS_OUTER_R
CORE_PENALTY = -2.50
LAYER_1_PENALTY = -1.50
LAYER_2_PENALTY = -0.50
LAYER_3_PENALTY = -0.05
OBSTACLE_OSCILLATION_AMPLITUDE = 0.0
OBSTACLE_OSCILLATION_PERIOD = 250.0
OBSTACLE_OSCILLATION_OMEGA = 2.0 * math.pi / OBSTACLE_OSCILLATION_PERIOD

# Policy bounds
U_MAX = 2.0

# HOCBF / CDDP
GAMMA_CBF = 0.7
TAU_CDDP = 0.3
ALPHA_CDDP = 10.0
P_REF = 0.35
ALLOW_HARD_CBF_BYPASS = False

# Goal-seeking exploration at high stakes
GOAL_SEEK_BLEND = 0.15
GOAL_SEEK_STAKES_THRESHOLD = 0.7

# Goal-capture shaping
GOAL_CAPTURE_SHAPING = True
GOAL_DISTANCE_PENALTY = 0.02
NEAR_GOAL_RADIUS = 1.0
NEAR_GOAL_BONUS_SCALE = 0.5
ACTION_PENALTY = 0.0

# ------------------------------------------------------------------ v5 new --
# Actor now receives stakes as a 6th input dimension.
ACTOR_STATE_DIM = 6   # [x, y, vx, vy, clock, stakes]

# Risk-aligned reward: small per-step bonus for being near/inside an obstacle
# outer radius when stakes are high.  This creates "propulsion toward risk"
# that the actor can follow before Z_free has calibrated on those states.
# Formula: +RISK_ALIGN_ALPHA * stakes * max_obstacle_penetration_depth
RISK_ALIGN_ALPHA = 0.05

# Bootstrap aggressive policy: run a naive goal-seeking (no CBF) policy for
# the first N CDDP episodes so the tight_util_buffer gets successful
# direct-path trajectories before the actor can discover them on its own.
BOOTSTRAP_CDDP_EPISODES = 2000

# Shortened pre-CDDP phase: was 10000 in v4.  Starting CDDP at ep ~5000
# (3k seed + 2k pretrain) prevents Z_deon from training to P_cat=1 on 13k
# monotone-failure episodes before seeing any relaxed-compliance data.
PRETRAIN_EPISODES_DEFAULT = 2000
# --------------------------------------------------------------------------

# Training
N_EPISODES = 75_000
N_DEONTIC_SEED_EPISODES = 3_000
MIN_EPISODE_STEPS = TIGHT_CLOCK_STEPS
MAX_EPISODE_STEPS = LONG_CLOCK_STEPS
BATCH_SIZE = 512
REPLAY_CAP = 500_000
TIGHT_CLOCK_THRESHOLD = 800
LR_ACTOR = 3e-4
LR_Z_FREE = 1e-3
LR_Z_DEON = 1e-3
GAMMA_RL = 0.99
TAU_SOFT = 0.005
WARMUP_STEPS = 1_000
TRAIN_EVERY = 12
GRADIENT_STEPS = 3
TIMER_CURRICULUM = "compression"

# OU exploration
OU_THETA = 0.15
OU_SIGMA = 0.35
OU_SIGMA_DECAY = 0.99995
OU_SIGMA_MIN = 0.1

LOG_INTERVAL = 100

# Network architecture
MODEL_HIDDEN_DIM = 256
MODEL_HIDDEN_LAYERS = 3
DEON_HIDDEN_DIM    = 128
DEON_HIDDEN_LAYERS = 2

# Artifacts
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
RESULTS_DIR = os.path.join(RESULTS_ROOT, "cddp")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
TRAJECTORY_PLOT_PATH = os.path.join(RESULTS_DIR, "cddp_trajectories.png")
EVAL_LOG_JSON_PATH = os.path.join(RESULTS_DIR, "cddp_eval_log.json")
EVAL_LOG_CSV_PATH = os.path.join(RESULTS_DIR, "cddp_eval_log.csv")
EVAL_SEED = 42
EVAL_START_STATE = [0.5, 0.5, 0.0, 0.0, 1.0]


def _safe_run_name(run_name: str | None) -> str | None:
    if run_name is None:
        return None
    clean = run_name.strip()
    if not clean:
        return None
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in clean)


def configure_results_dir(run_name: str | None = None) -> None:
    global RESULTS_DIR, CHECKPOINT_DIR, TRAJECTORY_PLOT_PATH, EVAL_LOG_JSON_PATH, EVAL_LOG_CSV_PATH
    clean = _safe_run_name(run_name)
    RESULTS_DIR = os.path.join(RESULTS_ROOT, "cddp") if clean is None else os.path.join(RESULTS_ROOT, "cddp", clean)
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
    TRAJECTORY_PLOT_PATH = os.path.join(RESULTS_DIR, "cddp_trajectories.png")
    EVAL_LOG_JSON_PATH = os.path.join(RESULTS_DIR, "cddp_eval_log.json")
    EVAL_LOG_CSV_PATH = os.path.join(RESULTS_DIR, "cddp_eval_log.csv")


def goal_zone_mask(pos: torch.Tensor) -> torch.Tensor:
    low = GOAL_LOW.to(device=pos.device, dtype=pos.dtype)
    high = GOAL_HIGH.to(device=pos.device, dtype=pos.dtype)
    return ((pos >= low) & (pos <= high)).all(dim=-1)


def exact_goal_mask(pos: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    goal_d = goal.to(device=pos.device, dtype=pos.dtype)
    if goal_d.ndim == 1:
        goal_d = goal_d.unsqueeze(0).expand(pos.shape[0], -1)
    return torch.norm(pos - goal_d, dim=-1) < GOAL_TOL


def success_mask(
    pos: torch.Tensor,
    goal: torch.Tensor,
    exact_reached: torch.Tensor | None = None,
) -> torch.Tensor:
    if TERMINATE_ON_GOAL_ZONE:
        return goal_zone_mask(pos)
    if exact_reached is not None:
        return exact_reached.to(device=pos.device, dtype=torch.bool)
    return exact_goal_mask(pos, goal)


def reward_config() -> dict[str, object]:
    return {
        "goal_capture_shaping": bool(GOAL_CAPTURE_SHAPING),
        "goal_distance_penalty": float(GOAL_DISTANCE_PENALTY),
        "near_goal_radius": float(NEAR_GOAL_RADIUS),
        "near_goal_bonus_scale": float(NEAR_GOAL_BONUS_SCALE),
        "action_penalty": float(ACTION_PENALTY),
        "gamma_rl": float(GAMMA_RL),
        "terminal_success_bonus": 500.0,
        "step_penalty": "escalating: -0.2*(1+0.5*t) where t=time_fraction",
        "early_finish_bonus_max": 150.0,
        "timeout_penalty_base": "-250 - 20*dist_to_goal",
        "risk_align_alpha": float(RISK_ALIGN_ALPHA),
    }


def task_config() -> dict[str, object]:
    return {
        "goal_low": [float(v) for v in GOAL_LOW.tolist()],
        "goal_high": [float(v) for v in GOAL_HIGH.tolist()],
        "goal_tolerance": float(GOAL_TOL),
        "terminate_on_goal_zone": bool(TERMINATE_ON_GOAL_ZONE),
        "timer_curriculum": TIMER_CURRICULUM,
        "allow_hard_cbf_bypass": bool(ALLOW_HARD_CBF_BYPASS),
        "dt": float(DT),
        "v_max": float(V_MAX_ENV),
        "obstacle_oscillation_amplitude": float(OBSTACLE_OSCILLATION_AMPLITUDE),
        "obstacle_oscillation_omega": float(OBSTACLE_OSCILLATION_OMEGA),
        "actor_state_dim": ACTOR_STATE_DIM,
        "reward": reward_config(),
    }


def compute_goal_capture_diagnostics(
    trajectory: list[tuple[float, float]],
    goal_xy: tuple[float, float],
    timer: int,
    success: bool,
    total_violations: int,
    stakes_by_step: list[float] | None = None,
) -> dict[str, object]:
    stakes_by_step = stakes_by_step or []
    total_steps = max(0, len(trajectory) - 1)
    if not trajectory:
        return {
            "timer": int(timer),
            "success": bool(success),
            "total_steps": 0,
            "total_violations": int(total_violations),
        }

    gx, gy = goal_xy
    dists = [math.hypot(x - gx, y - gy) for x, y in trajectory]
    path_length = sum(
        math.hypot(trajectory[i][0] - trajectory[i - 1][0], trajectory[i][1] - trajectory[i - 1][1])
        for i in range(1, len(trajectory))
    )
    straight_line_distance = math.hypot(trajectory[0][0] - gx, trajectory[0][1] - gy)

    goal_low = [float(v) for v in GOAL_LOW.tolist()]
    goal_high = [float(v) for v in GOAL_HIGH.tolist()]
    first_zone = None
    first_tol = None
    steps_dist_lt_0_5 = 0
    steps_dist_lt_1_0 = 0
    for step_idx, ((x, y), dist) in enumerate(zip(trajectory[1:], dists[1:]), start=1):
        if first_zone is None and goal_low[0] <= x <= goal_high[0] and goal_low[1] <= y <= goal_high[1]:
            first_zone = step_idx
        if first_tol is None and dist < GOAL_TOL:
            first_tol = step_idx
        if dist < 0.5:
            steps_dist_lt_0_5 += 1
        if dist < 1.0:
            steps_dist_lt_1_0 += 1

    if stakes_by_step:
        avg_stakes = sum(stakes_by_step) / len(stakes_by_step)
        max_stakes = max(stakes_by_step)
        frac_stakes_gt_0_8 = sum(1 for s in stakes_by_step if s > 0.8) / len(stakes_by_step)
    else:
        avg_stakes = 0.0
        max_stakes = 0.0
        frac_stakes_gt_0_8 = 0.0

    return {
        "timer": int(timer),
        "success": bool(success),
        "total_steps": int(total_steps),
        "total_violations": int(total_violations),
        "final_distance_to_goal": float(dists[-1]),
        "min_distance_to_goal": float(min(dists)),
        "first_step_inside_goal_zone": first_zone,
        "first_step_inside_goal_tol": first_tol,
        "steps_after_first_entering_goal_zone": None if first_zone is None else int(total_steps - first_zone),
        "steps_dist_lt_0_5": int(steps_dist_lt_0_5),
        "steps_dist_lt_1_0": int(steps_dist_lt_1_0),
        "path_length": float(path_length),
        "straight_line_distance": float(straight_line_distance),
        "path_efficiency": float(straight_line_distance / max(path_length, 1e-8)),
        "avg_stakes": float(avg_stakes),
        "max_stakes": float(max_stakes),
        "frac_stakes_gt_0_8": float(frac_stakes_gt_0_8),
    }


# ---------------------------------------------------------------------------
# Replay Buffers
# ---------------------------------------------------------------------------


class UtilReplayBuffer:
    """Replay for Z_free and Actor. Contains only dynamic-boundary rollouts.

    States stored as 5-dim (raw env state).  Stakes are recomputed from
    Z_deon at training time so the actor can be given the *current* stakes
    estimate rather than the stale one from collection time.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.s = torch.empty((capacity, 5), dtype=torch.float32, device=DEVICE)
        self.a = torch.empty((capacity, 2), dtype=torch.float32, device=DEVICE)
        self.r = torch.empty((capacity, 1), dtype=torch.float32, device=DEVICE)
        self.s_next = torch.empty((capacity, 5), dtype=torch.float32, device=DEVICE)
        self.done = torch.empty((capacity, 1), dtype=torch.float32, device=DEVICE)

    def _write_indices(self, n: int) -> torch.Tensor:
        if n >= self.capacity:
            self.ptr = 0
            self.size = self.capacity
            return torch.arange(self.capacity, device=DEVICE)

        idx = (torch.arange(n, device=DEVICE) + self.ptr) % self.capacity
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        return idx

    def push_batch(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        s_next: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        n = int(s.shape[0])
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=DEVICE)
        if n > self.capacity:
            s = s[-self.capacity:]
            a = a[-self.capacity:]
            r = r[-self.capacity:]
            s_next = s_next[-self.capacity:]
            done = done[-self.capacity:]
            n = self.capacity

        idx = self._write_indices(n)
        self.s[idx] = s.detach()
        self.a[idx] = a.detach()
        self.r[idx] = r.detach().reshape(-1, 1)
        self.s_next[idx] = s_next.detach()
        self.done[idx] = done.detach().to(dtype=torch.float32).reshape(-1, 1)
        return idx

    def sample(self, n: int) -> tuple[torch.Tensor, ...]:
        idx = torch.randint(self.size, (n,), device=DEVICE)
        return self.s[idx], self.a[idx], self.r[idx], self.s_next[idx], self.done[idx]

    def __len__(self) -> int:
        return self.size


class DeonticReplayBuffer:
    """Replay for Z_deon. Contains only strict S=0 rollouts."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.s = torch.empty((capacity, 5), dtype=torch.float32, device=DEVICE)
        self.a = torch.empty((capacity, 2), dtype=torch.float32, device=DEVICE)
        self.r = torch.empty((capacity, 1), dtype=torch.float32, device=DEVICE)
        self.s_next = torch.empty((capacity, 5), dtype=torch.float32, device=DEVICE)
        self.done = torch.empty((capacity, 1), dtype=torch.float32, device=DEVICE)
        self.y = torch.empty((capacity, 1), dtype=torch.float32, device=DEVICE)
        self.labeled = torch.zeros(capacity, dtype=torch.bool, device=DEVICE)
        self.labeled_count = 0

    def _write_indices(self, n: int) -> torch.Tensor:
        if n >= self.capacity:
            self.ptr = 0
            self.size = self.capacity
            return torch.arange(self.capacity, device=DEVICE)

        idx = (torch.arange(n, device=DEVICE) + self.ptr) % self.capacity
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        return idx

    def push_batch(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        s_next: torch.Tensor,
        done: torch.Tensor,
        y_cat: torch.Tensor,
        labeled: bool = True,
    ) -> torch.Tensor:
        n = int(s.shape[0])
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=DEVICE)
        if n > self.capacity:
            s = s[-self.capacity:]
            a = a[-self.capacity:]
            r = r[-self.capacity:]
            s_next = s_next[-self.capacity:]
            done = done[-self.capacity:]
            y_cat = y_cat[-self.capacity:]
            n = self.capacity

        idx = self._write_indices(n)
        previously_labeled = int(self.labeled[idx].sum().item())
        self.s[idx] = s.detach()
        self.a[idx] = a.detach()
        self.r[idx] = r.detach().reshape(-1, 1)
        self.s_next[idx] = s_next.detach()
        self.done[idx] = done.detach().to(dtype=torch.float32).reshape(-1, 1)
        self.y[idx] = y_cat.detach().to(dtype=torch.float32).reshape(-1, 1)
        self.labeled[idx] = labeled
        self.labeled_count += (n if labeled else 0) - previously_labeled
        self.labeled_count = max(0, min(self.labeled_count, self.size))
        return idx

    def label_indices(self, indices: torch.Tensor, y_cat: bool) -> None:
        if indices.numel() == 0:
            return
        indices = indices[indices >= 0].unique()
        if indices.numel() == 0:
            return
        newly_labeled = int((~self.labeled[indices]).sum().item())
        self.y[indices] = float(y_cat)
        self.labeled[indices] = True
        self.labeled_count = min(self.size, self.labeled_count + newly_labeled)

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        labeled_idx = torch.where(self.labeled[: self.size])[0]
        idx = labeled_idx[torch.randint(labeled_idx.shape[0], (n,), device=DEVICE)]
        return self.s[idx], self.y[idx]

    def __len__(self) -> int:
        return self.labeled_count


# ---------------------------------------------------------------------------
# Neural Networks
# ---------------------------------------------------------------------------

def _mlp(
    in_dim: int,
    out_dim: int,
    hidden: int = MODEL_HIDDEN_DIM,
    hidden_layers: int = MODEL_HIDDEN_LAYERS,
) -> nn.Sequential:
    dims = [in_dim] + [hidden] * hidden_layers + [out_dim]
    mods: list[nn.Module] = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


class Actor(nn.Module):
    """Nominal policy pi(s, stakes) — stakes is an explicit 6th input.

    The actor now receives [x, y, vx, vy, clock, stakes] so the gradient
    signal for "output more aggressive raw actions when stakes is high" flows
    directly, without needing to entangle stakes inference with the action
    policy via the clock signal alone.
    """

    def __init__(
        self,
        state_dim: int = ACTOR_STATE_DIM,   # 6: includes stakes
        action_dim: int = 2,
        hidden: int = MODEL_HIDDEN_DIM,
        hidden_layers: int = MODEL_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()
        self.net = _mlp(state_dim, action_dim, hidden=hidden, hidden_layers=hidden_layers)

    def forward(self, s_aug: torch.Tensor) -> torch.Tensor:
        """s_aug: [..., 6] where last dim is [state(5), stakes(1)]."""
        return U_MAX * torch.tanh(self.net(s_aug))


class UtilitarianCritic(nn.Module):
    """Z_free(s, a): standard task-return critic.  Still uses 5-dim state."""

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        hidden: int = MODEL_HIDDEN_DIM,
        hidden_layers: int = MODEL_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()
        self.net = _mlp(state_dim + action_dim, 1, hidden=hidden, hidden_layers=hidden_layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


class DeonticCritic(nn.Module):
    """Z_deon(s): P_cat(s) under strict S=0 compliance.  Uses 5-dim state."""

    def __init__(
        self,
        state_dim: int = 5,
        hidden: int = DEON_HIDDEN_DIM,
        hidden_layers: int = DEON_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()
        self.net = _mlp(state_dim, 1, hidden=hidden, hidden_layers=hidden_layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class OUNoise:
    """Ornstein-Uhlenbeck noise for temporally coherent exploration."""

    def __init__(self, action_dim: int, sigma: float = OU_SIGMA) -> None:
        self.action_dim = action_dim
        self.sigma = sigma
        self.state = torch.zeros(NUM_ENVS, action_dim, device=DEVICE)

    def reset(self) -> None:
        self.state.zero_()

    def reset_idx(self, indices: torch.Tensor) -> None:
        if indices.numel() > 0:
            self.state[indices] = 0.0

    def sample(self) -> torch.Tensor:
        dx = OU_THETA * (-self.state) + self.sigma * torch.randn_like(self.state)
        self.state = self.state + dx
        return self.state

    def decay(self) -> None:
        self.sigma = max(OU_SIGMA_MIN, self.sigma * OU_SIGMA_DECAY)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ---------------------------------------------------------------------------
# Stakes augmentation helpers
# ---------------------------------------------------------------------------

def aug_s(s: torch.Tensor, stakes: torch.Tensor) -> torch.Tensor:
    """Concatenate stakes to state to produce actor input.

    Args:
        s:      [N, 5] raw env state
        stakes: [N]    stakes signal in [0, 1]

    Returns:
        [N, 6] augmented state for actor input
    """
    return torch.cat([s, stakes.view(-1, 1)], dim=-1)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CDDPAgent:
    """Actor plus Z_free and Z_deon, sharing the analytical CBF projection.

    v5 change: Actor takes augmented 6-dim input [state, stakes].
    """

    def __init__(self, env: DifferentiablePointMassEnv) -> None:
        self.env = env

        self.actor = Actor().to(DEVICE)
        self.actor_target = Actor().to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.requires_grad_(False)

        self.z_free = UtilitarianCritic().to(DEVICE)
        self.z_free_target = UtilitarianCritic().to(DEVICE)
        self.z_free_target.load_state_dict(self.z_free.state_dict())
        self.z_free_target.requires_grad_(False)

        self.z_free_2 = UtilitarianCritic().to(DEVICE)
        self.z_free_2_target = UtilitarianCritic().to(DEVICE)
        self.z_free_2_target.load_state_dict(self.z_free_2.state_dict())
        self.z_free_2_target.requires_grad_(False)

        self.z_deon = DeonticCritic().to(DEVICE)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_z_free = torch.optim.Adam(self.z_free.parameters(), lr=LR_Z_FREE)
        self.opt_z_free_2 = torch.optim.Adam(self.z_free_2.parameters(), lr=LR_Z_FREE)
        self.opt_z_deon = torch.optim.Adam(self.z_deon.parameters(), lr=LR_Z_DEON)

    def stakes_from_deon(self, s: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """Compute S(s) = clip(P_cat(s) / P_ref, 0, 1)."""
        logits = self.z_deon(s)
        p_cat = torch.sigmoid(logits)
        if detach:
            p_cat = p_cat.detach()
        return torch.clamp(p_cat.squeeze(1) / P_REF, 0.0, 1.0)

    def safe_action(
        self,
        s: torch.Tensor,
        u_nom: torch.Tensor,
        stakes: torch.Tensor,
        gamma: float | None = None,
    ) -> torch.Tensor:
        if gamma is None:
            gamma = GAMMA_CBF
        u_safe = self.env.project_action_cbf(
            s,
            u_nom,
            stakes=stakes,
            gamma=gamma,
            tau=TAU_CDDP,
            alpha_prime=ALPHA_CDDP,
        )

        if ALLOW_HARD_CBF_BYPASS:
            bypass_mask = (stakes > 0.95).view(-1, 1)
            return torch.where(bypass_mask, u_nom, u_safe)
        return u_safe

    @torch.no_grad()
    def select_action(
        self,
        s: torch.Tensor,
        stakes: torch.Tensor,
        ou_noise: OUNoise | None = None,
        random_nominal: bool = False,
    ) -> torch.Tensor:
        """Return projected action for environment interaction.

        v5: actor receives augmented state [s, stakes].
        """
        s_b = s if s.ndim == 2 else s.unsqueeze(0)
        stk = stakes.view(-1) if s.ndim == 2 else stakes.view(-1)[:1]

        if random_nominal:
            u_nom = (torch.rand(s_b.shape[0], 2, device=DEVICE) * 2.0 - 1.0) * U_MAX
        else:
            s_aug = aug_s(s_b, stk)
            u_nom = self.actor(s_aug)
            if ou_noise is not None:
                u_nom = torch.clamp(u_nom + ou_noise.sample()[: s_b.shape[0]], -U_MAX, U_MAX)

        u_safe = self.safe_action(s_b, u_nom, stakes=stk.reshape(-1))
        return u_safe if s.ndim == 2 else u_safe.squeeze(0)

    def update_utilitarian(
        self,
        tight_buffer: UtilReplayBuffer,
        loose_buffer: UtilReplayBuffer,
        strict_stakes: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Train Z_free, Z_free_2, and Actor.

        v5: actor calls use aug_s() to prepend stakes to state.
        States in replay buffers are still 5-dim; stakes are recomputed
        from the *current* Z_deon so the actor gets fresh estimates.
        """
        half = BATCH_SIZE // 2
        tight_ok = len(tight_buffer) >= half
        loose_ok = len(loose_buffer) >= half

        if tight_ok and loose_ok:
            s1, a1, r1, s1_, d1 = tight_buffer.sample(half)
            s2, a2, r2, s2_, d2 = loose_buffer.sample(half)
            s    = torch.cat([s1, s2],  dim=0)
            a    = torch.cat([a1, a2],  dim=0)
            r    = torch.cat([r1, r2],  dim=0)
            s_   = torch.cat([s1_, s2_], dim=0)
            done = torch.cat([d1, d2],  dim=0)
        elif loose_ok and len(loose_buffer) >= BATCH_SIZE:
            s, a, r, s_, done = loose_buffer.sample(BATCH_SIZE)
        elif tight_ok and len(tight_buffer) >= BATCH_SIZE:
            s, a, r, s_, done = tight_buffer.sample(BATCH_SIZE)
        else:
            return None, None

        with torch.no_grad():
            # Compute stakes for next states
            if strict_stakes:
                stakes_next = torch.zeros(s_.shape[0], dtype=s_.dtype, device=s_.device)
            else:
                stakes_next = self.stakes_from_deon(s_, detach=True)

            # Actor target uses augmented next state
            s_next_aug = aug_s(s_, stakes_next)
            u_nom_next = self.actor_target(s_next_aug)
            noise = (torch.randn_like(u_nom_next) * 0.2).clamp(-0.5, 0.5)
            u_nom_next = torch.clamp(u_nom_next + noise, -U_MAX, U_MAX)
            u_safe_next = self.safe_action(s_, u_nom_next, stakes_next)
            q_tgt = r + GAMMA_RL * (1.0 - done) * torch.min(
                self.z_free_target(s_, u_safe_next),
                self.z_free_2_target(s_, u_safe_next),
            )

        q_pred = self.z_free(s, a)
        loss_z_free = F.mse_loss(q_pred, q_tgt)
        self.opt_z_free.zero_grad()
        loss_z_free.backward()
        nn.utils.clip_grad_norm_(self.z_free.parameters(), 1.0)
        self.opt_z_free.step()

        q_pred_2 = self.z_free_2(s, a)
        loss_z_free_2 = F.mse_loss(q_pred_2, q_tgt)
        self.opt_z_free_2.zero_grad()
        loss_z_free_2.backward()
        nn.utils.clip_grad_norm_(self.z_free_2.parameters(), 1.0)
        self.opt_z_free_2.step()

        # Actor update — uses augmented state with current stakes
        if strict_stakes:
            stakes_actor = torch.zeros(s.shape[0], dtype=s.dtype, device=s.device)
        else:
            stakes_actor = self.stakes_from_deon(s, detach=True)
        s_aug = aug_s(s, stakes_actor)
        u_nom_actor = self.actor(s_aug)
        u_safe_actor = self.safe_action(s, u_nom_actor, stakes_actor)
        loss_actor = -self.z_free(s, u_safe_actor).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        soft_update(self.actor_target, self.actor, TAU_SOFT)
        soft_update(self.z_free_target, self.z_free, TAU_SOFT)
        soft_update(self.z_free_2_target, self.z_free_2, TAU_SOFT)
        return loss_z_free.detach(), loss_actor.detach()

    def update_deontic(self, deontic_buffer: DeonticReplayBuffer) -> torch.Tensor:
        """Train Z_deon from strict S=0 data only, with class-balanced BCE."""
        s, y_cat = deontic_buffer.sample(BATCH_SIZE)
        logits = self.z_deon(s)
        n_pos = y_cat.sum().item()
        n_neg = (1.0 - y_cat).sum().item()
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor(n_neg / n_pos, device=DEVICE)
        else:
            pos_weight = torch.ones(1, device=DEVICE)
        loss_z_deon = F.binary_cross_entropy_with_logits(
            logits, y_cat, pos_weight=pos_weight
        )

        self.opt_z_deon.zero_grad()
        loss_z_deon.backward()
        nn.utils.clip_grad_norm_(self.z_deon.parameters(), 1.0)
        self.opt_z_deon.step()
        return loss_z_deon.detach()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_env() -> DifferentiablePointMassEnv:
    return DifferentiablePointMassEnv(
        goal=GOAL,
        obstacle_c=OBS_C,
        obstacle_r=OBS_R,
        dt=DT,
        goal_tol=GOAL_TOL,
        v_max=V_MAX_ENV,
        obstacle_amplitude=OBSTACLE_OSCILLATION_AMPLITUDE,
        obstacle_omega=OBSTACLE_OSCILLATION_OMEGA,
        device=DEVICE,
    )


def _sample_box(
    low: torch.Tensor,
    high: torch.Tensor,
    n: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if generator is None:
        unit = torch.rand(n, 2, device=device)
    else:
        unit = torch.rand(n, 2, generator=generator).to(device)
    low_d = low.to(device)
    high_d = high.to(device)
    return low_d + unit * (high_d - low_d)


def sample_episode_context(
    n: int = 1,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    p = _sample_box(START_LOW, START_HIGH, n, DEVICE, generator=generator)
    goals = _sample_box(GOAL_LOW, GOAL_HIGH, n, DEVICE, generator=generator)
    states = torch.cat(
        [p, torch.zeros(n, 2, device=DEVICE), torch.ones(n, 1, device=DEVICE)],
        dim=1,
    )
    return states, goals


def random_start(n: int = 1) -> torch.Tensor:
    states, _ = sample_episode_context(n)
    return states


def random_goal(n: int = 1) -> torch.Tensor:
    _, goals = sample_episode_context(n)
    return goals


def obstacle_centers_at(
    timestep: torch.Tensor | int | float | None = None,
    batch_size: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    device = device or DEVICE
    base = OBS_C.to(device=device, dtype=dtype)
    if batch_size is None:
        t = torch.as_tensor(0.0 if timestep is None else timestep, dtype=dtype, device=device)
        centers = base.clone()
        centers[:, 0] = centers[:, 0] + OBSTACLE_OSCILLATION_AMPLITUDE * torch.sin(
            OBSTACLE_OSCILLATION_OMEGA * t
        )
        return centers

    if timestep is None:
        t_batch = torch.zeros(batch_size, dtype=dtype, device=device)
    else:
        t_batch = torch.as_tensor(timestep, dtype=dtype, device=device)
        if t_batch.ndim == 0:
            t_batch = t_batch.expand(batch_size)
    centers = base.unsqueeze(0).expand(batch_size, -1, -1).clone()
    centers[:, :, 0] = centers[:, :, 0] + (
        OBSTACLE_OSCILLATION_AMPLITUDE
        * torch.sin(OBSTACLE_OSCILLATION_OMEGA * t_batch).unsqueeze(1)
    )
    return centers


def obstacle_layer_penalty(
    pos: torch.Tensor,
    timestep: torch.Tensor | int | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return strongest active layer penalty plus per-obstacle core violations."""
    centers = obstacle_centers_at(
        timestep=timestep,
        batch_size=pos.shape[0],
        device=pos.device,
        dtype=pos.dtype,
    )
    distances = torch.norm(pos.unsqueeze(1) - centers, dim=2)
    core_r = OBS_R.to(device=pos.device, dtype=pos.dtype).unsqueeze(0)
    layer_1_r = LAYER_1_R.to(device=pos.device, dtype=pos.dtype).unsqueeze(0)
    layer_2_r = LAYER_2_R.to(device=pos.device, dtype=pos.dtype).unsqueeze(0)
    layer_3_r = LAYER_3_R.to(device=pos.device, dtype=pos.dtype).unsqueeze(0)

    violations = distances < core_r
    penalties = torch.zeros_like(distances)
    penalties = torch.where(violations, torch.full_like(penalties, CORE_PENALTY), penalties)
    penalties = torch.where(
        (distances >= core_r) & (distances < layer_1_r),
        torch.full_like(penalties, LAYER_1_PENALTY),
        penalties,
    )
    penalties = torch.where(
        (distances >= layer_1_r) & (distances < layer_2_r),
        torch.full_like(penalties, LAYER_2_PENALTY),
        penalties,
    )
    penalties = torch.where(
        (distances >= layer_2_r) & (distances < layer_3_r),
        torch.full_like(penalties, LAYER_3_PENALTY),
        penalties,
    )
    return penalties.min(dim=1).values, violations, distances


def compute_obstacle_proximity(
    pos: torch.Tensor,
    timestep: torch.Tensor | int | float | None = None,
) -> torch.Tensor:
    """Max penetration depth into any obstacle outer-radius zone.

    Returns [N] tensor in [0, max_outer_r].  Zero when outside all outer radii,
    positive when inside at least one outer radius (proportional to depth).
    Used for the risk-aligned reward bonus.
    """
    centers = obstacle_centers_at(
        timestep=timestep,
        batch_size=pos.shape[0],
        device=pos.device,
        dtype=pos.dtype,
    )
    distances = torch.norm(pos.unsqueeze(1) - centers, dim=2)          # [N, 5]
    outer_r = OBS_OUTER_R.to(device=pos.device, dtype=pos.dtype).unsqueeze(0)  # [1, 5]
    penetration = (outer_r - distances).clamp_min(0.0)                 # [N, 5]
    return penetration.max(dim=1).values                               # [N]


def format_start_state(position: torch.Tensor, clock_steps: int) -> torch.Tensor:
    position = position.to(device=DEVICE, dtype=torch.float32).reshape(1, 2)
    return torch.cat(
        [
            position,
            torch.zeros(1, 2, device=DEVICE),
            torch.ones(1, 1, device=DEVICE),
        ],
        dim=1,
    )


def sample_episode_steps(n: int, curriculum: str | None = None, progress: float = 0.0) -> torch.Tensor:
    curriculum = TIMER_CURRICULUM if curriculum is None else curriculum
    if curriculum == "mixed":
        sample = torch.rand(n, device=DEVICE)
        tight = torch.randint(300, 600 + 1, (n,), device=DEVICE)
        medium = torch.randint(600, 1600 + 1, (n,), device=DEVICE)
        long = torch.randint(1600, 3000 + 1, (n,), device=DEVICE)
        return torch.where(sample < 0.30, tight, torch.where(sample < 0.70, medium, long))
    if curriculum == "long_only":
        return torch.randint(800, 2500 + 1, (n,), device=DEVICE)
    if curriculum == "compression":
        if progress < 0.20:
            lo = 2000
        elif progress < 0.50:
            frac = (progress - 0.20) / 0.30
            lo = int(2000 - frac * (2000 - 1200))
        elif progress < 0.80:
            frac = (progress - 0.50) / 0.30
            lo = int(1200 - frac * (1200 - 600))
        else:
            frac = (progress - 0.80) / 0.20
            lo = int(600 - frac * (600 - 420))
        lo = max(lo, 420)
        return torch.randint(lo, 2500 + 1, (n,), device=DEVICE)
    if curriculum != "bimodal":
        raise ValueError(f"Unsupported timer curriculum '{curriculum}'.")
    tight = torch.randint(300, 450 + 1, (n,), device=DEVICE)
    long = torch.randint(2000, 3000 + 1, (n,), device=DEVICE)
    return torch.where(torch.rand(n, device=DEVICE) < 0.5, tight, long)


def compute_reward(
    pos: torch.Tensor,
    prev_pos: torch.Tensor,
    goal: torch.Tensor,
    reached: torch.Tensor | bool,
    unsafe: torch.Tensor | bool,
    timeout: torch.Tensor | bool,
    timestep: torch.Tensor | int | float | None = None,
    max_episode_steps: torch.Tensor | int | float | None = None,
) -> torch.Tensor | float:
    # 1. Dense distance shaping (progress toward goal)
    goal = goal.to(pos.device)
    if goal.ndim == 1:
        goal = goal.unsqueeze(0).expand(pos.shape[0], -1)
    dist_now = (pos - goal).norm(dim=-1)
    dist_prev = (prev_pos - goal).norm(dim=-1)
    progress = dist_prev - dist_now

    if max_episode_steps is not None and timestep is not None:
        t = torch.as_tensor(timestep, dtype=pos.dtype, device=pos.device)
        m = torch.as_tensor(max_episode_steps, dtype=pos.dtype, device=pos.device)
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(pos.shape[0])
        if m.ndim == 0:
            m = m.unsqueeze(0).expand(pos.shape[0])
        time_fraction = (t / m.clamp(min=1.0)).clamp(0.0, 1.0)
        step_penalty = -0.2 * (1.0 + 0.5 * time_fraction)
        early_finish_bonus: torch.Tensor | None = 150.0 * (1.0 - time_fraction)
    else:
        step_penalty = -0.2
        early_finish_bonus = None

    reward = (progress * 40.0) + step_penalty

    # Velocity damping near goal
    speed_sq = ((pos - prev_pos) ** 2).sum(dim=-1) / (DT ** 2)
    near_goal = dist_now < 1.5
    reward = torch.where(near_goal, reward - 0.15 * speed_sq, reward)
    if GOAL_CAPTURE_SHAPING:
        reward = reward - (GOAL_DISTANCE_PENALTY * dist_now)
        near_goal = dist_now < NEAR_GOAL_RADIUS
        near_goal_bonus = NEAR_GOAL_BONUS_SCALE * (NEAR_GOAL_RADIUS - dist_now)
        reward = torch.where(near_goal, reward + near_goal_bonus, reward)

    # 3. Terminal rewards
    reached_t = torch.as_tensor(reached, device=pos.device, dtype=torch.bool)
    timeout_t = torch.as_tensor(timeout, device=pos.device, dtype=torch.bool)
    if early_finish_bonus is not None:
        reward = torch.where(reached_t, reward + 500.0 + early_finish_bonus, reward)
    else:
        reward = torch.where(reached_t, reward + 500.0, reward)
    reward = torch.where(timeout_t, reward - (250.0 + (dist_now * 20.0)), reward)

    if reward.ndim == 0:
        return float(reward.item())
    return reward


def get_evaluation_logger() -> logging.Logger:
    logger = logging.getLogger("cddp_evaluation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def save_evaluation_logs(records: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(EVAL_LOG_JSON_PATH), exist_ok=True)
    logger = get_evaluation_logger()

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    json_handler = logging.FileHandler(EVAL_LOG_JSON_PATH, mode="w", encoding="utf-8")
    json_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(json_handler)
    logger.info(json.dumps(records, indent=2))
    logger.removeHandler(json_handler)
    json_handler.close()
    logger.addHandler(logging.NullHandler())

    csv_fields = [
        "seed", "context_id", "agent_profile", "timer", "max_steps",
        "start_x", "start_y", "goal_x", "goal_y",
        "success", "total_steps", "total_violations", "total_reward",
        "final_distance_to_goal", "min_distance_to_goal",
        "first_step_inside_goal_zone", "first_step_inside_goal_tol",
        "steps_after_first_entering_goal_zone",
        "steps_dist_lt_0_5", "steps_dist_lt_1_0",
        "path_length", "straight_line_distance", "path_efficiency",
        "avg_stakes", "max_stakes", "frac_stakes_gt_0_8",
        "constraint_violations",
    ]
    with open(EVAL_LOG_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in csv_fields}
            row["constraint_violations"] = json.dumps(record.get("constraint_violations", []))
            writer.writerow(row)

    logger.info("Wrote %d evaluation rows to %s", len(records), EVAL_LOG_CSV_PATH)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoints(agent: CDDPAgent) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(agent.actor.state_dict(), os.path.join(CHECKPOINT_DIR, "cddp_actor.pt"))
    torch.save(agent.z_free.state_dict(), os.path.join(CHECKPOINT_DIR, "cddp_z_free.pt"))
    torch.save(agent.z_free_2.state_dict(), os.path.join(CHECKPOINT_DIR, "cddp_z_free_2.pt"))
    torch.save(agent.z_deon.state_dict(), os.path.join(CHECKPOINT_DIR, "cddp_z_deon.pt"))


def save_best_checkpoints(agent: CDDPAgent) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(agent.actor.state_dict(), os.path.join(CHECKPOINT_DIR, "cddp_actor_best.pt"))
    torch.save(agent.z_free.state_dict(), os.path.join(CHECKPOINT_DIR, "cddp_z_free_best.pt"))
    torch.save(agent.z_free_2.state_dict(), os.path.join(CHECKPOINT_DIR, "cddp_z_free_2_best.pt"))
    torch.save(agent.z_deon.state_dict(), os.path.join(CHECKPOINT_DIR, "cddp_z_deon_best.pt"))


def load_best_checkpoints(agent: CDDPAgent) -> bool:
    actor_path = os.path.join(CHECKPOINT_DIR, "cddp_actor_best.pt")
    z_free_path = os.path.join(CHECKPOINT_DIR, "cddp_z_free_best.pt")
    z_free_2_path = os.path.join(CHECKPOINT_DIR, "cddp_z_free_2_best.pt")
    z_deon_path = os.path.join(CHECKPOINT_DIR, "cddp_z_deon_best.pt")
    if not all(os.path.exists(path) for path in (actor_path, z_free_path, z_free_2_path, z_deon_path)):
        return False

    try:
        agent.actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
        agent.z_free.load_state_dict(torch.load(z_free_path, map_location=DEVICE))
        agent.z_free_2.load_state_dict(torch.load(z_free_2_path, map_location=DEVICE))
        agent.z_deon.load_state_dict(torch.load(z_deon_path, map_location=DEVICE))
    except RuntimeError as exc:
        print(f"  Best checkpoint incompatible with current architecture: {exc}")
        return False
    return True


def save_periodic_checkpoint(
    agent: CDDPAgent,
    episode: int,
    best_score: float,
    best_util_success: float,
    best_util_p_excess: float,
    best_util_avg_viol: float,
    best_avg_steps_after_goal_zone: float,
    training_state: dict | None = None,
) -> None:
    snap_dir = os.path.join(CHECKPOINT_DIR, f"ep{episode:07d}")
    os.makedirs(snap_dir, exist_ok=True)

    torch.save(agent.actor.state_dict(), os.path.join(snap_dir, "cddp_actor.pt"))
    torch.save(agent.z_free.state_dict(), os.path.join(snap_dir, "cddp_z_free.pt"))
    torch.save(agent.z_free_2.state_dict(), os.path.join(snap_dir, "cddp_z_free_2.pt"))
    torch.save(agent.z_deon.state_dict(), os.path.join(snap_dir, "cddp_z_deon.pt"))

    torch.save(agent.actor_target.state_dict(), os.path.join(snap_dir, "cddp_actor_target.pt"))
    torch.save(agent.z_free_target.state_dict(), os.path.join(snap_dir, "cddp_z_free_target.pt"))
    torch.save(agent.z_free_2_target.state_dict(), os.path.join(snap_dir, "cddp_z_free_2_target.pt"))

    torch.save(agent.opt_actor.state_dict(), os.path.join(snap_dir, "opt_actor.pt"))
    torch.save(agent.opt_z_free.state_dict(), os.path.join(snap_dir, "opt_z_free.pt"))
    torch.save(agent.opt_z_free_2.state_dict(), os.path.join(snap_dir, "opt_z_free_2.pt"))
    torch.save(agent.opt_z_deon.state_dict(), os.path.join(snap_dir, "opt_z_deon.pt"))

    metrics = {
        "episode": episode,
        "best_score": best_score,
        "best_util_success": best_util_success,
        "best_util_p_excess": best_util_p_excess,
        "best_util_avg_viol": best_util_avg_viol,
        "best_avg_steps_after_goal_zone": best_avg_steps_after_goal_zone,
    }
    with open(os.path.join(snap_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if training_state is not None:
        with open(os.path.join(snap_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)

    print(f"  [checkpoint] Saved full resume snapshot at episode {episode} → {snap_dir}")
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_checkpoint_for_resume(agent: CDDPAgent, snap_dir: str) -> dict:
    required = ["cddp_actor.pt", "cddp_z_free.pt", "cddp_z_free_2.pt", "cddp_z_deon.pt", "metrics.json"]
    for fname in required:
        path = os.path.join(snap_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Resume checkpoint missing file: {path}")

    agent.actor.load_state_dict(torch.load(os.path.join(snap_dir, "cddp_actor.pt"), map_location=DEVICE))
    agent.z_free.load_state_dict(torch.load(os.path.join(snap_dir, "cddp_z_free.pt"), map_location=DEVICE))
    agent.z_free_2.load_state_dict(torch.load(os.path.join(snap_dir, "cddp_z_free_2.pt"), map_location=DEVICE))
    agent.z_deon.load_state_dict(torch.load(os.path.join(snap_dir, "cddp_z_deon.pt"), map_location=DEVICE))

    for tgt_file, tgt_net, src_net in [
        ("cddp_actor_target.pt", agent.actor_target, agent.actor),
        ("cddp_z_free_target.pt", agent.z_free_target, agent.z_free),
        ("cddp_z_free_2_target.pt", agent.z_free_2_target, agent.z_free_2),
    ]:
        tgt_path = os.path.join(snap_dir, tgt_file)
        if os.path.exists(tgt_path):
            tgt_net.load_state_dict(torch.load(tgt_path, map_location=DEVICE))
        else:
            tgt_net.load_state_dict(src_net.state_dict())

    for opt_file, optimizer in [
        ("opt_actor.pt",    agent.opt_actor),
        ("opt_z_free.pt",   agent.opt_z_free),
        ("opt_z_free_2.pt", agent.opt_z_free_2),
        ("opt_z_deon.pt",   agent.opt_z_deon),
    ]:
        opt_path = os.path.join(snap_dir, opt_file)
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=DEVICE))
        else:
            print(f"  [resume] WARNING: {opt_file} not found — optimizer state reset.")

    with open(os.path.join(snap_dir, "metrics.json")) as f:
        metrics = json.load(f)

    training_state_path = os.path.join(snap_dir, "training_state.json")
    training_state: dict = {}
    if os.path.exists(training_state_path):
        with open(training_state_path) as f:
            training_state = json.load(f)

    for k, v in metrics.items():
        training_state.setdefault(k, v)

    print(f"  [resume] Loaded checkpoint from {snap_dir} (ep {metrics['episode']}, "
          f"BestScore={metrics['best_score']:+.3f})")
    return training_state


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = sorted(
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("ep") and os.path.isdir(os.path.join(checkpoint_dir, d))
    )
    return os.path.join(checkpoint_dir, candidates[-1]) if candidates else None


CHECKPOINT_INTERVAL = 500


def eval_clock_config() -> tuple[int, int]:
    return TIGHT_CLOCK_STEPS, LONG_CLOCK_STEPS


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    global BATCH_SIZE, NUM_ENVS, REPLAY_CAP, GOAL_TOL, TERMINATE_ON_GOAL_ZONE, GAMMA_RL
    global TIMER_CURRICULUM, ALLOW_HARD_CBF_BYPASS, GOAL_CAPTURE_SHAPING
    global GOAL_DISTANCE_PENALTY, NEAR_GOAL_RADIUS, NEAR_GOAL_BONUS_SCALE, ACTION_PENALTY
    global RISK_ALIGN_ALPHA, BOOTSTRAP_CDDP_EPISODES

    configure_results_dir(args.run_name)
    NUM_ENVS = max(1, int(args.num_envs))
    BATCH_SIZE = max(1, int(args.batch_size))
    REPLAY_CAP = max(BATCH_SIZE, int(args.replay_cap))
    GOAL_TOL = float(args.goal_tol)
    TERMINATE_ON_GOAL_ZONE = bool(args.terminate_on_goal_zone)
    GAMMA_RL = float(args.gamma_rl)
    TIMER_CURRICULUM = args.timer_curriculum
    ALLOW_HARD_CBF_BYPASS = bool(args.allow_hard_cbf_bypass)
    GOAL_CAPTURE_SHAPING = not bool(args.disable_goal_capture_shaping)
    GOAL_DISTANCE_PENALTY = float(args.goal_distance_penalty)
    NEAR_GOAL_RADIUS = float(args.near_goal_radius)
    NEAR_GOAL_BONUS_SCALE = float(args.near_goal_bonus_scale)
    ACTION_PENALTY = float(args.action_penalty)
    RISK_ALIGN_ALPHA = float(args.risk_align_alpha)
    BOOTSTRAP_CDDP_EPISODES = int(args.bootstrap_cddp_episodes)
    train_every = max(1, int(args.train_every))
    gradient_steps = max(0, int(args.gradient_steps))

    print("=" * 78)
    print("  Phase 5 v5: Adaptive CDDP — Stakes-Input Actor + Risk-Aligned Reward")
    print(
        f"  Plane=[{PLANE_MIN}, {PLANE_MAX}]^2  "
        f"GoalZone={[GOAL_LOW.tolist(), GOAL_HIGH.tolist()]}  Obstacles={len(OBS_R)}"
    )
    print(
        f"  ActorDim={ACTOR_STATE_DIM}  RiskAlignAlpha={RISK_ALIGN_ALPHA:.3f}  "
        f"BootstrapCDDPEps={BOOTSTRAP_CDDP_EPISODES}"
    )
    print(
        f"  Success={'goal zone' if TERMINATE_ON_GOAL_ZONE else 'sampled goal tolerance'}  "
        f"GoalTol={GOAL_TOL:.3f}  TimerCurriculum={TIMER_CURRICULUM}"
    )
    print(
        f"  Episodes={args.episodes}  Pretrain={args.pretrain_episodes}  "
        f"VectorEnvs={NUM_ENVS}  MaxSteps=[tight {TIGHT_CLOCK_STEPS}, long {LONG_CLOCK_STEPS}]"
    )
    print(
        f"  Learner: batch={BATCH_SIZE}  train_every={train_every}  "
        f"gradient_steps={gradient_steps}  replay_cap={REPLAY_CAP}"
    )
    print(f"  Gamma_CBF={GAMMA_CBF}  Gamma_RL={GAMMA_RL}  P_ref={P_REF}")
    print(f"  Hard CBF bypass allowed={ALLOW_HARD_CBF_BYPASS}")
    print("  v5 changes: actor takes stakes as input (6-dim), risk-aligned reward bonus,")
    print(f"              pretrain={args.pretrain_episodes} (was 10000), bootstrap={BOOTSTRAP_CDDP_EPISODES} aggressive eps")
    if args.run_name:
        print(f"  Run artifacts: {RESULTS_DIR}")
    print("=" * 78)

    env = make_env()
    agent = CDDPAgent(env)
    deontic_buffer    = DeonticReplayBuffer(REPLAY_CAP)
    tight_util_buffer = UtilReplayBuffer(REPLAY_CAP // 2)
    loose_util_buffer = UtilReplayBuffer(REPLAY_CAP // 2)
    utilitarian_buffer = loose_util_buffer   # legacy alias
    ou_noise = OUNoise(action_dim=2)

    # Episode mode IDs
    mode_pre  = 0   # strict S=0, data → loose_util_buffer
    mode_deon = 1   # strict S=0, data → deontic_buffer (labeled after episode)
    mode_util = 2   # dynamic S, data → tight/loose_util_buffer
    mode_seed = 3   # aggressive proportional ctrl, data → deontic_buffer
    mode_boost = 4  # aggressive straight-to-goal (no CBF), data → tight_util_buffer
                    # Used during first BOOTSTRAP_CDDP_EPISODES of CDDP phase to
                    # seed Z_free with successful direct-path trajectories.

    pretrain_target = min(args.pretrain_episodes, args.episodes)
    seed_target = min(args.seed_episodes, args.episodes)
    announced_seed_complete = False
    cddp_episodes_launched = 0   # tracks how many CDDP-phase episodes started
    boost_episodes_launched = 0  # tracks how many boost episodes have been launched

    env.reset(torch.zeros(NUM_ENVS, 5, dtype=torch.float32, device=DEVICE), max_steps=LONG_CLOCK_STEPS)
    current_goals = GOAL.to(DEVICE).unsqueeze(0).expand(NUM_ENVS, -1).clone()
    env.set_goal(current_goals)
    active = torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE)
    mode_ids = torch.full((NUM_ENVS,), -1, dtype=torch.long, device=DEVICE)
    episode_max_steps = torch.zeros(NUM_ENVS, dtype=torch.long)  # CPU
    per_env_steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=DEVICE)
    current_returns = torch.zeros(NUM_ENVS, dtype=torch.float32, device=DEVICE)
    current_successes = torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE)
    current_crashed = torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE)
    current_stakes_sum = torch.zeros(NUM_ENVS, dtype=torch.float32, device=DEVICE)
    current_stakes_count = torch.zeros(NUM_ENVS, dtype=torch.long, device=DEVICE)
    current_excess_violations = torch.zeros(NUM_ENVS, dtype=torch.long, device=DEVICE)
    current_first_goal_zone_step = torch.full((NUM_ENVS,), -1, dtype=torch.long, device=DEVICE)
    max_episode_clock = LONG_CLOCK_STEPS
    deon_episode_indices = torch.full(
        (NUM_ENVS, max_episode_clock + 1),
        -1,
        dtype=torch.long,
        device=DEVICE,
    )
    deon_episode_counts = torch.zeros(NUM_ENVS, dtype=torch.long, device=DEVICE)

    total_steps = 0
    vector_steps = 0
    util_update_steps = 0
    deon_update_steps = 0
    launched_episodes = 0
    completed_episodes = 0
    next_log_episode = LOG_INTERVAL
    train_started_at = time.perf_counter()

    ep_returns: list[float] = []
    ep_successes: list[bool] = []
    ep_failures: list[bool] = []
    ep_avg_stakes: list[float] = []
    ep_excess_violations: list[int] = []
    ep_steps_after_goal_zone_entry: list[int] = []
    ep_modes: list[str] = []

    last_loss_z_free = float("nan")
    last_loss_actor = float("nan")
    last_loss_z_deon = float("nan")
    announced_cddp_phase = False
    best_score = -float("inf")
    best_util_success = float("nan")
    best_util_p_excess = float("nan")
    best_util_avg_viol = float("nan")
    best_avg_steps_after_goal_zone = float("nan")

    # ------------------------------------------------------------------
    # Resume from checkpoint if requested
    # ------------------------------------------------------------------
    resume_snap = None
    if getattr(args, "resume", None):
        resume_snap = args.resume
        if resume_snap == "auto":
            resume_snap = find_latest_checkpoint(CHECKPOINT_DIR)
            if resume_snap is None:
                print("  [resume] No checkpoint found — starting from scratch.")
    if resume_snap is not None:
        try:
            ts = load_checkpoint_for_resume(agent, resume_snap)
            completed_episodes          = int(ts.get("completed_episodes",          ts.get("episode", 0)))
            # Reset launched to completed on resume: in-flight episodes at
            # checkpoint time are lost when the env is re-initialized, so
            # re-launching from completed prevents phase targets being missed.
            launched_episodes           = completed_episodes
            total_steps                 = int(ts.get("total_steps",                 0))
            vector_steps                = int(ts.get("vector_steps",                0))
            util_update_steps           = int(ts.get("util_update_steps",           0))
            deon_update_steps           = int(ts.get("deon_update_steps",           0))
            cddp_episodes_launched      = int(ts.get("cddp_episodes_launched",      0))
            boost_episodes_launched     = int(ts.get("boost_episodes_launched",     0))
            best_score                  = float(ts.get("best_score",                best_score))
            best_util_success           = float(ts.get("best_util_success",         float("nan")))
            best_util_p_excess          = float(ts.get("best_util_p_excess",        float("nan")))
            best_util_avg_viol          = float(ts.get("best_util_avg_viol",        float("nan")))
            best_avg_steps_after_goal_zone = float(ts.get("best_avg_steps_after_goal_zone", float("nan")))
            ou_noise.sigma              = float(ts.get("ou_sigma",                  ou_noise.sigma))
            announced_cddp_phase        = bool(ts.get("announced_cddp_phase",       False))
            announced_seed_complete     = bool(ts.get("announced_seed_complete",    False))
            next_log_episode = completed_episodes + LOG_INTERVAL
            print(f"  [resume] Continuing from episode {completed_episodes}/{args.episodes} "
                  f"(BestScore={best_score:+.3f}, OU_sigma={ou_noise.sigma:.4f})")
        except FileNotFoundError as exc:
            print(f"  [resume] WARNING: {exc}. Starting from scratch.")
    # ------------------------------------------------------------------

    def maybe_complete_seed_phase() -> None:
        nonlocal announced_seed_complete
        if announced_seed_complete or completed_episodes < seed_target:
            return
        print("\n" + "=" * 78)
        print(
            f"  Seed phase complete after {completed_episodes} episodes. "
            "Starting pre-training."
        )
        print("=" * 78)
        announced_seed_complete = True
        ou_noise.sigma = OU_SIGMA
        ou_noise.reset()

    def maybe_start_cddp_phase() -> None:
        nonlocal announced_cddp_phase
        if announced_cddp_phase or completed_episodes < seed_target + pretrain_target:
            return

        print("\n" + "=" * 78)
        print(
            f"  Pre-training complete after {completed_episodes} completed episodes. "
            "Starting Bifurcated CDDP Phase (v5: stakes-input actor + bootstrap)."
        )
        print("=" * 78)
        announced_cddp_phase = True

        ou_noise.sigma = 0.5
        ou_noise.reset()

    def format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m"
        if minutes:
            return f"{minutes:d}m{secs:02d}s"
        return f"{secs:d}s"

    def launch_slots(slot_indices: torch.Tensor) -> None:
        nonlocal launched_episodes, cddp_episodes_launched, boost_episodes_launched

        launch_slots_list: list[int] = []
        launch_modes: list[int] = []
        launch_clocks: list[int] = []

        for slot_t in slot_indices:
            slot = int(slot_t.item())
            if launched_episodes >= args.episodes:
                break
            if not announced_seed_complete and launched_episodes >= seed_target:
                break
            if announced_seed_complete and not announced_cddp_phase and launched_episodes >= seed_target + pretrain_target:
                break

            if not announced_seed_complete:
                mode_id = mode_seed
                clock_override = None
            elif not announced_cddp_phase:
                mode_id = mode_pre
                clock_override = None
            else:
                # CDDP phase: deon / util / boost interleaving.
                # Pattern during bootstrap phase (first BOOTSTRAP_CDDP_EPISODES):
                #   cddp_idx % 3 == 0 → deon
                #   cddp_idx % 3 == 1 → util
                #   cddp_idx % 3 == 2 → boost  (if boost budget remains)
                # After bootstrap budget exhausted or past bootstrap phase:
                #   cddp_idx % 2 == 0 → deon
                #   cddp_idx % 2 == 1 → util
                cddp_idx = cddp_episodes_launched
                in_boost_phase = boost_episodes_launched < BOOTSTRAP_CDDP_EPISODES

                if in_boost_phase:
                    slot_in_triple = cddp_idx % 3
                    if slot_in_triple == 0:
                        mode_id = mode_deon
                        clock_override = None
                    elif slot_in_triple == 1:
                        mode_id = mode_util
                        clock_override = None
                    else:
                        mode_id = mode_boost
                        # Force tight clock for bootstrap episodes
                        clock_override = int(torch.randint(
                            TIGHT_CLOCK_STEPS, TIGHT_CLOCK_THRESHOLD + 1, (1,)
                        ).item())
                        boost_episodes_launched += 1
                else:
                    mode_id = mode_deon if cddp_idx % 2 == 0 else mode_util
                    clock_override = None

                cddp_episodes_launched += 1

            launch_slots_list.append(slot)
            launch_modes.append(mode_id)
            launch_clocks.append(clock_override)   # None = use curriculum
            launched_episodes += 1

        if not launch_slots_list:
            return

        slots = torch.tensor(launch_slots_list, dtype=torch.long, device=DEVICE)
        n = int(slots.numel())
        starts, goals = sample_episode_context(n)

        # Sample clocks from curriculum, then override where specified
        clocks_tensor = sample_episode_steps(
            n,
            curriculum=TIMER_CURRICULUM,
            progress=launched_episodes / max(1, args.episodes),
        ).cpu()
        for i, override in enumerate(launch_clocks):
            if override is not None:
                clocks_tensor[i] = override

        starts[:, 4] = clocks_tensor.to(device=DEVICE, dtype=starts.dtype) / float(LONG_CLOCK_STEPS)
        env._state[slots] = starts
        if env.elapsed_steps is not None:
            env.elapsed_steps[slots] = 0.0
        current_goals[slots] = goals
        env.set_goal(current_goals)
        active[slots] = True
        mode_ids[slots] = torch.tensor(launch_modes, dtype=torch.long, device=DEVICE)
        episode_max_steps[slots.cpu()] = clocks_tensor
        per_env_steps[slots] = 0
        current_returns[slots] = 0.0
        current_successes[slots] = False
        current_crashed[slots] = False
        current_stakes_sum[slots] = 0.0
        current_stakes_count[slots] = 0
        current_excess_violations[slots] = 0
        current_first_goal_zone_step[slots] = -1
        ou_noise.reset_idx(slots)
        deon_episode_indices[slots] = -1
        deon_episode_counts[slots] = 0

    def log_progress(force: bool = False) -> None:
        nonlocal best_score, best_util_success, best_util_p_excess
        nonlocal best_util_avg_viol, best_avg_steps_after_goal_zone, next_log_episode
        if not ep_returns:
            return
        if not force and completed_episodes < next_log_episode:
            return

        while next_log_episode <= completed_episodes:
            next_log_episode += LOG_INTERVAL

        keep = LOG_INTERVAL * 2
        if len(ep_returns) > keep:
            del ep_returns[:len(ep_returns) - keep]
            del ep_successes[:len(ep_successes) - keep]
            del ep_failures[:len(ep_failures) - keep]
            del ep_avg_stakes[:len(ep_avg_stakes) - keep]
            del ep_excess_violations[:len(ep_excess_violations) - keep]
            del ep_steps_after_goal_zone_entry[:len(ep_steps_after_goal_zone_entry) - keep]
            del ep_modes[:len(ep_modes) - keep]

        idx0 = max(0, len(ep_returns) - LOG_INTERVAL)
        pre_idx   = [i for i in range(idx0, len(ep_returns)) if ep_modes[i] == "pre"]
        util_idx  = [i for i in range(idx0, len(ep_returns)) if ep_modes[i] == "util"]
        boost_idx = [i for i in range(idx0, len(ep_returns)) if ep_modes[i] == "boost"]
        deon_idx  = [i for i in range(idx0, len(ep_returns)) if ep_modes[i] == "deon"]

        avg_ret      = sum(ep_returns[idx0:]) / (len(ep_returns) - idx0)
        succ_rate    = sum(ep_successes[idx0:]) / (len(ep_successes) - idx0)
        fail_rate    = sum(ep_failures[idx0:]) / (len(ep_failures) - idx0)
        avg_s_util   = sum(ep_avg_stakes[i] for i in util_idx) / max(1, len(util_idx))
        excess_util  = sum(ep_excess_violations[i] for i in util_idx)
        util_success = sum(1 for i in util_idx if ep_successes[i]) / max(1, len(util_idx))
        boost_success = sum(1 for i in boost_idx if ep_successes[i]) / max(1, len(boost_idx)) if boost_idx else float("nan")
        util_p_excess = sum(1 for i in util_idx if ep_excess_violations[i] > 0) / max(1, len(util_idx))
        util_avg_viol = sum(ep_excess_violations[i] for i in util_idx) / max(1, len(util_idx))
        util_avg_steps_after_goal_zone = sum(ep_steps_after_goal_zone_entry[i] for i in util_idx) / max(1, len(util_idx))
        deon_fail    = sum(1 for i in deon_idx if ep_failures[i]) / max(1, len(deon_idx))
        elapsed      = max(1e-9, time.perf_counter() - train_started_at)
        eps_per_sec  = completed_episodes / elapsed
        env_steps_per_sec = total_steps / elapsed
        updates_per_sec = (util_update_steps + deon_update_steps) / elapsed
        eta = (
            format_duration((args.episodes - completed_episodes) / eps_per_sec)
            if eps_per_sec > 0
            else "n/a"
        )

        best_note = ""
        current_score = float("nan")
        if announced_cddp_phase and util_idx:
            if args.best_criterion == "success":
                current_score = util_success
            else:
                current_score = (
                    util_success
                    - 0.50 * util_p_excess
                    - 0.002 * util_avg_viol
                    - 0.001 * util_avg_steps_after_goal_zone
                )
            if current_score >= best_score:
                best_score = current_score
                best_util_success = util_success
                best_util_p_excess = util_p_excess
                best_util_avg_viol = util_avg_viol
                best_avg_steps_after_goal_zone = util_avg_steps_after_goal_zone
                save_best_checkpoints(agent)
                best_note = " | saved best"

        best_score_display           = "n/a" if not math.isfinite(best_score) else f"{best_score:+.3f}"
        best_util_success_display    = "n/a" if not math.isfinite(best_util_success) else f"{best_util_success:.0%}"
        best_util_p_excess_display   = "n/a" if not math.isfinite(best_util_p_excess) else f"{best_util_p_excess:.0%}"
        best_util_avg_viol_display   = "n/a" if not math.isfinite(best_util_avg_viol) else f"{best_util_avg_viol:.1f}"
        best_avg_steps_display       = "n/a" if not math.isfinite(best_avg_steps_after_goal_zone) else f"{best_avg_steps_after_goal_zone:.1f}"
        boost_succ_display           = "n/a" if not math.isfinite(boost_success) else f"{boost_success:.0%}"

        print(
            f"Ep {completed_episodes:>4}/{args.episodes} | "
            f"Phase={'PRE' if not announced_cddp_phase else 'CDDP'} | "
            f"Ret={avg_ret:+7.2f} | "
            f"Succ={succ_rate:.0%} | Fail={fail_rate:.0%} | "
            f"BestScore={best_score_display} | "
            f"BestUtilSucc={best_util_success_display} | "
            f"BestUtilPExcess={best_util_p_excess_display} | "
            f"BestUtilAvgViol={best_util_avg_viol_display} | "
            f"BestAvgGoalZoneSteps={best_avg_steps_display} | "
            f"StrictFail={deon_fail:.0%} | "
            f"AvgS_util={avg_s_util:.3f} | "
            f"ExcessViol_util={excess_util:>3} | "
            f"BoostSucc={boost_succ_display} | "
            f"BoostEps={boost_episodes_launched} | "
            f"PreEps={len(pre_idx):>2} | "
            f"Active={int(active.sum().item()):>2}/{NUM_ENVS} | "
            f"Buf(deon/tight/loose)={len(deontic_buffer)}/{len(tight_util_buffer)}/{len(loose_util_buffer)} | "
            f"Rate={eps_per_sec:.2f}eps/s {env_steps_per_sec:.0f}steps/s "
            f"{updates_per_sec:.1f}upd/s ETA={eta} | "
            f"OU_sigma={ou_noise.sigma:.3f} | "
            f"Lfree={last_loss_z_free:.3f} Lact={last_loss_actor:.3f} "
            f"Ldeon={last_loss_z_deon:.3f}"
            f"{best_note}"
        )

    # Save ep0 snapshot
    if completed_episodes == 0:
        save_periodic_checkpoint(
            agent, 0, best_score, best_util_success, best_util_p_excess,
            best_util_avg_viol, best_avg_steps_after_goal_zone,
            training_state={
                "completed_episodes": 0, "launched_episodes": 0,
                "total_steps": 0, "vector_steps": 0,
                "util_update_steps": 0, "deon_update_steps": 0,
                "cddp_episodes_launched": 0, "boost_episodes_launched": 0,
                "best_score": best_score, "best_util_success": best_util_success,
                "best_util_p_excess": best_util_p_excess,
                "best_util_avg_viol": best_util_avg_viol,
                "best_avg_steps_after_goal_zone": best_avg_steps_after_goal_zone,
                "ou_sigma": ou_noise.sigma,
                "announced_cddp_phase": announced_cddp_phase,
                "announced_seed_complete": announced_seed_complete,
            },
        )

    if pretrain_target == 0:
        maybe_start_cddp_phase()
    launch_slots(torch.arange(NUM_ENVS, device=DEVICE))

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    while completed_episodes < args.episodes:
        if not active.any():
            maybe_complete_seed_phase()
            maybe_start_cddp_phase()
            launch_slots(torch.arange(NUM_ENVS, device=DEVICE))
            if not active.any():
                break

        s_before = env.state.detach().clone()
        prev_pos = s_before[:, :2].clone()

        # Compute stakes for util environments (zeros for all others)
        stakes = torch.zeros(NUM_ENVS, dtype=torch.float32, device=DEVICE)
        pre_mask   = active & (mode_ids == mode_pre)
        deon_mask  = active & (mode_ids == mode_deon)
        util_mask  = active & (mode_ids == mode_util)
        seed_mask  = active & (mode_ids == mode_seed)
        boost_mask = active & (mode_ids == mode_boost)

        if util_mask.any():
            with torch.no_grad():
                stakes[util_mask] = agent.stakes_from_deon(s_before[util_mask], detach=True)
            current_stakes_sum[util_mask] += stakes[util_mask]
            current_stakes_count[util_mask] += 1

        with torch.no_grad():
            # Build augmented state [s, stakes] for actor input
            s_aug = aug_s(s_before, stakes)   # [NUM_ENVS, 6]

            u_nom = torch.zeros(NUM_ENVS, 2, dtype=s_before.dtype, device=DEVICE)
            if active.any():
                u_nom[active] = agent.actor(s_aug[active])

            deon_exploration_allowed = bool(args.deon_exploration_noise)
            if total_steps < args.warmup_steps:
                random_mask = active if deon_exploration_allowed else (pre_mask | util_mask | seed_mask | boost_mask)
                if random_mask.any():
                    u_nom[random_mask] = (torch.rand(int(random_mask.sum().item()), 2, device=DEVICE) * 2.0 - 1.0) * U_MAX
            else:
                noise_mask = active if deon_exploration_allowed else (pre_mask | util_mask | seed_mask | boost_mask)
                if noise_mask.any():
                    noise = ou_noise.sample()
                    u_nom[noise_mask] = torch.clamp(u_nom[noise_mask] + noise[noise_mask], -U_MAX, U_MAX)

            # Goal-seeking blend at high stakes (util episodes only)
            if announced_cddp_phase and GOAL_SEEK_BLEND > 0:
                high_stakes_util = util_mask & (stakes > GOAL_SEEK_STAKES_THRESHOLD)
                if high_stakes_util.any():
                    hs_pos = s_before[high_stakes_util, :2]
                    hs_goals = current_goals[high_stakes_util]
                    goal_dir = hs_goals - hs_pos
                    goal_dir = F.normalize(goal_dir, dim=-1) * U_MAX
                    u_nom[high_stakes_util] = (
                        (1.0 - GOAL_SEEK_BLEND) * u_nom[high_stakes_util]
                        + GOAL_SEEK_BLEND * goal_dir
                    )

            u_nom[~active] = 0.0
            u_safe = agent.safe_action(s_before, u_nom, stakes)
            u_safe[~active] = 0.0

            # Seed episodes: proportional control toward goal (bypasses CBF)
            if seed_mask.any():
                pos_seed = s_before[seed_mask, :2]
                goal_seed = current_goals[seed_mask]
                direction = goal_seed - pos_seed
                dist = direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                u_prop = U_MAX * direction / dist
                seed_noise = 0.3 * torch.randn_like(u_prop)
                u_safe[seed_mask] = torch.clamp(u_prop + seed_noise, -U_MAX, U_MAX)

            # Bootstrap episodes: straight-to-goal, no CBF (aggressive baseline)
            # This provides positive Q-targets for direct-path states so Z_free
            # doesn't assign near-zero value to obstacle-proximal states when
            # stakes are high.
            if boost_mask.any():
                pos_boost  = s_before[boost_mask, :2]
                goal_boost = current_goals[boost_mask]
                direction  = goal_boost - pos_boost
                dist       = direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                u_boost    = U_MAX * direction / dist
                boost_noise = 0.3 * torch.randn_like(u_boost)
                u_safe[boost_mask] = torch.clamp(u_boost + boost_noise, -U_MAX, U_MAX)

            out = env.step(u_safe)

        next_state = out["state"].detach()
        per_env_steps[active] += 1
        _active_cpu  = active.cpu()
        _ems_active  = episode_max_steps[_active_cpu].to(DEVICE)
        active_steps_remaining = (_ems_active - per_env_steps[active]).clamp_min(0)
        next_state[active, 4] = (
            active_steps_remaining.to(dtype=next_state.dtype)
            / float(LONG_CLOCK_STEPS)
        )
        env._state = next_state
        if env.elapsed_steps is not None:
            env.elapsed_steps[active] = per_env_steps[active].to(dtype=env.elapsed_steps.dtype)

        reached = success_mask(next_state[:, :2], current_goals, out["goal_reached"].detach()) & active
        unsafe  = out["unsafe"].detach() & active
        timeout = active & (per_env_steps >= episode_max_steps.to(DEVICE)) & ~reached
        done    = reached | timeout
        inside_goal_zone = goal_zone_mask(next_state[:, :2]) & active
        first_goal_zone_entry = inside_goal_zone & (current_first_goal_zone_step < 0)
        current_first_goal_zone_step[first_goal_zone_entry] = per_env_steps[first_goal_zone_entry]

        # Base reward
        reward = torch.zeros(NUM_ENVS, dtype=torch.float32, device=DEVICE)
        if active.any():
            reward[active] = compute_reward(
                next_state[active, :2],
                prev_pos[active],
                current_goals[active],
                reached[active],
                unsafe[active],
                timeout[active],
                timestep=per_env_steps[active],
                max_episode_steps=_ems_active,
            )

        # ------------------------------------------------------------------ v5
        # Risk-aligned reward bonus for util episodes:
        #   +RISK_ALIGN_ALPHA * stakes * obstacle_proximity
        # This breaks the gradient dead zone by providing a positive signal
        # for near-obstacle states when stakes are high, before Z_free has
        # calibrated on successful direct-path trajectories.
        if RISK_ALIGN_ALPHA > 0 and util_mask.any():
            with torch.no_grad():
                prox = compute_obstacle_proximity(
                    next_state[:, :2],
                    timestep=per_env_steps,
                )
                risk_bonus = RISK_ALIGN_ALPHA * stakes * prox
                reward[util_mask] = reward[util_mask] + risk_bonus[util_mask]
        # ------------------------------------------------------------------

        # Route transitions to replay buffers
        util_pre_boost_mask = active & (
            (mode_ids == mode_pre) | (mode_ids == mode_util) | (mode_ids == mode_boost)
        )
        if util_pre_boost_mask.any():
            _upm_cpu = util_pre_boost_mask.cpu()
            _tc_cpu  = episode_max_steps <= TIGHT_CLOCK_THRESHOLD   # CPU bool
            tight_cpu = _upm_cpu & _tc_cpu
            loose_cpu = _upm_cpu & ~_tc_cpu
            if tight_cpu.any():
                tight_mask_gpu = tight_cpu.to(DEVICE)
                tight_util_buffer.push_batch(
                    s_before[tight_mask_gpu],
                    u_safe[tight_mask_gpu],
                    reward[tight_mask_gpu],
                    next_state[tight_mask_gpu],
                    done[tight_mask_gpu],
                )
            if loose_cpu.any():
                loose_mask_gpu = loose_cpu.to(DEVICE)
                loose_util_buffer.push_batch(
                    s_before[loose_mask_gpu],
                    u_safe[loose_mask_gpu],
                    reward[loose_mask_gpu],
                    next_state[loose_mask_gpu],
                    done[loose_mask_gpu],
                )

        seed_or_deon_mask = active & ((mode_ids == mode_deon) | (mode_ids == mode_seed))
        if seed_or_deon_mask.any():
            sd_slots = torch.where(seed_or_deon_mask)[0]
            written_idx = deontic_buffer.push_batch(
                s_before[seed_or_deon_mask],
                u_safe[seed_or_deon_mask],
                reward[seed_or_deon_mask],
                next_state[seed_or_deon_mask],
                done[seed_or_deon_mask],
                torch.zeros(sd_slots.shape[0], dtype=torch.float32, device=DEVICE),
                labeled=False,
            )
            write_cols = deon_episode_counts[sd_slots].clamp_max(max_episode_clock)
            deon_episode_indices[sd_slots, write_cols] = written_idx
            deon_episode_counts[sd_slots] += 1

        current_returns[active] += reward[active]
        current_successes |= reached
        current_crashed   |= unsafe
        current_excess_violations[util_mask] += unsafe[util_mask].to(dtype=torch.long)

        active_transitions = int(active.sum().item())
        total_steps  += active_transitions
        vector_steps += 1

        # Learner updates
        if total_steps >= args.warmup_steps and gradient_steps > 0 and vector_steps % train_every == 0:
            for _ in range(gradient_steps):
                _tight_ready = len(tight_util_buffer) >= BATCH_SIZE // 2
                _loose_ready = len(loose_util_buffer) >= BATCH_SIZE // 2
                if _tight_ready or _loose_ready:
                    _lz, _la = agent.update_utilitarian(
                        tight_util_buffer,
                        loose_util_buffer,
                        strict_stakes=not announced_cddp_phase,
                    )
                    if _lz is not None:
                        last_loss_z_free = _lz.item()
                        last_loss_actor  = _la.item()
                        util_update_steps += 1
                if len(deontic_buffer) >= BATCH_SIZE:
                    loss_d = agent.update_deontic(deontic_buffer)
                    last_loss_z_deon = loss_d.item()
                    deon_update_steps += 1

        # Episode completion
        done_indices = torch.where(done)[0]
        if done_indices.numel() > 0:
            batch = {
                'mode_ids':        mode_ids[done_indices].cpu().tolist(),
                'episode_failed':  (timeout[done_indices] | (current_crashed[done_indices] & ~current_successes[done_indices])).cpu().tolist(),
                'deon_counts':     deon_episode_counts[done_indices].cpu().tolist(),
                'returns':         current_returns[done_indices].cpu().tolist(),
                'successes':       current_successes[done_indices].cpu().tolist(),
                'stakes_sum':      current_stakes_sum[done_indices].cpu().tolist(),
                'stakes_count':    current_stakes_count[done_indices].cpu().tolist(),
                'excess_viol':     current_excess_violations[done_indices].cpu().tolist(),
                'first_goal_step': current_first_goal_zone_step[done_indices].cpu().tolist(),
                'per_env_steps':   per_env_steps[done_indices].cpu().tolist(),
            }
        for i, idx_t_loop in enumerate(done_indices):
            idx    = int(idx_t_loop.item())
            idx_t  = torch.tensor([idx], dtype=torch.long, device=DEVICE)
            mode_id = int(batch['mode_ids'][i])
            mode_name = (
                "seed"  if mode_id == mode_seed  else
                "pre"   if mode_id == mode_pre   else
                "deon"  if mode_id == mode_deon  else
                "boost" if mode_id == mode_boost else "util"
            )
            episode_failed = bool(batch['episode_failed'][i])

            if mode_id in (mode_deon, mode_seed) and batch['deon_counts'][i] > 0:
                count = int(min(batch['deon_counts'][i], max_episode_clock + 1))
                episode_replay_idx = deon_episode_indices[idx, :count]
                deontic_buffer.label_indices(episode_replay_idx, episode_failed)

            ep_returns.append(float(batch['returns'][i]))
            ep_successes.append(bool(batch['successes'][i]))
            ep_failures.append(episode_failed)
            stakes_count = max(1, int(batch['stakes_count'][i]))
            ep_avg_stakes.append(float(batch['stakes_sum'][i]) / stakes_count)
            ep_excess_violations.append(int(batch['excess_viol'][i]))
            first_goal_zone_step = int(batch['first_goal_step'][i])
            ep_steps_after_goal_zone_entry.append(
                int(batch['per_env_steps'][i]) - first_goal_zone_step if first_goal_zone_step >= 0 else 0
            )
            ep_modes.append(mode_name)
            completed_episodes += 1

            if completed_episodes % 500 == 0 and len(deontic_buffer) > BATCH_SIZE:
                with torch.no_grad():
                    diag_s, diag_y = deontic_buffer.sample(256)
                    p_cat_diag = torch.sigmoid(agent.z_deon(diag_s))
                p_cat_mean = p_cat_diag.mean().item()
                p_cat_frac_high = (p_cat_diag > 0.8).float().mean().item()
                label_pos_rate = diag_y.mean().item()
                print(f"[Z_deon diag ep={completed_episodes}] "
                      f"mean_P_cat={p_cat_mean:.3f} frac>0.8={p_cat_frac_high:.3f} "
                      f"label_pos_rate={label_pos_rate:.3f} "
                      f"boost_launched={boost_episodes_launched}")

            if completed_episodes % CHECKPOINT_INTERVAL == 0:
                save_periodic_checkpoint(
                    agent, completed_episodes,
                    best_score, best_util_success, best_util_p_excess,
                    best_util_avg_viol, best_avg_steps_after_goal_zone,
                    training_state={
                        "completed_episodes":             completed_episodes,
                        "launched_episodes":              launched_episodes,
                        "total_steps":                    total_steps,
                        "vector_steps":                   vector_steps,
                        "util_update_steps":              util_update_steps,
                        "deon_update_steps":              deon_update_steps,
                        "cddp_episodes_launched":         cddp_episodes_launched,
                        "boost_episodes_launched":        boost_episodes_launched,
                        "best_score":                     best_score,
                        "best_util_success":              best_util_success,
                        "best_util_p_excess":             best_util_p_excess,
                        "best_util_avg_viol":             best_util_avg_viol,
                        "best_avg_steps_after_goal_zone": best_avg_steps_after_goal_zone,
                        "ou_sigma":                       ou_noise.sigma,
                        "announced_cddp_phase":           announced_cddp_phase,
                        "announced_seed_complete":        announced_seed_complete,
                    },
                )

            active[idx] = False
            mode_ids[idx] = -1
            deon_episode_indices[idx] = -1
            deon_episode_counts[idx] = 0
            current_returns[idx] = 0.0
            current_successes[idx] = False
            current_crashed[idx] = False
            current_stakes_sum[idx] = 0.0
            current_stakes_count[idx] = 0
            current_excess_violations[idx] = 0
            current_first_goal_zone_step[idx] = -1
            ou_noise.reset_idx(idx_t)
            ou_noise.decay()

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if completed_episodes % 500 == 0:
                gc.collect()

        maybe_complete_seed_phase()
        maybe_start_cddp_phase()
        launch_slots(torch.where(~active)[0])
        log_progress(force=completed_episodes >= args.episodes)

    print("\n" + "=" * 78)
    print("  Saving Checkpoints and CDDP Trajectory Visualization")
    print("=" * 78)
    if load_best_checkpoints(agent):
        print(f"  Loaded best CDDP weights for final evaluation and plotting "
              f"(BestScore={best_score:+.3f}, BestUtilSucc={best_util_success:.0%}, "
              f"BestUtilPExcess={best_util_p_excess:.0%}, "
              f"BestUtilAvgViol={best_util_avg_viol:.1f}, "
              f"BestAvgStepsAfterGoalZoneEntry={best_avg_steps_after_goal_zone:.1f}).")
    else:
        print("  No best CDDP checkpoint found; using current weights for final artifacts.")
    save_checkpoints(agent)

    plot_generated = False
    try:
        plot_best()
        plot_generated = True
    except Exception as e:
        print(f"  Warning: Could not generate trajectory plot: {e}")

    print(f"  Saved actor/Z_free/Z_deon checkpoints to: {CHECKPOINT_DIR}")
    tight_eval_steps, long_eval_steps = eval_clock_config()
    print(
        f"  Eval clock calibration: tight={tight_eval_steps} steps, "
        f"long={long_eval_steps} steps."
    )
    if plot_generated:
        print(f"  Saved trajectory plot to: {TRAJECTORY_PLOT_PATH}")

    print("\nAdaptive CDDP v5 complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Adaptive CDDP v5.")
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--pretrain-episodes", type=int, default=PRETRAIN_EPISODES_DEFAULT)
    parser.add_argument("--seed-episodes", type=int, default=N_DEONTIC_SEED_EPISODES)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--replay-cap", type=int, default=REPLAY_CAP)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT_DIR_OR_auto",
        help="Resume from a checkpoint snapshot dir, or 'auto' to pick the latest.",
    )
    parser.add_argument("--goal-tol", type=float, default=GOAL_TOL)
    parser.add_argument("--terminate-on-goal-zone", action="store_true")
    parser.add_argument("--gamma-rl", type=float, default=GAMMA_RL)
    parser.add_argument(
        "--timer-curriculum",
        choices=["bimodal", "mixed", "long_only", "compression"],
        default=TIMER_CURRICULUM,
    )
    parser.add_argument("--allow-hard-cbf-bypass", action="store_true")
    parser.add_argument("--deon-exploration-noise", action="store_true")
    parser.add_argument("--disable-goal-capture-shaping", action="store_true")
    parser.add_argument("--goal-distance-penalty", type=float, default=GOAL_DISTANCE_PENALTY)
    parser.add_argument("--near-goal-radius", type=float, default=NEAR_GOAL_RADIUS)
    parser.add_argument("--near-goal-bonus-scale", type=float, default=NEAR_GOAL_BONUS_SCALE)
    parser.add_argument("--action-penalty", type=float, default=ACTION_PENALTY)
    parser.add_argument("--best-criterion", choices=["composite", "success"], default="composite")
    parser.add_argument("--train-every", type=int, default=TRAIN_EVERY)
    parser.add_argument("--gradient-steps", type=int, default=GRADIENT_STEPS)
    # v5-specific
    parser.add_argument(
        "--risk-align-alpha",
        type=float,
        default=RISK_ALIGN_ALPHA,
        help="Coefficient for risk-aligned reward bonus: +alpha*stakes*obstacle_proximity.",
    )
    parser.add_argument(
        "--bootstrap-cddp-episodes",
        type=int,
        default=BOOTSTRAP_CDDP_EPISODES,
        help="Number of CDDP episodes to run bootstrap aggressive-policy episodes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    train(parse_args())
