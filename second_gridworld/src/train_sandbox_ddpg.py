"""
train_sandbox_ddpg.py
=====================
Phase 3: Differentiable DDPG through the HOCBF QP Layer.

This file proves the key CDDP claim:
  Gradients can flow backwards from the Critic, through the torch.where
  operations inside project_action_cbf(), and into the Actor weights.

Architecture:
  Actor(s)     ->  u_nom  (B, 2), tanh-bounded
  project_cbf(s, u_nom, S=0) -> u_safe (fully differentiable)
  Critic(s, u_safe) -> Q  (B, 1)

  actor_loss = -Critic(s, project_cbf(s, Actor(s))).mean()
  backward() flows:  Critic <- QP projection <- Actor

Stakes are fixed at S=0 (strict mode) for this proof-of-concept.
The Deontic Critic that dynamically sets S will be Phase 4.

Run from repo root:
  python sandbox/train_sandbox_ddpg.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import random
import collections
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analytical_sandbox import DifferentiablePointMassEnv

# ---------------------------------------------------------------------------
# Hyperparameters  (single source of truth)
# ---------------------------------------------------------------------------
DEVICE       = torch.device("cpu")

# Environment
GOAL         = torch.tensor([3.0, 3.0])
OBS_C        = torch.tensor([1.5, 1.5])
OBS_R        = 0.5
DT           = 0.02
V_MAX_ENV    = 4.0

# Policy bounds  (tanh output scaled by U_MAX)
U_MAX        = 2.0

# HOCBF / CDDP
GAMMA_CBF    = 3.0          # class-K coefficient for HOCBF
TAU_CDDP     = 0.3          # relaxation threshold (unused at S=0)
ALPHA_CDDP   = 10.0         # softplus sharpness
STAKES       = 0.0          # S=0 → strict mode; change to 1.0 for deontic override

# Training
N_EPISODES   = 500
MAX_STEPS    = 150          # per episode
BATCH_SIZE   = 256
REPLAY_CAP   = 50_000
LR_ACTOR     = 3e-4
LR_CRITIC    = 1e-3
GAMMA_RL     = 0.99         # discount for TD
TAU_SOFT     = 0.005        # Polyak averaging for target nets
WARMUP_STEPS = 1_000        # random actions before learning

# Exploration noise (Gaussian)
NOISE_STD    = 0.3
NOISE_DECAY  = 0.995        # multiply per episode
NOISE_MIN    = 0.02

# Logging
LOG_INTERVAL = 50


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    s:     torch.Tensor  # (4,)
    a:     torch.Tensor  # (2,)   <-- u_safe actually applied in env
    r:     float
    s_:    torch.Tensor  # (4,)
    done:  bool


class ReplayBuffer:
    """Fixed-capacity circular replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.buf: collections.deque[Transition] = collections.deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buf.append(Transition(*args))

    def sample(self, n: int) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self.buf, n)
        s, a, r, s_, done = zip(*batch)
        return (
            torch.stack(s).to(DEVICE),
            torch.stack(a).to(DEVICE),
            torch.tensor(r,    dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.stack(s_).to(DEVICE),
            torch.tensor(done, dtype=torch.float32, device=DEVICE).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# Neural Networks
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, out_dim: int, hidden: int = 256, layers: int = 2) -> nn.Sequential:
    """Build a simple MLP with ReLU activations."""
    dims = [in_dim] + [hidden] * layers + [out_dim]
    mods: list[nn.Module] = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


class Actor(nn.Module):
    """Maps state -> u_nom.

    Output is tanh-scaled to [-U_MAX, U_MAX] so the actor always proposes
    a bounded nominal action.  The QP projection then makes it safe.
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden: int = 256) -> None:
        super().__init__()
        self.net = _mlp(state_dim, action_dim, hidden)
        self.scale = U_MAX

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.net(s))   # (B, 2)


class Critic(nn.Module):
    """Maps (state, action) -> Q-value.

    Action is concatenated to the state at the input layer.
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden: int = 256) -> None:
        super().__init__()
        self.net = _mlp(state_dim + action_dim, 1, hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))    # (B, 1)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging:  θ_target ← τ·θ_source + (1−τ)·θ_target."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


# ---------------------------------------------------------------------------
# DDPG Agent  (wraps Actor + Critic + QP projection)
# ---------------------------------------------------------------------------

class CDDPAgent:
    """DDPG agent with an analytical HOCBF safety layer.

    The safety layer is NOT a learned component — it is a closed-form
    differentiable projection.  Its only effect on learning is to redirect
    gradients: the actor is trained to produce safe-by-design nominal actions.
    """

    def __init__(self, env: DifferentiablePointMassEnv) -> None:
        self.env = env

        # Actor and its Polyak target (target not used for gradient but for TD)
        self.actor        = Actor().to(DEVICE)
        self.actor_target = Actor().to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.requires_grad_(False)

        # Critic and its Polyak target
        self.critic        = Critic().to(DEVICE)
        self.critic_target = Critic().to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Fixed stakes for Phase 3: S = 0 everywhere (strict safety)
        self._stakes_val = float(STAKES)

    # ------------------------------------------------------------------
    def _safe_action(
        self,
        s: torch.Tensor,
        u_nom: torch.Tensor,
        stakes_val: float | None = None,
    ) -> torch.Tensor:
        """Pass u_nom through the HOCBF QP projection.

        All operations inside project_action_cbf use torch ops, so
        autograd can track this call.

        Parameters
        ----------
        s          : (B, 4) states
        u_nom      : (B, 2) nominal actions from Actor
        stakes_val : scalar S value to broadcast; default = self._stakes_val

        Returns
        -------
        u_safe : (B, 2)  safe actions (gradient-connected to u_nom)
        """
        B = s.shape[0]
        sv = stakes_val if stakes_val is not None else self._stakes_val
        stakes = torch.full((B,), sv, dtype=s.dtype, device=s.device)

        return self.env.project_action_cbf(
            s, u_nom,
            stakes=stakes,
            gamma=GAMMA_CBF,
            tau=TAU_CDDP,
            alpha_prime=ALPHA_CDDP,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, s: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        """Select u_safe for environment interaction (no gradient tracking)."""
        u_nom = self.actor(s.unsqueeze(0))           # (1, 2)
        if noise_std > 0:
            u_nom = u_nom + noise_std * torch.randn_like(u_nom)
            u_nom = u_nom.clamp(-U_MAX, U_MAX)
        u_safe = self._safe_action(s.unsqueeze(0), u_nom)
        return u_safe.squeeze(0)                      # (2,)

    # ------------------------------------------------------------------
    def update(self, replay: ReplayBuffer) -> tuple[float, float]:
        """One DDPG gradient step.

        Returns (critic_loss, actor_loss) as Python floats.
        """
        s, a, r, s_, done = replay.sample(BATCH_SIZE)

        # ---- Critic update -----------------------------------------------
        # Target Q = r + γ · Q_target(s', u_safe_target(s'))
        # We do NOT project through QP for target (target is used only in TD)
        with torch.no_grad():
            u_nom_  = self.actor_target(s_)
            u_safe_ = self._safe_action(s_, u_nom_)
            q_tgt   = r + GAMMA_RL * (1 - done) * self.critic_target(s_, u_safe_)

        q_pred   = self.critic(s, a)
        loss_crt = F.mse_loss(q_pred, q_tgt)

        self.opt_critic.zero_grad()
        loss_crt.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        # ---- Actor update (CDDP gradient flow) ---------------------------
        # The key proof:  gradients flow through QP projection into actor.
        #
        #   actor_loss = -Critic(s, project_cbf(s, Actor(s))).mean()
        #
        # autograd chain:
        #   ∂L/∂θ_actor = ∂L/∂Q · ∂Q/∂u_safe · ∂u_safe/∂u_nom · ∂u_nom/∂θ
        #
        # ∂u_safe/∂u_nom is defined everywhere by torch.where (both branches
        # are differentiable; the mask is treated as a stop-gradient constant).

        u_nom_actor  = self.actor(s)              # gradient enabled
        u_safe_actor = self._safe_action(s, u_nom_actor)   # QP — still in graph
        q_actor      = self.critic(s, u_safe_actor)
        loss_act     = -q_actor.mean()

        self.opt_actor.zero_grad()
        loss_act.backward()

        # ---- Verify gradient actually flowed into actor weights -----------
        # (We check once here; disable to save time after initial validation)
        for name, p in self.actor.named_parameters():
            if p.grad is None:
                raise RuntimeError(f"Actor gradient is None for param '{name}'! "
                                   "Graph is broken at the QP projection.")

        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        # ---- Polyak target update ----------------------------------------
        soft_update(self.actor_target,  self.actor,  TAU_SOFT)
        soft_update(self.critic_target, self.critic, TAU_SOFT)

        return loss_crt.item(), loss_act.item()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_env() -> DifferentiablePointMassEnv:
    return DifferentiablePointMassEnv(
        goal=GOAL,
        obstacle_c=OBS_C,
        obstacle_r=OBS_R,
        dt=DT,
        goal_tol=0.15,
        v_max=V_MAX_ENV,
        device=DEVICE,
    )


def random_start() -> torch.Tensor:
    """Sample a random start state outside the obstacle."""
    while True:
        p = torch.rand(2) * 2.5               # p ∈ [0, 2.5]^2
        h = (p - OBS_C).norm() ** 2 - OBS_R ** 2
        if h > 0.5:                            # ensure clear of obstacle
            v = torch.zeros(2)
            return torch.cat([p, v])           # (4,)


def compute_reward(
    pos: torch.Tensor,
    prev_pos: torch.Tensor,
    goal: torch.Tensor,
    h_val: float,
) -> float:
    """Dense reward shaping:
    - Progress toward goal
    - Small penalty for safety violations (h < 0)
    """
    dist_now  = (pos  - goal).norm().item()
    dist_prev = (prev_pos - goal).norm().item()
    progress  = dist_prev - dist_now          # positive when getting closer
    safety_pen = min(0.0, h_val) * 5.0        # penalize obstacle penetration
    goal_bonus = 5.0 if dist_now < 0.15 else 0.0
    return float(progress + safety_pen + goal_bonus)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train() -> None:
    print("=" * 65)
    print("  Phase 3: CDDP-DDPG  —  Differentiable QP Gradient Flow Proof")
    print(f"  Goal: {GOAL.tolist()}   Obstacle: c={OBS_C.tolist()} R={OBS_R}")
    print(f"  Stakes S={STAKES} (strict mode)   Episodes={N_EPISODES}")
    print("=" * 65)

    env     = make_env()
    agent   = CDDPAgent(env)
    replay  = ReplayBuffer(REPLAY_CAP)

    total_steps = 0
    noise_std   = NOISE_STD

    # Running stats for logging
    ep_returns:   list[float] = []
    ep_successes: list[bool]  = []
    ep_viols:     list[bool]  = []

    for ep in range(N_EPISODES):
        s0    = random_start()
        state = s0.clone().to(DEVICE)
        env.reset(state.unsqueeze(0))

        ep_ret  = 0.0
        success = False
        violated = False

        for step in range(MAX_STEPS):
            # ---- Select action -------------------------------------------
            if total_steps < WARMUP_STEPS:
                # Pure random safe action during warm-up
                u_nom_rnd = (torch.rand(2, device=DEVICE) * 2 - 1) * U_MAX
                s_t = env.state
                u_safe = agent._safe_action(
                    s_t, u_nom_rnd.unsqueeze(0)
                ).squeeze(0).detach()
            else:
                u_safe = agent.select_action(env.state.squeeze(0), noise_std)

            # ---- Step environment ----------------------------------------
            prev_pos = env.state.squeeze(0)[:2].clone()
            out      = env.step(u_safe.unsqueeze(0))

            next_state = out["state"].squeeze(0).detach()
            h_val      = out["h_cbf"].item()
            reached    = out["goal_reached"].item()
            unsafe     = out["unsafe"].item()

            reward = compute_reward(
                next_state[:2], prev_pos, GOAL, h_val
            )
            done = bool(reached) or (step == MAX_STEPS - 1)

            # ---- Store transition ----------------------------------------
            replay.push(
                env.state.squeeze(0).detach().clone(),  # s  (before step)
                u_safe.detach(),                         # a  (u_safe as applied)
                reward,
                next_state,
                done,
            )

            # (env.step already updated internal state via update_internal=True)

            ep_ret   += reward
            success   = success or bool(reached)
            violated  = violated or bool(unsafe)
            total_steps += 1

            # ---- Learn ---------------------------------------------------
            if total_steps >= WARMUP_STEPS and len(replay) >= BATCH_SIZE:
                agent.update(replay)

            if done:
                break

        ep_returns.append(ep_ret)
        ep_successes.append(success)
        ep_viols.append(violated)
        noise_std = max(NOISE_MIN, noise_std * NOISE_DECAY)

        # ---- Logging -----------------------------------------------------
        if (ep + 1) % LOG_INTERVAL == 0:
            window = slice(-LOG_INTERVAL, None)
            avg_ret    = sum(ep_returns[window])   / LOG_INTERVAL
            succ_rate  = sum(ep_successes[window]) / LOG_INTERVAL
            viol_rate  = sum(ep_viols[window])     / LOG_INTERVAL
            print(
                f"Ep {ep+1:>4}/{N_EPISODES}  |  "
                f"AvgReturn={avg_ret:+7.2f}  |  "
                f"Success={succ_rate:.0%}  |  "
                f"SafetyViol={viol_rate:.0%}  |  "
                f"Noise={noise_std:.3f}  |  "
                f"Steps={total_steps}"
            )

    # ---- Final evaluation ------------------------------------------------
    print("\n" + "=" * 65)
    print("  Final Evaluation (10 episodes, no noise)")
    print("=" * 65)
    eval_success = 0
    eval_viol    = 0
    for _ in range(10):
        state = random_start().to(DEVICE)
        env.reset(state.unsqueeze(0))
        for _ in range(MAX_STEPS):
            u_safe = agent.select_action(env.state.squeeze(0), noise_std=0)
            out    = env.step(u_safe.unsqueeze(0))
            if out["goal_reached"].item():
                eval_success += 1
                break
            if out["unsafe"].item():
                eval_viol += 1
                break

    print(f"  Success: {eval_success}/10   |   Safety violations: {eval_viol}/10")

    # ---- Gradient flow final verification --------------------------------
    print("\n" + "=" * 65)
    print("  Gradient Flow Verification")
    print("=" * 65)
    s_test = random_start().unsqueeze(0).to(DEVICE)
    env.reset(s_test)
    u_nom  = agent.actor(s_test)
    stakes = torch.zeros(1, device=DEVICE)
    u_safe = env.project_action_cbf(
        s_test, u_nom, stakes=stakes,
        gamma=GAMMA_CBF, tau=TAU_CDDP, alpha_prime=ALPHA_CDDP,
    )
    q_val  = agent.critic(s_test, u_safe)
    loss   = -q_val.mean()
    loss.backward()

    all_grads_ok = True
    for name, p in agent.actor.named_parameters():
        if p.grad is None or p.grad.isnan().any():
            print(f"  ✗  BROKEN gradient at actor param: {name}")
            all_grads_ok = False

    if all_grads_ok:
        sample_grad = next(agent.actor.parameters()).grad
        print(f"  ✓  Gradients flow cleanly through:")
        print(f"        Critic -> u_safe -> torch.where (QP) -> u_nom -> Actor weights")
        print(f"     Sample actor.net[0].weight.grad norm = "
              f"{sample_grad.norm():.6f}")
    else:
        print("  ✗  Gradient flow BROKEN. Check QP projection implementation.")

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    train()
