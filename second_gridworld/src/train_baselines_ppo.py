"""
train_baselines_ppo.py
======================
On-policy PPO-family baselines for the batched CDDP point-mass sandbox.

Algorithms:
  ppo         Standard unconstrained PPO.
  strict      PPO with strict HOCBF/CBF shielding before every environment step.
  lagrangian  PPO-Lagrangian with an adaptive unsafe-step multiplier.
  cpo         TRPO-Lagrangian equivalent with a KL-constrained policy step.

Run examples from the repo root:
  python3 src/train_baselines_ppo.py --algo ppo
  python3 src/train_baselines_ppo.py --algo strict
  python3 src/train_baselines_ppo.py --algo lagrangian
  python3 src/train_baselines_ppo.py --algo cpo
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, kl_divergence

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

import src.train_sandbox_cddp as cddp


DEVICE = cddp.DEVICE
DEFAULT_NUM_ENVS = 128
STATE_DIM = 5
ACTION_DIM = 2
U_MAX = cddp.U_MAX
RESULTS_ROOT = cddp.RESULTS_ROOT
BASELINE_ALGOS = {"ppo", "strict", "lagrangian", "cpo"}


def result_dir_for_algo(algo: str) -> str:
    if algo not in BASELINE_ALGOS:
        raise ValueError(f"Unsupported baseline algo '{algo}'.")
    return os.path.join(RESULTS_ROOT, algo)


def checkpoint_dir_for_algo(algo: str) -> str:
    return os.path.join(result_dir_for_algo(algo), "checkpoints")


def checkpoint_path_for_algo(algo: str) -> str:
    return os.path.join(checkpoint_dir_for_algo(algo), f"{algo}_actor_critic.pt")


CHECKPOINT_DIR = RESULTS_ROOT


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int) -> nn.Sequential:
    dims = [in_dim] + [hidden] * layers + [out_dim]
    mods: list[nn.Module] = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.Tanh())
    return nn.Sequential(*mods)


class GaussianActor(nn.Module):
    """Diagonal Gaussian policy with tanh-bounded mean."""

    def __init__(self, hidden: int, layers: int) -> None:
        super().__init__()
        self.net = _mlp(STATE_DIM, ACTION_DIM, hidden, layers)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -0.5))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return U_MAX * torch.tanh(self.net(obs))

    def dist(self, obs: torch.Tensor) -> Independent:
        mean = self(obs)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Independent(Normal(mean, std), 1)


class ValueNet(nn.Module):
    def __init__(self, hidden: int, layers: int) -> None:
        super().__init__()
        self.net = _mlp(STATE_DIM, 1, hidden, layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class LagrangeMultiplier(nn.Module):
    """Positive lambda with dual gradient ascent implemented as descent on -lambda*violation."""

    def __init__(self, init_value: float, lr: float, cost_limit: float) -> None:
        super().__init__()
        init_raw = math.log(math.exp(max(init_value, 1e-6)) - 1.0)
        self.raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32, device=DEVICE))
        self.optimizer = torch.optim.Adam([self.raw], lr=lr)
        self.cost_limit = float(cost_limit)

    @property
    def value(self) -> torch.Tensor:
        return F.softplus(self.raw)

    def update(self, observed_episode_cost: float) -> float:
        violation = torch.tensor(
            observed_episode_cost - self.cost_limit,
            dtype=torch.float32,
            device=DEVICE,
        )
        loss = -(self.value * violation.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.raw.clamp_(-20.0, 10.0)
        return float(self.value.detach().cpu().item())


class ActorCritic(nn.Module):
    def __init__(self, hidden: int, layers: int) -> None:
        super().__init__()
        self.actor = GaussianActor(hidden, layers)
        self.reward_value = ValueNet(hidden, layers)
        self.cost_value = ValueNet(hidden, layers)

    @torch.no_grad()
    def act_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs).clamp(-U_MAX, U_MAX)


class RolloutBuffer:
    def __init__(self, rollout_steps: int, num_envs: int, device: torch.device) -> None:
        self.rollout_steps = int(rollout_steps)
        self.num_envs = int(num_envs)
        self.device = device
        self.obs = torch.empty((rollout_steps, num_envs, STATE_DIM), dtype=torch.float32, device=device)
        self.actions = torch.empty((rollout_steps, num_envs, ACTION_DIM), dtype=torch.float32, device=device)
        self.log_probs = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.costs = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.reward_values = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.cost_values = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.reward_adv = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.cost_adv = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.reward_returns = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.cost_returns = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)

    def add(
        self,
        step: int,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        reward_value: torch.Tensor,
        cost_value: torch.Tensor,
    ) -> None:
        self.obs[step] = obs.detach()
        self.actions[step] = action.detach()
        self.log_probs[step] = log_prob.detach()
        self.rewards[step] = reward.detach()
        self.costs[step] = cost.detach()
        self.dones[step] = done.detach().to(dtype=torch.float32)
        self.reward_values[step] = reward_value.detach()
        self.cost_values[step] = cost_value.detach()

    def compute_gae(
        self,
        last_reward_value: torch.Tensor,
        last_cost_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        reward_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        cost_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_reward_value = last_reward_value
                next_cost_value = last_cost_value
            else:
                next_reward_value = self.reward_values[t + 1]
                next_cost_value = self.cost_values[t + 1]

            next_nonterminal = 1.0 - self.dones[t]
            reward_delta = self.rewards[t] + gamma * next_reward_value * next_nonterminal - self.reward_values[t]
            cost_delta = self.costs[t] + gamma * next_cost_value * next_nonterminal - self.cost_values[t]
            reward_gae = reward_delta + gamma * gae_lambda * next_nonterminal * reward_gae
            cost_gae = cost_delta + gamma * gae_lambda * next_nonterminal * cost_gae
            self.reward_adv[t] = reward_gae
            self.cost_adv[t] = cost_gae

        self.reward_returns = self.reward_adv + self.reward_values
        self.cost_returns = self.cost_adv + self.cost_values

    def flat(self) -> dict[str, torch.Tensor]:
        return {
            "obs": self.obs.reshape(-1, STATE_DIM),
            "actions": self.actions.reshape(-1, ACTION_DIM),
            "log_probs": self.log_probs.reshape(-1),
            "reward_adv": self.reward_adv.reshape(-1),
            "cost_adv": self.cost_adv.reshape(-1),
            "reward_returns": self.reward_returns.reshape(-1),
            "cost_returns": self.cost_returns.reshape(-1),
        }


@dataclass
class EpisodeStats:
    returns: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    crashes: list[bool] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)

    def extend(self, batch: Iterable[dict[str, object]]) -> None:
        for item in batch:
            self.returns.append(float(item["return"]))
            self.costs.append(float(item["cost"]))
            self.successes.append(bool(item["success"]))
            self.crashes.append(bool(item["crash"]))
            self.lengths.append(int(item["length"]))


class BatchedRunner:
    """Owns the fully batched env state and auto-resets completed slots."""

    def __init__(self, num_envs: int, fixed_max_steps: int | None) -> None:
        self.num_envs = int(num_envs)
        self.fixed_max_steps = fixed_max_steps
        self.env = cddp.make_env()
        self.env.reset(
            torch.zeros(self.num_envs, STATE_DIM, dtype=torch.float32, device=DEVICE),
            max_steps=cddp.LONG_CLOCK_STEPS,
        )
        self.current_goals = cddp.GOAL.to(DEVICE).unsqueeze(0).expand(self.num_envs, -1).clone()
        self.per_env_steps = torch.zeros(self.num_envs, dtype=torch.long, device=DEVICE)
        self.episode_max_steps = torch.full(
            (self.num_envs,),
            cddp.LONG_CLOCK_STEPS,
            dtype=torch.long,
            device=DEVICE,
        )
        self.current_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=DEVICE)
        self.current_costs = torch.zeros(self.num_envs, dtype=torch.float32, device=DEVICE)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.bool, device=DEVICE)
        self.current_crashed = torch.zeros(self.num_envs, dtype=torch.bool, device=DEVICE)
        self.reset_slots(torch.arange(self.num_envs, dtype=torch.long, device=DEVICE))

    def reset_slots(self, slots: torch.Tensor) -> None:
        if slots.numel() == 0:
            return
        n = int(slots.numel())
        starts, goals = cddp.sample_episode_context(n)
        if self.fixed_max_steps is None:
            clocks = cddp.sample_episode_steps(n)
        else:
            clocks = torch.full((n,), int(self.fixed_max_steps), dtype=torch.long, device=DEVICE)
        starts[:, 4] = clocks.to(dtype=starts.dtype) / 1000.0

        self.env._state[slots] = starts
        self.current_goals[slots] = goals
        self.env.set_goal(self.current_goals)
        if self.env.elapsed_steps is not None:
            self.env.elapsed_steps[slots] = 0.0
        self.per_env_steps[slots] = 0
        self.episode_max_steps[slots] = clocks
        self.current_returns[slots] = 0.0
        self.current_costs[slots] = 0.0
        self.current_successes[slots] = False
        self.current_crashed[slots] = False

    def project_if_needed(self, algo: str, state: torch.Tensor, nominal_action: torch.Tensor) -> torch.Tensor:
        if algo != "strict":
            return nominal_action.clamp(-U_MAX, U_MAX)
        stakes = torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)
        return self.env.project_action_cbf(
            state,
            nominal_action,
            stakes=stakes,
            gamma=cddp.GAMMA_CBF,
            tau=cddp.TAU_CDDP,
            alpha_prime=cddp.ALPHA_CDDP,
        )

    def step(self, algo: str, nominal_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, object]]]:
        state = self.env.state.detach().clone()
        prev_pos = state[:, :2].clone()
        env_action = self.project_if_needed(algo, state, nominal_action)
        out = self.env.step(env_action)

        next_state = out["state"].detach().clone()
        self.per_env_steps += 1
        remaining = (self.episode_max_steps - self.per_env_steps).clamp_min(0)
        next_state[:, 4] = remaining.to(dtype=next_state.dtype) / 1000.0
        self.env._state = next_state
        if self.env.elapsed_steps is not None:
            self.env.elapsed_steps[:] = self.per_env_steps.to(dtype=self.env.elapsed_steps.dtype)

        reached = cddp.success_mask(next_state[:, :2], self.current_goals, out["goal_reached"].detach())
        unsafe = out["unsafe"].detach()
        timeout = (self.per_env_steps >= self.episode_max_steps) & ~reached
        done = reached | timeout
        reward = cddp.compute_reward(
            next_state[:, :2],
            prev_pos,
            self.current_goals,
            reached,
            unsafe,
            timeout,
            timestep=self.per_env_steps,
        ).to(dtype=torch.float32)
        cost = unsafe.to(dtype=torch.float32)

        self.current_returns += reward
        self.current_costs += cost
        self.current_successes |= reached
        self.current_crashed |= unsafe

        finished: list[dict[str, object]] = []
        done_slots = torch.where(done)[0]
        for slot_t in done_slots:
            slot = int(slot_t.item())
            finished.append({
                "return": float(self.current_returns[slot].detach().cpu().item()),
                "cost": float(self.current_costs[slot].detach().cpu().item()),
                "success": bool(self.current_successes[slot].detach().cpu().item()),
                "crash": bool(self.current_crashed[slot].detach().cpu().item()),
                "length": int(self.per_env_steps[slot].detach().cpu().item()),
            })
        self.reset_slots(done_slots)
        return reward, cost, done, finished


def collect_rollout(
    args: argparse.Namespace,
    agent: ActorCritic,
    runner: BatchedRunner,
) -> tuple[RolloutBuffer, list[dict[str, object]]]:
    buffer = RolloutBuffer(args.rollout_steps, args.num_envs, DEVICE)
    finished: list[dict[str, object]] = []

    for step in range(args.rollout_steps):
        obs = runner.env.state.detach().clone()
        with torch.no_grad():
            dist = agent.actor.dist(obs)
            raw_action = dist.sample()
            nominal_action = raw_action.clamp(-U_MAX, U_MAX)
            log_prob = dist.log_prob(raw_action)
            reward_value = agent.reward_value(obs)
            cost_value = agent.cost_value(obs)

        reward, cost, done, done_infos = runner.step(args.algo, nominal_action)
        buffer.add(step, obs, raw_action, log_prob, reward, cost, done, reward_value, cost_value)
        finished.extend(done_infos)

    with torch.no_grad():
        last_obs = runner.env.state.detach()
        last_reward_value = agent.reward_value(last_obs)
        last_cost_value = agent.cost_value(last_obs)
    buffer.compute_gae(last_reward_value, last_cost_value, args.gamma, args.gae_lambda)
    return buffer, finished


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def minibatches(n: int, batch_size: int) -> Iterable[torch.Tensor]:
    indices = torch.randperm(n, device=DEVICE)
    for start in range(0, n, batch_size):
        yield indices[start : start + batch_size]


def update_ppo(
    args: argparse.Namespace,
    agent: ActorCritic,
    actor_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    lambda_value: float,
) -> dict[str, float]:
    data = buffer.flat()
    obs = data["obs"]
    actions = data["actions"]
    old_log_probs = data["log_probs"]

    # Combine advantages BEFORE normalizing so that lambda_value has genuine
    # leverage over the combined signal.  Double-normalizing (normalizing each
    # term independently and then normalizing again) erases the relative scaling
    # and makes the multiplier mathematically inert.
    if args.algo == "lagrangian":
        policy_adv = normalize(data["reward_adv"] - lambda_value * data["cost_adv"])
    else:
        policy_adv = normalize(data["reward_adv"])

    n = obs.shape[0]
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0
    last_kl = 0.0

    for _ in range(args.update_epochs):
        for idx in minibatches(n, args.batch_size):
            dist = agent.actor.dist(obs[idx])
            new_log_probs = dist.log_prob(actions[idx])
            entropy = dist.entropy().mean()
            log_ratio = new_log_probs - old_log_probs[idx]
            ratio = log_ratio.exp()
            unclipped = ratio * policy_adv[idx]
            clipped = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * policy_adv[idx]
            policy_loss = -torch.min(unclipped, clipped).mean() - args.entropy_coef * entropy

            actor_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
            actor_optimizer.step()

            reward_pred = agent.reward_value(obs[idx])
            value_loss = F.mse_loss(reward_pred, data["reward_returns"][idx])
            if args.algo == "lagrangian":
                cost_pred = agent.cost_value(obs[idx])
                value_loss = value_loss + F.mse_loss(cost_pred, data["cost_returns"][idx])
            value_loss = value_loss * args.value_coef

            value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                list(agent.reward_value.parameters()) + list(agent.cost_value.parameters()),
                args.max_grad_norm,
            )
            value_optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
            last_policy_loss = float(policy_loss.detach().cpu().item())
            last_value_loss = float(value_loss.detach().cpu().item())
            last_entropy = float(entropy.detach().cpu().item())
            last_kl = float(approx_kl.detach().cpu().item())

    return {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "entropy": last_entropy,
        "kl": last_kl,
    }


def flat_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def set_flat_params(module: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat[offset : offset + n].view_as(p))
        offset += n


def flat_grad(grads: Iterable[torch.Tensor | None], params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(grad.reshape(-1))
    return torch.cat(pieces)


def conjugate_gradients(fvp_fn, b: torch.Tensor, steps: int, residual_tol: float) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(steps):
        avp = fvp_fn(p)
        alpha = rdotr / (torch.dot(p, avp) + 1e-8)
        x = x + alpha * p
        r = r - alpha * avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x


def update_values(
    args: argparse.Namespace,
    agent: ActorCritic,
    value_optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
) -> float:
    data = buffer.flat()
    obs = data["obs"]
    n = obs.shape[0]
    last_loss = 0.0
    for _ in range(args.update_epochs):
        for idx in minibatches(n, args.batch_size):
            reward_pred = agent.reward_value(obs[idx])
            cost_pred = agent.cost_value(obs[idx])
            value_loss = (
                F.mse_loss(reward_pred, data["reward_returns"][idx])
                + F.mse_loss(cost_pred, data["cost_returns"][idx])
            ) * args.value_coef
            value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                list(agent.reward_value.parameters()) + list(agent.cost_value.parameters()),
                args.max_grad_norm,
            )
            value_optimizer.step()
            last_loss = float(value_loss.detach().cpu().item())
    return last_loss


def update_trpo_lagrangian(
    args: argparse.Namespace,
    agent: ActorCritic,
    value_optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    lambda_value: float,
) -> dict[str, float]:
    value_loss = update_values(args, agent, value_optimizer, buffer)
    data = buffer.flat()
    obs = data["obs"]
    actions = data["actions"]
    old_log_probs = data["log_probs"]

    # Combine raw advantages before the single normalize so lambda has real leverage.
    policy_adv = normalize(data["reward_adv"] - lambda_value * data["cost_adv"])

    params = list(agent.actor.parameters())
    with torch.no_grad():
        old_dist = agent.actor.dist(obs)

    def surrogate() -> torch.Tensor:
        dist = agent.actor.dist(obs)
        return ((dist.log_prob(actions) - old_log_probs).exp() * policy_adv).mean()

    surr_before = surrogate().detach()
    policy_grads = torch.autograd.grad(surrogate(), params)
    g = flat_grad(policy_grads, params).detach()

    def fvp(vector: torch.Tensor) -> torch.Tensor:
        dist = agent.actor.dist(obs)
        kl = kl_divergence(old_dist, dist).mean()
        grad_kl = torch.autograd.grad(kl, params, create_graph=True)
        flat_grad_kl = flat_grad(grad_kl, params)
        kl_v = (flat_grad_kl * vector).sum()
        grad2 = torch.autograd.grad(kl_v, params)
        return flat_grad(grad2, params).detach() + args.cg_damping * vector

    step_dir = conjugate_gradients(fvp, g, args.cg_iters, args.cg_residual_tol)
    shs = 0.5 * torch.dot(step_dir, fvp(step_dir))
    if not torch.isfinite(shs) or shs.item() <= 0.0:
        return {"policy_loss": float(-surr_before.cpu().item()), "value_loss": value_loss, "entropy": 0.0, "kl": 0.0}

    step_scale = torch.sqrt(torch.tensor(args.target_kl, device=DEVICE) / (shs + 1e-8))
    full_step = step_dir * step_scale
    old_params = flat_params(agent.actor)
    accepted = False
    final_kl = torch.tensor(0.0, device=DEVICE)
    final_surr = surr_before

    for fraction in [0.5 ** i for i in range(args.line_search_steps)]:
        candidate = old_params + fraction * full_step
        set_flat_params(agent.actor, candidate)
        with torch.no_grad():
            dist = agent.actor.dist(obs)
            final_kl = kl_divergence(old_dist, dist).mean()
            final_surr = ((dist.log_prob(actions) - old_log_probs).exp() * policy_adv).mean()
        if final_kl <= args.target_kl and final_surr >= surr_before:
            accepted = True
            break

    if not accepted:
        set_flat_params(agent.actor, old_params)
        final_kl = torch.tensor(0.0, device=DEVICE)
        final_surr = surr_before

    with torch.no_grad():
        entropy = agent.actor.dist(obs).entropy().mean()
    return {
        "policy_loss": float((-final_surr).detach().cpu().item()),
        "value_loss": value_loss,
        "entropy": float(entropy.detach().cpu().item()),
        "kl": float(final_kl.detach().cpu().item()),
    }


@torch.no_grad()
def rollout_eval_episode(
    agent: ActorCritic,
    algo: str,
    env: object,
    start_state: torch.Tensor,
    goal: torch.Tensor,
    clock_max_steps: int,
    agent_profile: str,
    seed: int,
    context_id: str,
) -> dict[str, object]:
    start_batch = start_state.reshape(1, STATE_DIM).to(DEVICE).clone()
    goal_batch = goal.reshape(1, 2).to(DEVICE).clone()
    start_batch[:, 4] = float(clock_max_steps) / 1000.0
    env.reset(start_batch, max_steps=clock_max_steps)
    env.set_goal(goal_batch)
    if env.elapsed_steps is not None:
        env.elapsed_steps[:] = 0.0

    start_xy = tuple(float(v) for v in start_batch[0, :2].detach().cpu().tolist())
    goal_xy = tuple(float(v) for v in goal_batch[0].detach().cpu().tolist())
    constraint_violations: list[dict[str, object]] = []
    traj = [start_xy]
    stakes_by_step: list[float] = []
    total_reward = 0.0
    total_steps = 0
    episode_success = False

    for step_idx in range(clock_max_steps):
        state = env.state.detach().clone()
        prev_pos = state[:, :2].clone()
        nominal_action = agent.act_mean(state)
        if algo == "strict":
            action = env.project_action_cbf(
                state,
                nominal_action,
                stakes=torch.zeros(1, dtype=state.dtype, device=state.device),
                gamma=cddp.GAMMA_CBF,
                tau=cddp.TAU_CDDP,
                alpha_prime=cddp.ALPHA_CDDP,
            )
        else:
            action = nominal_action.clamp(-U_MAX, U_MAX)
        out = env.step(action)

        next_state = out["state"].detach().clone()
        remaining_steps = max(0, clock_max_steps - (step_idx + 1))
        next_state[:, 4] = float(remaining_steps) / 1000.0
        env._state = next_state
        if env.elapsed_steps is not None:
            env.elapsed_steps[:] = float(step_idx + 1)

        reached = cddp.success_mask(next_state[:, :2], goal_batch, out["goal_reached"].detach())
        episode_success = episode_success or bool(reached.detach().cpu().item())
        timeout = torch.tensor([step_idx + 1 >= clock_max_steps], dtype=torch.bool, device=DEVICE) & ~reached
        reward = cddp.compute_reward(
            next_state[:, :2],
            prev_pos,
            goal_batch,
            reached,
            out["unsafe"].detach(),
            timeout,
            timestep=torch.tensor([step_idx + 1], dtype=torch.float32, device=DEVICE),
        )
        total_reward += float(reward.detach().cpu().item())
        total_steps = step_idx + 1

        next_xy = tuple(float(v) for v in next_state[0, :2].detach().cpu().tolist())
        traj.append(next_xy)
        stakes_by_step.append(0.0)
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
    record["total_steps"] = total_steps
    return record


def save_evaluation_logs(algo: str, records: list[dict[str, object]]) -> tuple[str, str]:
    results_dir = result_dir_for_algo(algo)
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f"{algo}_eval_log.json")
    csv_path = os.path.join(results_dir, f"{algo}_eval_log.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    fields = [
        "seed",
        "context_id",
        "agent_profile",
        "timer",
        "max_steps",
        "start_x",
        "start_y",
        "goal_x",
        "goal_y",
        "success",
        "total_steps",
        "total_violations",
        "total_reward",
        "final_distance_to_goal",
        "min_distance_to_goal",
        "first_step_inside_goal_zone",
        "first_step_inside_goal_tol",
        "steps_after_first_entering_goal_zone",
        "steps_dist_lt_0_5",
        "steps_dist_lt_1_0",
        "path_length",
        "straight_line_distance",
        "path_efficiency",
        "avg_stakes",
        "max_stakes",
        "frac_stakes_gt_0_8",
        "constraint_violations",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in fields}
            row["constraint_violations"] = json.dumps(record.get("constraint_violations", []))
            writer.writerow(row)
    return json_path, csv_path


def evaluate_and_save(agent: ActorCritic, algo: str, seed: int) -> tuple[str, str]:
    env = cddp.make_env()
    start_states, goals = cddp.sample_episode_context(1, seed=seed)
    start_state = start_states[0].detach().clone()
    goal = goals[0].detach().clone()
    tight_steps, long_steps = cddp.eval_clock_config()
    label = {
        "ppo": "PPO",
        "strict": "Strict CMDP",
        "lagrangian": "PPO-Lagrangian",
        "cpo": "TRPO-Lagrangian",
    }[algo]
    context_id = f"seed-{seed}"
    records = [
        rollout_eval_episode(
            agent,
            algo,
            env,
            start_state,
            goal,
            tight_steps,
            f"{label} Tight Clock Agent",
            seed,
            context_id,
        ),
        rollout_eval_episode(
            agent,
            algo,
            env,
            start_state,
            goal,
            long_steps,
            f"{label} Long Clock Agent",
            seed,
            context_id,
        ),
    ]
    return save_evaluation_logs(algo, records)


def save_checkpoint(agent: ActorCritic, algo: str) -> str:
    path = checkpoint_path_for_algo(algo)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(agent.state_dict(), path)
    return path


def mean_tail(values: list[float], n: int) -> float:
    if not values:
        return float("nan")
    tail = values[-n:]
    return sum(tail) / len(tail)


def train(args: argparse.Namespace) -> None:
    args.num_envs = max(1, int(args.num_envs))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cddp.GOAL_TOL = float(args.goal_tol)
    cddp.TERMINATE_ON_GOAL_ZONE = bool(args.terminate_on_goal_zone)
    cddp.TIMER_CURRICULUM = args.timer_curriculum
    cddp.GOAL_CAPTURE_SHAPING = not bool(args.disable_goal_capture_shaping)
    cddp.GOAL_DISTANCE_PENALTY = float(args.goal_distance_penalty)
    cddp.NEAR_GOAL_RADIUS = float(args.near_goal_radius)
    cddp.NEAR_GOAL_BONUS_SCALE = float(args.near_goal_bonus_scale)
    cddp.ACTION_PENALTY = float(args.action_penalty)

    agent = ActorCritic(args.hidden, args.layers).to(DEVICE)
    runner = BatchedRunner(args.num_envs, args.fixed_max_steps)
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=args.actor_lr)
    value_optimizer = torch.optim.Adam(
        list(agent.reward_value.parameters()) + list(agent.cost_value.parameters()),
        lr=args.value_lr,
    )
    lagrange = None
    if args.algo in ("lagrangian", "cpo"):
        lagrange = LagrangeMultiplier(args.lambda_init, args.lambda_lr, args.cost_limit)

    stats = EpisodeStats()
    update_idx = 0
    total_steps = 0
    started_at = time.perf_counter()
    label = {
        "ppo": "Standard PPO",
        "strict": "Strict CMDP PPO",
        "lagrangian": "PPO-Lagrangian",
        "cpo": "TRPO-Lagrangian",
    }[args.algo]

    print("=" * 78)
    print(f"  {label} baseline on DifferentiablePointMassEnv")
    print(
        f"  Episodes={args.episodes}  VectorEnvs={args.num_envs}  "
        f"RolloutSteps={args.rollout_steps}  Device={DEVICE}"
    )
    print(
        f"  ActorCritic hidden={args.hidden} layers={args.layers}  "
        f"gamma={args.gamma} gae_lambda={args.gae_lambda}"
    )
    print(
        f"  Success={'goal zone' if cddp.TERMINATE_ON_GOAL_ZONE else 'sampled goal tolerance'}  "
        f"GoalTol={cddp.GOAL_TOL:.3f}  TimerCurriculum={cddp.TIMER_CURRICULUM}"
    )
    print(
        "  Reward constants: "
        f"goal_capture_shaping={cddp.GOAL_CAPTURE_SHAPING} "
        f"goal_distance_penalty={cddp.GOAL_DISTANCE_PENALTY} "
        f"near_goal_radius={cddp.NEAR_GOAL_RADIUS} "
        f"near_goal_bonus_scale={cddp.NEAR_GOAL_BONUS_SCALE}"
    )
    if lagrange is not None:
        print(f"  Constraint: unsafe-step episode cost limit={args.cost_limit}")
    print("=" * 78)

    last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
    while len(stats.returns) < args.episodes:
        update_idx += 1
        buffer, finished = collect_rollout(args, agent, runner)
        stats.extend(finished)
        total_steps += args.rollout_steps * args.num_envs

        lambda_value = 0.0
        observed_cost = 0.0
        if lagrange is not None:
            if finished:
                observed_cost = sum(float(ep["cost"]) for ep in finished) / len(finished)
            else:
                observed_cost = float(buffer.costs.sum(dim=0).mean().detach().cpu().item())
            lambda_value = lagrange.update(observed_cost)

        if args.algo == "cpo":
            last_losses = update_trpo_lagrangian(args, agent, value_optimizer, buffer, lambda_value)
        else:
            last_losses = update_ppo(args, agent, actor_optimizer, value_optimizer, buffer, lambda_value)

        if len(stats.returns) >= args.next_log or update_idx % args.log_every_updates == 0:
            while args.next_log <= len(stats.returns):
                args.next_log += args.log_interval
            recent = min(args.log_interval, max(1, len(stats.returns)))
            success_rate = sum(stats.successes[-recent:]) / recent if stats.successes else 0.0
            crash_rate = sum(stats.crashes[-recent:]) / recent if stats.crashes else 0.0
            elapsed = max(1e-9, time.perf_counter() - started_at)
            rate = total_steps / elapsed
            lambda_note = ""
            if lagrange is not None:
                lambda_note = f" | lambda={lambda_value:.3f} cost={observed_cost:.2f}"
            print(
                f"Ep {len(stats.returns):>5}/{args.episodes} | "
                f"Ret={mean_tail(stats.returns, recent):+8.2f} | "
                f"Cost={mean_tail(stats.costs, recent):6.2f} | "
                f"Succ={success_rate:.0%} Crash={crash_rate:.0%} | "
                f"KL={last_losses['kl']:.4f} Ent={last_losses['entropy']:.3f} | "
                f"Lpi={last_losses['policy_loss']:.3f} Lv={last_losses['value_loss']:.3f} | "
                f"Rate={rate:.0f}steps/s{lambda_note}"
            )

    checkpoint_path = save_checkpoint(agent, args.algo)
    print(f"Saved checkpoint to: {checkpoint_path}")
    if not args.no_eval:
        json_path, csv_path = evaluate_and_save(agent, args.algo, args.eval_seed)
        print(f"Saved evaluation logs to: {json_path} and {csv_path}")
    print(f"{label} baseline complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO-family baselines for the CDDP sandbox.")
    parser.add_argument("--algo", choices=["ppo", "strict", "lagrangian", "cpo"], required=True)
    parser.add_argument("--episodes", type=int, default=15000)  # was 5000; lag/CPO need more data to stabilize lambda
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=cddp.MODEL_HIDDEN_DIM)
    parser.add_argument("--layers", type=int, default=cddp.MODEL_HIDDEN_LAYERS)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=cddp.GAMMA_RL)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)       # was 0.01; too tight → almost all TRPO steps rejected
    parser.add_argument("--cg-iters", type=int, default=10)
    parser.add_argument("--cg-damping", type=float, default=0.01)      # was 0.1; high damping degrades natural gradient quality
    parser.add_argument("--cg-residual-tol", type=float, default=1e-10)
    parser.add_argument("--line-search-steps", type=int, default=15)   # was 10; more chances to find an acceptable step
    parser.add_argument("--cost-limit", type=float, default=5.0)       # was 0.0; zero is unreachable → lambda diverges immediately
    parser.add_argument("--lambda-init", type=float, default=0.1)      # was 1.0; high init → constraint dominates before any learning
    parser.add_argument("--lambda-lr", type=float, default=1e-2)       # was 5e-2; too fast → lambda oscillates wildly
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=cddp.EVAL_SEED)
    parser.add_argument("--goal-tol", type=float, default=cddp.GOAL_TOL)
    parser.add_argument("--terminate-on-goal-zone", action="store_true")
    parser.add_argument("--timer-curriculum", choices=["bimodal", "mixed"], default=cddp.TIMER_CURRICULUM)
    parser.add_argument("--disable-goal-capture-shaping", action="store_true")
    parser.add_argument("--goal-distance-penalty", type=float, default=cddp.GOAL_DISTANCE_PENALTY)
    parser.add_argument("--near-goal-radius", type=float, default=cddp.NEAR_GOAL_RADIUS)
    parser.add_argument("--near-goal-bonus-scale", type=float, default=cddp.NEAR_GOAL_BONUS_SCALE)
    parser.add_argument("--action-penalty", type=float, default=cddp.ACTION_PENALTY)
    parser.add_argument(
        "--fixed-max-steps",
        type=int,
        default=None,
        help="Optional fixed episode horizon, mainly for quick smoke tests.",
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--log-every-updates", type=int, default=10)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    args.next_log = args.log_interval
    return args


if __name__ == "__main__":
    train(parse_args())
