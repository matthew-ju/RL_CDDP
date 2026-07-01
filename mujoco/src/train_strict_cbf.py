"""
train_strict_cbf.py
===================
Strict CBF baseline — DDPG with a hard CBF projection (S=0 always).

Expected behaviour: The strict CBF forces the agent around all obstacles.
With ample time it safely navigates the detour. But when the timer is very
short, it runs out of time taking the safe path, suffering the -500 catastrophe.
→ Zero violations, but high catastrophe rate at short timers.

Usage:
  python src/train_strict_cbf.py [--episodes N] [--max-steps T]
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC = Path(__file__).parent
_MJ_ROOT = _SRC.parent
sys.path.insert(0, str(_MJ_ROOT))
sys.path.insert(0, str(_SRC))

from src.crowd_env import CrowdNavEnv, STATE_DIM
from src.networks import Actor, UtilitarianCritic, soft_update
from src.cbf_layer import project_action_cbf
from src.replay_buffer import ReplayBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

GAMMA_RL    = 0.99
TAU_SOFT    = 0.005
LR_ACTOR    = 3e-4
LR_CRITIC   = 1e-3
BATCH_SIZE  = 256
REPLAY_CAP  = 300_000
WARMUP      = 2_000
TRAIN_EVERY = 2
GRAD_STEPS  = 2
EXPL_NOISE  = 0.15
EXPL_DECAY  = 0.9999
EXPL_MIN    = 0.05
GAMMA_CBF   = 1.5
LOG_INTERVAL = 50


class StrictCBFAgent:
    """DDPG + hard CBF projection (stakes always zero)."""

    def __init__(self, state_dim: int, action_dim: int, act_limit: float,
                 obs_centers: np.ndarray, obs_radii: np.ndarray,
                 device: torch.device) -> None:
        self.device = device
        self.act_limit = act_limit
        self.expl_noise = EXPL_NOISE
        self.obs_centers = torch.tensor(obs_centers, dtype=torch.float32, device=device)
        self.obs_radii   = torch.tensor(obs_radii,   dtype=torch.float32, device=device)

        self.actor         = Actor(state_dim, action_dim, act_limit).to(device)
        self.actor_target  = Actor(state_dim, action_dim, act_limit).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.requires_grad_(False)

        self.critic        = UtilitarianCritic(state_dim, action_dim).to(device)
        self.critic_target = UtilitarianCritic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        u_nom = self.actor(s)
        noise = torch.randn_like(u_nom) * self.expl_noise
        u_nom = (u_nom + noise).clamp(-self.act_limit, self.act_limit)
        pos   = s[:, :2]
        stakes = torch.zeros(1, device=self.device)  # Always strict
        u_safe = project_action_cbf(
            pos, u_nom, self.obs_centers, self.obs_radii,
            stakes=stakes, gamma=GAMMA_CBF,
        )
        return u_safe.squeeze(0).cpu().numpy()

    def update(self, buf: ReplayBuffer) -> tuple[float, float]:
        s, a, r, s_, done = buf.sample(BATCH_SIZE)
        stakes_ = torch.zeros(s_.shape[0], device=self.device)
        with torch.no_grad():
            u_nom_ = self.actor_target(s_)
            u_safe_ = project_action_cbf(
                s_[:, :2], u_nom_, self.obs_centers, self.obs_radii,
                stakes=stakes_, gamma=GAMMA_CBF,
            )
            q_tgt = r + GAMMA_RL * (1.0 - done) * self.critic_target(s_, u_safe_)

        loss_q = F.mse_loss(self.critic(s, a), q_tgt)
        self.opt_critic.zero_grad(); loss_q.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        stakes_a = torch.zeros(s.shape[0], device=self.device)
        u_nom_a  = self.actor(s)
        u_safe_a = project_action_cbf(
            s[:, :2], u_nom_a, self.obs_centers, self.obs_radii,
            stakes=stakes_a, gamma=GAMMA_CBF,
        )
        loss_a = -self.critic(s, u_safe_a).mean()
        self.opt_actor.zero_grad(); loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        soft_update(self.actor_target, self.actor, TAU_SOFT)
        soft_update(self.critic_target, self.critic, TAU_SOFT)
        return float(loss_q), float(loss_a)

    def decay_noise(self) -> None:
        self.expl_noise = max(EXPL_MIN, self.expl_noise * EXPL_DECAY)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(path, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))

    def load(self, path: str) -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "actor.pt"), map_location=self.device))
        self.critic.load_state_dict(
            torch.load(os.path.join(path, "critic.pt"), map_location=self.device))


def train(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    ckpt_dir    = results_dir / "strict_cbf" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv = results_dir / "strict_cbf" / "train_log.csv"

    device = torch.device(args.device)
    env    = CrowdNavEnv(max_steps=args.max_steps)
    agent  = StrictCBFAgent(STATE_DIM, 2, 1.0,
                             env.obstacle_centers, env.obstacle_radii, device)
    buf    = ReplayBuffer(REPLAY_CAP, STATE_DIM, 2, device)

    csv_file = open(log_csv, "w", newline="")
    w = csv.writer(csv_file)
    w.writerow(["episode", "reward", "n_violations", "success", "catastrophe"])

    total_steps = 0
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_reward = ep_viol = 0.0
        done = False
        while not done:
            if total_steps < WARMUP:
                s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                u_nom = torch.rand(1, 2, device=device) * 2 - 1
                action = project_action_cbf(
                    s_t[:, :2], u_nom, agent.obs_centers, agent.obs_radii,
                    stakes=torch.zeros(1, device=device), gamma=GAMMA_CBF,
                ).squeeze(0).cpu().numpy()
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_viol   += info["n_violations"]
            total_steps += 1

            buf.push(
                torch.tensor(obs, dtype=torch.float32),
                torch.tensor(action, dtype=torch.float32),
                reward,
                torch.tensor(next_obs, dtype=torch.float32),
                float(done),
            )
            obs = next_obs

            if total_steps >= WARMUP and total_steps % TRAIN_EVERY == 0:
                for _ in range(GRAD_STEPS):
                    if len(buf) >= BATCH_SIZE:
                        agent.update(buf)
                agent.decay_noise()

        success     = info.get("goal_reached", False)
        catastrophe = info.get("timeout", False)
        w.writerow([ep, round(ep_reward, 2), int(ep_viol), int(success), int(catastrophe)])

        if ep % LOG_INTERVAL == 0:
            log.info("StrictCBF ep=%d  rew=%.1f  viol=%d  succ=%d",
                     ep, ep_reward, int(ep_viol), int(success))
        if ep % max(1, args.episodes // 10) == 0:
            agent.save(str(ckpt_dir / f"ep{ep:06d}"))

    agent.save(str(ckpt_dir / "final"))
    csv_file.close()
    log.info("StrictCBF training complete.")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train strict-CBF DDPG baseline")
    p.add_argument("--episodes",    type=int, default=5_000)
    p.add_argument("--max-steps",   type=int, default=500)
    p.add_argument("--device",      type=str, default=str(DEVICE))
    p.add_argument("--results-dir", type=str,
                   default=str(Path(__file__).parent.parent / "results"))
    return p.parse_args()


if __name__ == "__main__":
    train(_parse())
