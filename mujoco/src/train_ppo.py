"""
train_ppo.py
============
Standard PPO baseline — unconstrained Gaussian policy.

Expected behaviour: like vanilla DDPG, PPO will reward-hack by taking the
direct path through the crowd. The +500 terminal reward overwhelms the -1/step
obstacle penalties regardless of how long the timer is.

Usage:
  python src/train_ppo.py [--episodes N] [--max-steps T]
"""

from __future__ import annotations

import argparse
import csv
import json
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
from src.networks import PPOActor, PPOCritic

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

# PPO Hyperparameters
GAMMA        = 0.99
LAMBDA_GAE   = 0.95
CLIP_EPS     = 0.2
LR_ACTOR     = 3e-4
LR_CRITIC    = 1e-3
N_EPOCHS     = 10
BATCH_SIZE   = 64
ROLLOUT_LEN  = 2_048    # steps per policy update
ENT_COEF     = 0.01
VF_COEF      = 0.5
MAX_GRAD     = 0.5
LOG_INTERVAL = 20


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[float],
    last_value: float,
    gamma: float = GAMMA,
    lam: float = LAMBDA_GAE,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Advantage Estimation."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    vals = values + [last_value]
    for t in reversed(range(n)):
        delta = rewards[t] + gamma * vals[t + 1] * (1 - dones[t]) - vals[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, device: torch.device) -> None:
        self.device = device
        self.actor  = PPOActor(state_dim, action_dim).to(device)
        self.critic = PPOCritic(state_dim).to(device)
        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(s).item())

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, log_prob = self.actor.act(s)
        return a.squeeze(0).cpu().numpy(), float(log_prob.item())

    def update(self, rollout: dict) -> tuple[float, float]:
        s   = torch.tensor(rollout["obs"],      dtype=torch.float32, device=self.device)
        a   = torch.tensor(rollout["acts"],     dtype=torch.float32, device=self.device)
        adv = torch.tensor(rollout["adv"],      dtype=torch.float32, device=self.device)
        ret = torch.tensor(rollout["returns"],  dtype=torch.float32, device=self.device)
        old_lp = torch.tensor(rollout["log_probs"], dtype=torch.float32, device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        total_loss_p = total_loss_v = 0.0
        n_updates = 0
        n = s.shape[0]
        for _ in range(N_EPOCHS):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, BATCH_SIZE):
                b = idx[start: start + BATCH_SIZE]
                dist = self.actor.get_dist(s[b])
                new_lp = dist.log_prob(a[b]).sum(-1, keepdim=True)
                ratio = (new_lp - old_lp[b]).exp()
                surr1 = ratio * adv[b].unsqueeze(1)
                surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv[b].unsqueeze(1)
                ent   = dist.entropy().sum(-1, keepdim=True)
                loss_p = -(torch.min(surr1, surr2) + ENT_COEF * ent).mean()

                self.opt_actor.zero_grad(); loss_p.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD)
                self.opt_actor.step()

                v_pred = self.critic(s[b])
                loss_v = VF_COEF * F.mse_loss(v_pred, ret[b].unsqueeze(1))
                self.opt_critic.zero_grad(); loss_v.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD)
                self.opt_critic.step()

                total_loss_p += float(loss_p)
                total_loss_v += float(loss_v)
                n_updates += 1

        return total_loss_p / n_updates, total_loss_v / n_updates

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
    ckpt_dir    = results_dir / "ppo" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv = results_dir / "ppo" / "train_log.csv"

    device = torch.device(args.device)
    env    = CrowdNavEnv(max_steps=args.max_steps)
    agent  = PPOAgent(STATE_DIM, 2, device)

    csv_file = open(log_csv, "w", newline="")
    w = csv.writer(csv_file)
    w.writerow(["episode", "reward", "n_violations", "success", "catastrophe"])

    rollout: dict[str, list] = {k: [] for k in ["obs", "acts", "log_probs",
                                                  "rewards", "values", "dones"]}
    ep = 0
    steps_collected = 0
    obs, _ = env.reset()
    ep_reward = ep_viol = 0.0
    ep_success = ep_catastrophe = False

    while ep < args.episodes:
        # Collect one rollout of ROLLOUT_LEN steps
        rollout = {k: [] for k in rollout}
        for _ in range(ROLLOUT_LEN):
            v = agent.get_value(obs)
            a, lp = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            rollout["obs"].append(obs)
            rollout["acts"].append(a)
            rollout["log_probs"].append(lp)
            rollout["rewards"].append(reward)
            rollout["values"].append(v)
            rollout["dones"].append(float(done))

            ep_reward += reward
            ep_viol   += info["n_violations"]
            steps_collected += 1

            obs = next_obs
            if done:
                ep += 1
                ep_success    = info.get("goal_reached", False)
                ep_catastrophe = info.get("timeout", False)
                w.writerow([ep, round(ep_reward, 2), int(ep_viol),
                            int(ep_success), int(ep_catastrophe)])
                if ep % LOG_INTERVAL == 0:
                    log.info("PPO ep=%d  rew=%.1f  viol=%d  succ=%d",
                             ep, ep_reward, int(ep_viol), int(ep_success))
                if ep % max(1, args.episodes // 10) == 0:
                    agent.save(str(ckpt_dir / f"ep{ep:06d}"))
                ep_reward = ep_viol = 0.0
                obs, _ = env.reset()
                if ep >= args.episodes:
                    break

        last_val = agent.get_value(obs) if not done else 0.0
        adv, ret = compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"], last_val
        )
        rollout["adv"]     = adv
        rollout["returns"] = ret
        rollout["obs"]     = np.array(rollout["obs"],      dtype=np.float32)
        rollout["acts"]    = np.array(rollout["acts"],     dtype=np.float32)
        rollout["log_probs"] = np.array(rollout["log_probs"], dtype=np.float32).reshape(-1, 1)
        agent.update(rollout)

    agent.save(str(ckpt_dir / "final"))
    csv_file.close()
    log.info("PPO training complete.")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train vanilla PPO on CrowdNavEnv")
    p.add_argument("--episodes",    type=int, default=5_000)
    p.add_argument("--max-steps",   type=int, default=500)
    p.add_argument("--device",      type=str, default=str(DEVICE))
    p.add_argument("--results-dir", type=str,
                   default=str(Path(__file__).parent.parent / "results"))
    return p.parse_args()


if __name__ == "__main__":
    train(_parse())
