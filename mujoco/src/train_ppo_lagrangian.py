"""
train_ppo_lagrangian.py
=======================
PPO-Lagrangian baseline — PPO with a learned Lagrange multiplier λ that
penalises constraint violations.

Expected behaviour: better than vanilla PPO/DDPG at avoiding violations,
but without structural safety guarantees. Relies on trial-and-error to
learn the penalty balance. At short timers it may still violate (or over-
constrain itself and time out). No explicit CBF projection: the Lagrange
dual simply adjusts the reward signal.

Cost signal: total obstacle penetration steps per episode.
Cost budget: d_cost (default 0 → zero-violation preference).

Lagrange update:
  λ ← max(0, λ + lr_λ * (J_cost - d_cost))

Usage:
  python src/train_ppo_lagrangian.py [--episodes N] [--max-steps T]
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
from src.networks import PPOActor, PPOCritic, CostCritic
from src.train_ppo import compute_gae

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

GAMMA      = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS   = 0.2
LR_ACTOR   = 3e-4
LR_CRITIC  = 1e-3
LR_COST    = 1e-3
LR_LAMBDA  = 0.01     # Lagrange multiplier step size
D_COST     = 0.0      # Cost budget (0 → prefer zero violations)
N_EPOCHS   = 10
BATCH_SIZE = 64
ROLLOUT_LEN = 2_048
ENT_COEF   = 0.01
VF_COEF    = 0.5
MAX_GRAD   = 0.5
LOG_INTERVAL = 20


class PPOLagAgent:
    def __init__(self, state_dim: int, action_dim: int, device: torch.device) -> None:
        self.device = device
        self.actor     = PPOActor(state_dim, action_dim).to(device)
        self.critic    = PPOCritic(state_dim).to(device)
        self.cost_crit = CostCritic(state_dim).to(device)
        self.opt_actor     = torch.optim.Adam(self.actor.parameters(),     lr=LR_ACTOR)
        self.opt_critic    = torch.optim.Adam(self.critic.parameters(),    lr=LR_CRITIC)
        self.opt_cost_crit = torch.optim.Adam(self.cost_crit.parameters(), lr=LR_COST)
        self.lam = 0.1   # Lagrange multiplier, trained online

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> tuple[float, float]:
        s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        v  = float(self.critic(s).item())
        vc = float(self.cost_crit(s).item())
        return v, vc

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, lp = self.actor.act(s)
        return a.squeeze(0).cpu().numpy(), float(lp.item())

    def update(self, rollout: dict, ep_cost: float) -> tuple[float, float, float]:
        s   = torch.tensor(rollout["obs"],       dtype=torch.float32, device=self.device)
        a   = torch.tensor(rollout["acts"],      dtype=torch.float32, device=self.device)
        adv = torch.tensor(rollout["adv"],       dtype=torch.float32, device=self.device)
        ret = torch.tensor(rollout["returns"],   dtype=torch.float32, device=self.device)
        cadv = torch.tensor(rollout["cadv"],     dtype=torch.float32, device=self.device)
        cret = torch.tensor(rollout["creturns"], dtype=torch.float32, device=self.device)
        old_lp = torch.tensor(rollout["log_probs"], dtype=torch.float32, device=self.device)

        adv  = (adv  - adv.mean())  / (adv.std()  + 1e-8)
        cadv = (cadv - cadv.mean()) / (cadv.std() + 1e-8)

        lam = max(0.0, self.lam)
        total_lp = total_lv = total_lc = 0.0
        n_upd = 0
        n = s.shape[0]

        for _ in range(N_EPOCHS):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, BATCH_SIZE):
                b = idx[start: start + BATCH_SIZE]
                dist = self.actor.get_dist(s[b])
                new_lp = dist.log_prob(a[b]).sum(-1, keepdim=True)
                ratio  = (new_lp - old_lp[b]).exp()

                # Combined advantage: reward - λ * cost
                adv_combined = adv[b].unsqueeze(1) - lam * cadv[b].unsqueeze(1)
                surr1 = ratio * adv_combined
                surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_combined
                ent   = dist.entropy().sum(-1, keepdim=True)
                loss_p = -(torch.min(surr1, surr2) + ENT_COEF * ent).mean()
                self.opt_actor.zero_grad(); loss_p.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD)
                self.opt_actor.step()

                loss_v = VF_COEF * F.mse_loss(self.critic(s[b]), ret[b].unsqueeze(1))
                self.opt_critic.zero_grad(); loss_v.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD)
                self.opt_critic.step()

                loss_c = VF_COEF * F.mse_loss(self.cost_crit(s[b]), cret[b].unsqueeze(1))
                self.opt_cost_crit.zero_grad(); loss_c.backward()
                nn.utils.clip_grad_norm_(self.cost_crit.parameters(), MAX_GRAD)
                self.opt_cost_crit.step()

                total_lp += float(loss_p)
                total_lv += float(loss_v)
                total_lc += float(loss_c)
                n_upd += 1

        # Update Lagrange multiplier
        self.lam = max(0.0, self.lam + LR_LAMBDA * (ep_cost - D_COST))
        return total_lp / n_upd, total_lv / n_upd, total_lc / n_upd

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),     os.path.join(path, "actor.pt"))
        torch.save(self.critic.state_dict(),    os.path.join(path, "critic.pt"))
        torch.save(self.cost_crit.state_dict(), os.path.join(path, "cost_critic.pt"))

    def load(self, path: str) -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "actor.pt"), map_location=self.device))
        self.critic.load_state_dict(
            torch.load(os.path.join(path, "critic.pt"), map_location=self.device))
        self.cost_crit.load_state_dict(
            torch.load(os.path.join(path, "cost_critic.pt"), map_location=self.device))


def train(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    ckpt_dir    = results_dir / "ppo_lag" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv = results_dir / "ppo_lag" / "train_log.csv"

    device = torch.device(args.device)
    env    = CrowdNavEnv(max_steps=args.max_steps)
    agent  = PPOLagAgent(STATE_DIM, 2, device)

    csv_file = open(log_csv, "w", newline="")
    w = csv.writer(csv_file)
    w.writerow(["episode", "reward", "n_violations", "success", "catastrophe", "lam"])

    rollout_keys = ["obs", "acts", "log_probs", "rewards", "costs",
                    "values", "cvalues", "dones"]
    rollout: dict[str, list] = {k: [] for k in rollout_keys}

    ep = 0
    ep_reward = ep_viol = 0.0
    obs, _ = env.reset()

    while ep < args.episodes:
        rollout = {k: [] for k in rollout_keys}
        for _ in range(ROLLOUT_LEN):
            v, vc = agent.get_value(obs)
            a, lp = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            cost = float(info["n_violations"])

            rollout["obs"].append(obs)
            rollout["acts"].append(a)
            rollout["log_probs"].append(lp)
            rollout["rewards"].append(reward)
            rollout["costs"].append(cost)
            rollout["values"].append(v)
            rollout["cvalues"].append(vc)
            rollout["dones"].append(float(done))

            ep_reward += reward
            ep_viol   += cost
            obs = next_obs

            if done:
                ep += 1
                success     = info.get("goal_reached", False)
                catastrophe = info.get("timeout", False)
                w.writerow([ep, round(ep_reward, 2), int(ep_viol),
                            int(success), int(catastrophe), round(agent.lam, 4)])
                if ep % LOG_INTERVAL == 0:
                    log.info("PPO-Lag ep=%d  rew=%.1f  viol=%d  succ=%d  λ=%.3f",
                             ep, ep_reward, int(ep_viol), int(success), agent.lam)
                if ep % max(1, args.episodes // 10) == 0:
                    agent.save(str(ckpt_dir / f"ep{ep:06d}"))
                ep_reward = ep_viol = 0.0
                obs, _ = env.reset()
                if ep >= args.episodes:
                    break

        last_v, last_vc = agent.get_value(obs) if not done else (0.0, 0.0)
        adv, ret   = compute_gae(rollout["rewards"], rollout["values"],
                                  rollout["dones"], last_v)
        cadv, cret = compute_gae(rollout["costs"],   rollout["cvalues"],
                                  rollout["dones"], last_vc)
        rollout["adv"]      = adv
        rollout["returns"]  = ret
        rollout["cadv"]     = cadv
        rollout["creturns"] = cret
        rollout["obs"]      = np.array(rollout["obs"],   dtype=np.float32)
        rollout["acts"]     = np.array(rollout["acts"],  dtype=np.float32)
        rollout["log_probs"] = np.array(rollout["log_probs"], dtype=np.float32).reshape(-1, 1)

        ep_cost_avg = float(np.mean(rollout["costs"]))
        agent.update(rollout, ep_cost_avg)

    agent.save(str(ckpt_dir / "final"))
    csv_file.close()
    log.info("PPO-Lagrangian training complete.  Final λ=%.4f", agent.lam)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO-Lagrangian on CrowdNavEnv")
    p.add_argument("--episodes",    type=int, default=5_000)
    p.add_argument("--max-steps",   type=int, default=500)
    p.add_argument("--device",      type=str, default=str(DEVICE))
    p.add_argument("--results-dir", type=str,
                   default=str(Path(__file__).parent.parent / "results"))
    return p.parse_args()


if __name__ == "__main__":
    train(_parse())
