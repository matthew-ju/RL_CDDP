"""
train_cddp.py
=============
CDDP-DDPG training script for the "Time-Pressure Crowd" MuJoCo environment.

Architecture
------------
  • Actor π_θ(s) → u_nom     (DDPG-style deterministic policy)
  • Z_free(s, a) → Q-value   (utilitarian critic, trained on dynamic-boundary data)
  • Z_deon(s) → logit P_cat  (deontic critic, trained on STRICT rollouts only)
  • CBF projection layer      (analytical, stakes-aware dynamic boundary)

Training protocol
-----------------
1. PRE-TRAINING phase (first `--pretrain-episodes`):
   Actor + Z_free trained under strict S=0 (no CDDP relaxation).
   This seeds the policy with basic goal-seeking behaviour before
   Z_deon labels accumulate.

2. MAIN phase: alternating rollout collection
   - Even episodes: strict S=0 → data fed to DeonticReplayBuffer (labeled
     post-episode as catastrophe / success).
   - Odd episodes: dynamic boundary using current Z_deon → utilitarian buffer.

Stakes functional: S(s) = clip(P_cat(s) / P_ref, 0, 1)
  where P_cat(s) = sigmoid(Z_deon(s)) and P_ref = 0.2.

Usage
-----
  python src/train_cddp.py [--episodes N] [--max-steps T] [--device cpu]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running from repo root or from mujoco/
_SRC = Path(__file__).parent
_MJ_ROOT = _SRC.parent
sys.path.insert(0, str(_MJ_ROOT))
sys.path.insert(0, str(_SRC))

from src.crowd_env import CrowdNavEnv, N_OBS, STATE_DIM
from src.networks import Actor, UtilitarianCritic, DeonticCritic, soft_update
from src.cbf_layer import project_action_cbf
from src.replay_buffer import ReplayBuffer, DeonticReplayBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (overridden by argparse)
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

STATE_DIM_      = STATE_DIM   # 15
ACTION_DIM      = 2
ACT_LIMIT       = 1.0

# CDDP hyperparameters (matching second_gridworld precedent)
P_REF           = 0.2
TAU_CDDP        = 0.3
ALPHA_PRIME     = 10.0
GAMMA_CBF       = 1.5

# RL hyperparameters
GAMMA_RL        = 0.99
TAU_SOFT        = 0.005
LR_ACTOR        = 3e-4
LR_Z_FREE       = 1e-3
LR_Z_DEON       = 1e-3
BATCH_SIZE      = 256
REPLAY_CAP      = 300_000
DEON_CAP        = 100_000
WARMUP_STEPS    = 2_000
TRAIN_EVERY     = 2
GRADIENT_STEPS  = 2

# Exploration
EXPL_NOISE      = 0.15   # Gaussian std on top of actor output
EXPL_DECAY      = 0.9999
EXPL_MIN        = 0.05

LOG_INTERVAL    = 50


# ---------------------------------------------------------------------------
# CDDP Agent
# ---------------------------------------------------------------------------

class CDDPAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        act_limit: float,
        obs_centers: np.ndarray,
        obs_radii: np.ndarray,
        device: torch.device,
    ) -> None:
        self.device = device
        self.act_limit = act_limit

        self.obs_centers = torch.tensor(obs_centers, dtype=torch.float32, device=device)
        self.obs_radii   = torch.tensor(obs_radii,   dtype=torch.float32, device=device)

        # Networks
        self.actor         = Actor(state_dim, action_dim, act_limit).to(device)
        self.actor_target  = Actor(state_dim, action_dim, act_limit).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.requires_grad_(False)

        self.z_free        = UtilitarianCritic(state_dim, action_dim).to(device)
        self.z_free_target = UtilitarianCritic(state_dim, action_dim).to(device)
        self.z_free_target.load_state_dict(self.z_free.state_dict())
        self.z_free_target.requires_grad_(False)

        self.z_deon = DeonticCritic(state_dim).to(device)

        # Optimizers
        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.opt_z_free = torch.optim.Adam(self.z_free.parameters(), lr=LR_Z_FREE)
        self.opt_z_deon = torch.optim.Adam(self.z_deon.parameters(), lr=LR_Z_DEON)

        self.expl_noise = EXPL_NOISE

    # ------------------------------------------------------------------
    def stakes(self, s: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """S(s) = clip(sigmoid(Z_deon(s)) / P_ref, 0, 1).  Shape: (B,)"""
        logits = self.z_deon(s)
        p_cat = torch.sigmoid(logits)
        if detach:
            p_cat = p_cat.detach()
        return torch.clamp(p_cat.squeeze(-1) / P_REF, 0.0, 1.0)

    def project(self, pos: torch.Tensor, u_nom: torch.Tensor,
                stakes: torch.Tensor) -> torch.Tensor:
        return project_action_cbf(
            pos, u_nom, self.obs_centers, self.obs_radii,
            stakes=stakes, gamma=GAMMA_CBF, tau=TAU_CDDP, alpha_prime=ALPHA_PRIME,
        )

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        strict: bool = False,
        explore: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Return (action, stakes_value)."""
        s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        u_nom = self.actor(s)
        if explore:
            noise = torch.randn_like(u_nom) * self.expl_noise
            u_nom = (u_nom + noise).clamp(-self.act_limit, self.act_limit)

        if strict:
            stk = torch.zeros(1, dtype=torch.float32, device=self.device)
        else:
            stk = self.stakes(s)

        pos = s[:, :2]
        u_safe = self.project(pos, u_nom, stk)
        return u_safe.squeeze(0).cpu().numpy(), float(stk.item())

    # ------------------------------------------------------------------
    def update_utilitarian(
        self,
        buf: ReplayBuffer,
        strict: bool = False,
    ) -> tuple[float, float]:
        s, a, r, s_, done = buf.sample(BATCH_SIZE)
        pos_ = s_[:, :2]

        with torch.no_grad():
            u_nom_ = self.actor_target(s_)
            if strict:
                stk_ = torch.zeros(s_.shape[0], device=self.device)
            else:
                stk_ = self.stakes(s_)
            u_safe_ = self.project(pos_, u_nom_, stk_)
            q_tgt = r + GAMMA_RL * (1.0 - done) * self.z_free_target(s_, u_safe_)

        q_pred = self.z_free(s, a)
        loss_q = F.mse_loss(q_pred, q_tgt)
        self.opt_z_free.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.z_free.parameters(), 1.0)
        self.opt_z_free.step()

        # Actor loss
        u_nom_a = self.actor(s)
        stk_a = torch.zeros(s.shape[0], device=self.device) if strict else self.stakes(s)
        u_safe_a = self.project(s[:, :2], u_nom_a, stk_a)
        loss_actor = -self.z_free(s, u_safe_a).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        soft_update(self.actor_target, self.actor, TAU_SOFT)
        soft_update(self.z_free_target, self.z_free, TAU_SOFT)
        return float(loss_q), float(loss_actor)

    def update_deontic(self, buf: DeonticReplayBuffer) -> float:
        s, y = buf.sample(BATCH_SIZE)
        logits = self.z_deon(s)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.opt_z_deon.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.z_deon.parameters(), 1.0)
        self.opt_z_deon.step()
        return float(loss)

    def decay_noise(self) -> None:
        self.expl_noise = max(EXPL_MIN, self.expl_noise * EXPL_DECAY)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(path, "actor.pt"))
        torch.save(self.z_free.state_dict(), os.path.join(path, "z_free.pt"))
        torch.save(self.z_deon.state_dict(), os.path.join(path, "z_deon.pt"))

    def load(self, path: str) -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "actor.pt"), map_location=self.device))
        self.z_free.load_state_dict(
            torch.load(os.path.join(path, "z_free.pt"), map_location=self.device))
        self.z_deon.load_state_dict(
            torch.load(os.path.join(path, "z_deon.pt"), map_location=self.device))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    ckpt_dir    = results_dir / "cddp" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv     = results_dir / "cddp" / "train_log.csv"
    log_json    = results_dir / "cddp" / "train_log.json"

    device = torch.device(args.device)

    env = CrowdNavEnv(max_steps=args.max_steps)
    agent = CDDPAgent(
        STATE_DIM_, ACTION_DIM, ACT_LIMIT,
        env.obstacle_centers, env.obstacle_radii, device,
    )

    util_buf  = ReplayBuffer(REPLAY_CAP, STATE_DIM_, ACTION_DIM, device)
    deon_buf  = DeonticReplayBuffer(DEON_CAP, STATE_DIM_, device)

    total_steps = 0
    episode_logs: list[dict] = []

    csv_file = open(log_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode", "timer", "total_steps_env", "reward", "n_violations",
        "success", "catastrophe", "avg_stakes", "loss_q", "loss_actor", "loss_deon",
    ])

    log.info("Starting CDDP training  [device=%s, episodes=%d, max_steps=%d]",
             device, args.episodes, args.max_steps)

    for ep in range(1, args.episodes + 1):
        pretrain = ep <= args.pretrain_episodes
        # Even episodes → strict rollout (for Z_deon); odd → dynamic boundary
        strict_ep = pretrain or (ep % 2 == 0)

        obs, _ = env.reset()
        deon_buf.start_episode()

        ep_reward     = 0.0
        ep_violations = 0
        ep_stakes_sum = 0.0
        ep_steps      = 0
        loss_q = loss_actor = loss_deon = 0.0

        done = False
        while not done:
            explore = total_steps >= WARMUP_STEPS or pretrain
            action, stk = agent.select_action(obs, strict=strict_ep, explore=explore)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward     += reward
            ep_violations += info["n_violations"]
            ep_stakes_sum += stk
            ep_steps      += 1
            total_steps   += 1

            s_t  = torch.tensor(obs,      dtype=torch.float32)
            a_t  = torch.tensor(action,   dtype=torch.float32)
            s_t1 = torch.tensor(next_obs, dtype=torch.float32)

            if strict_ep:
                deon_buf.push_state(s_t)

            # Fix: During pre-training, we must populate the utilitarian buffer
            # even in strict episodes so the agent learns basic navigation.
            if not strict_ep or pretrain:
                util_buf.push(s_t, a_t, reward, s_t1, float(done))

            obs = next_obs

            # Training updates
            if total_steps >= WARMUP_STEPS and total_steps % TRAIN_EVERY == 0:
                for _ in range(GRADIENT_STEPS):
                    if len(util_buf) >= BATCH_SIZE:
                        lq, la = agent.update_utilitarian(util_buf, strict=pretrain)
                        loss_q     = lq
                        loss_actor = la
                    if len(deon_buf) >= BATCH_SIZE:
                        ld = agent.update_deontic(deon_buf)
                        loss_deon = ld
                agent.decay_noise()

        # Label deontic episode outcome
        catastrophe = info.get("timeout", False)
        if strict_ep:
            deon_buf.label_episode(catastrophe)

        avg_stk = ep_stakes_sum / max(ep_steps, 1)
        success = info.get("goal_reached", False)

        row = [
            ep, args.max_steps, total_steps, round(ep_reward, 2),
            ep_violations, int(success), int(catastrophe),
            round(avg_stk, 4),
            round(loss_q, 4), round(loss_actor, 4), round(loss_deon, 4),
        ]
        csv_writer.writerow(row)
        episode_logs.append({
            "episode": ep, "reward": ep_reward, "n_violations": ep_violations,
            "success": success, "catastrophe": catastrophe, "avg_stakes": avg_stk,
        })

        if ep % LOG_INTERVAL == 0:
            log.info(
                "ep=%d  rew=%.1f  viol=%d  succ=%d  cat=%d  stk=%.3f  "
                "lq=%.4f  la=%.4f  ld=%.4f",
                ep, ep_reward, ep_violations, success, catastrophe,
                avg_stk, loss_q, loss_actor, loss_deon,
            )

        if ep % max(1, args.episodes // 10) == 0:
            agent.save(str(ckpt_dir / f"ep{ep:06d}"))

    # Final checkpoint
    agent.save(str(ckpt_dir / "final"))
    csv_file.close()

    with open(log_json, "w") as f:
        json.dump(episode_logs, f, indent=2)

    log.info("Training complete. Checkpoints at %s", ckpt_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CDDP agent on CrowdNavEnv")
    p.add_argument("--episodes",          type=int,   default=5_000)
    p.add_argument("--pretrain-episodes", type=int,   default=1_000)
    p.add_argument("--max-steps",         type=int,   default=500)
    p.add_argument("--device",            type=str,   default=str(DEVICE))
    p.add_argument("--results-dir",       type=str,
                   default=str(Path(__file__).parent.parent / "results"))
    return p.parse_args()


if __name__ == "__main__":
    train(_parse())
