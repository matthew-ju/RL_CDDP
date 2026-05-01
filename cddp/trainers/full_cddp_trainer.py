"""
Full CDDP Trainer — Phase 5: Two-timescale training with distributional critics.

Training Flow:
    1. DEONTIC WARMUP (Phase I):
       - Collect N_warmup episodes with forced deontic mode (shield always on)
       - Populate DeonticReplayBuffer
       - Pre-train Z_deon on deontic-only data

    2. MAIN LOOP (Phase II): for each outer epoch:
       a. CRITIC UPDATE (inner loop):
          - Freeze actor weights (deepcopy snapshot)
          - Collect rollouts using CDDP agent (mode switching)
          - Update Z_free from π_θ transitions
          - Update Z_deon from π_d transitions
          - Update stakes function with new critics

       b. ACTOR UPDATE (outer loop):
          - Collect rollouts using updated stakes
          - PPO update on actor using GAE advantages
          - Evaluate on both low/high stakes

Architecture:
    Actor:  9 → 64 → 64 → 2 (Gaussian continuous)
    Critic: 9 → 64 → 64 → 1 (value function for GAE)
    Z_free: 9 → 128 → 128 → N (distributional)
    Z_deon: 9 → 128 → 128 → N (distributional)

Stakes: ProspectiveStakes(Z_free, Z_deon) or AnalyticalStakes (oracle)
"""

import os
import sys
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

_omnisafe_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'omnisafe_SRL')
sys.path.insert(0, os.path.abspath(_omnisafe_dir))

from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.shield.action_shield import ActionShield
from cddp.stakes.analytical import AnalyticalStakesDmin
from cddp.stakes.prospective import ProspectiveStakes
from cddp.agents.ppo_actor import PPOActor, OmniSafeGaussianActor, discretize_action
from cddp.agents.cddp_agent import CDDPAgent
from cddp.critics.qr_critic import DualCritics
from cddp.replay.buffers import ReplayBuffer, DeonticReplayBuffer, Transition
from cddp.trainers.cddp_trainer import ValueCritic


class FullCDDPTrainer:
    """Full CDDP trainer with distributional critics and two-timescale updates.

    This is the complete training pipeline that:
    1. Trains Z_free and Z_deon critics on-policy data
    2. Computes prospective stakes from the critics
    3. Updates the actor with PPO using CDDP mode switching

    Args:
        layout_id: Grid layout
        bomb_timer: Bomb timer for high-stakes scenarios
        tau: Mode-switching threshold
        n_quantiles: Number of quantile atoms (21 dev, 51 final)
        use_oracle_stakes: If True, skip critic training and use analytical stakes
        warmup_episodes: Number of deontic-only warmup episodes
        critic_updates_per_epoch: Inner loop critic gradient steps
        lr_actor: Actor learning rate
        lr_critic: Critic learning rate
        gamma: Discount factor
        gae_lambda: GAE parameter
        clip_epsilon: PPO clip
        entropy_coef: Entropy bonus coefficient
        value_coef: Value loss coefficient
        ppo_epochs: PPO mini-batch epochs per update
        steps_per_epoch: Rollout steps per epoch
        total_epochs: Total training epochs
        buffer_capacity: Replay buffer size
        critic_batch_size: Critic update batch size
        save_dir: Results directory
        checkpoint_interval: Save every N epochs
    """

    def __init__(
        self,
        layout_id: int = 1,
        bomb_timer: int = 5,
        tau: float = 0.5,
        n_quantiles: int = 21,
        use_oracle_stakes: bool = False,
        warmup_episodes: int = 100,
        critic_updates_per_epoch: int = 20,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        ppo_epochs: int = 10,
        steps_per_epoch: int = 1000,
        total_epochs: int = 300,
        buffer_capacity: int = 50_000,
        critic_batch_size: int = 64,
        save_dir: str = 'cddp_results_full',
        checkpoint_interval: int = 50,
    ):
        self.layout_id = layout_id
        self.bomb_timer = bomb_timer
        self.tau = tau
        self.use_oracle_stakes = use_oracle_stakes
        self.warmup_episodes = warmup_episodes
        self.critic_updates_per_epoch = critic_updates_per_epoch
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.critic_batch_size = critic_batch_size
        self.save_dir = save_dir
        self.checkpoint_interval = checkpoint_interval

        obs_dim = 9
        self.obs_dim = obs_dim

        # ── Actor + Value Critic ──
        self.actor_net = OmniSafeGaussianActor(obs_dim=obs_dim, action_dim=2, hidden_dim=64)
        self.value_net = ValueCritic(obs_dim=obs_dim, hidden_dim=64)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.value_optim = optim.Adam(self.value_net.parameters(), lr=lr_actor)

        # ── Distributional Critics ──
        self.dual_critics = DualCritics(
            obs_dim=obs_dim, n_quantiles=n_quantiles,
            hidden_dim=128, lr=lr_critic,
        )

        # ── Replay Buffers ──
        self.buffer_free = ReplayBuffer(capacity=buffer_capacity, obs_dim=obs_dim)
        self.buffer_deon = DeonticReplayBuffer(capacity=buffer_capacity, obs_dim=obs_dim)

        # ── CDDP Agent ──
        self.actor_wrapper = PPOActor(obs_dim=obs_dim, hidden_dim=64)
        self.actor_wrapper.actor = self.actor_net
        self.shield = ActionShield()

        if use_oracle_stakes:
            self.stakes_fn = AnalyticalStakesDmin()
        else:
            self.stakes_fn = ProspectiveStakes(self.dual_critics, obs_dim=obs_dim)

        self.agent = CDDPAgent(self.actor_wrapper, self.shield, self.stakes_fn, tau=tau)

        # ── History ──
        self.history = defaultdict(list)
        self.episode_counter = 0

    def _pad_obs(self, obs: np.ndarray) -> torch.Tensor:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[-1] < self.obs_dim:
            padding = np.zeros(self.obs_dim - obs.shape[-1], dtype=np.float32)
            obs = np.concatenate([obs, padding])
        return torch.FloatTensor(obs)

    # ─── Phase I: Deontic Warmup ────────────────────────────────

    def deontic_warmup(self):
        """Collect deontic-only episodes to pre-train Z_deon."""
        print(f"\n{'─'*40}")
        print(f"Deontic Warmup: {self.warmup_episodes} episodes")
        print(f"{'─'*40}")

        total_steps = 0
        total_violations = 0

        for ep in range(self.warmup_episodes):
            # Alternate bomb timers for diverse states
            timer = self.bomb_timer if ep % 2 == 0 else None

            env = InstrumentedHostageEnv.make(
                layout_id=self.layout_id, bomb_timer=timer)
            obs, info = env.reset()
            done = False
            step = 0

            while not done:
                action, act_info = self.agent.select_deontic_action(obs, env)
                next_obs, reward, terminated, truncated, new_info = env.step(action)
                done = terminated or truncated

                # Store in deontic buffer
                self.buffer_deon.add(Transition(
                    obs=obs, action=action, reward=reward,
                    next_obs=next_obs, done=done,
                    mode='deontic',
                    T_rem=info.get('T_rem', float('inf')),
                    D_min=info.get('D_min', 0),
                    L_safe=info.get('L_safe', float('inf')),
                    in_failure_band=info.get('in_failure_band', False),
                    episode_id=self.episode_counter,
                    step_idx=step,
                ))

                obs = next_obs
                info = new_info
                step += 1
                total_steps += 1

            total_violations += len(env.trampled_civilians)
            self.episode_counter += 1

        print(f"  Steps: {total_steps}, Violations: {total_violations}")
        assert total_violations == 0, f"Warmup violations! {total_violations}"
        print(f"  ✓ Zero violations (Theorem 1)")

        # Pre-train Z_deon
        if len(self.buffer_deon) >= self.critic_batch_size:
            losses = []
            for _ in range(100):
                batch = self.buffer_deon.sample(self.critic_batch_size)
                loss = self.dual_critics.update_z_deon(
                    batch['obs'], batch['rewards'],
                    batch['next_obs'], batch['dones'],
                    gamma=self.gamma,
                )
                losses.append(loss)
            print(f"  Z_deon pre-training: loss {np.mean(losses):.4f}")

    # ─── Phase II: Main Training Loop ───────────────────────────
    
    def collect_deontic_rollouts(self, num_steps):
        """Collect strictly bounded π_d rollouts exclusively for Z_deon."""
        step_count = 0
        while step_count < num_steps:
            use_bomb = np.random.random() > 0.5
            timer = self.bomb_timer if use_bomb else None
            env = InstrumentedHostageEnv.make(layout_id=self.layout_id, bomb_timer=timer)
            obs, info = env.reset()
            done = False
            step = 0
            while not done and step_count < num_steps:
                obs_t = self._pad_obs(obs)
                with torch.no_grad():
                    mean, log_std = self.actor_net(obs_t.unsqueeze(0))
                    mean = mean.squeeze(0)
                    std = log_std.squeeze(0).exp()
                    noise = torch.randn_like(mean)
                    continuous_action = mean + std * noise
                discrete_action = discretize_action(continuous_action)
                safe = self.shield.get_safe_actions(env)
                if discrete_action not in safe:
                    discrete_action = self.agent.deontic_selector.select_safe_action(obs, env, safe)[0] if hasattr(self.agent.deontic_selector, 'select_safe_action') else self.agent.deontic_selector.select_action(env)
                next_obs, reward, terminated, truncated, new_info = env.step(discrete_action)
                done = terminated or truncated
                
                next_safe_actions_mask = np.zeros(4, dtype=np.float32)
                for sa in env.get_safe_actions(tuple(env.agent_pos)):
                    next_safe_actions_mask[sa] = 1.0
                
                transition = Transition(
                    obs=obs, action=discrete_action, reward=reward,
                    next_obs=next_obs, done=done, mode='deontic',
                    T_rem=info.get('T_rem', float('inf')),
                    D_min=info.get('D_min', 0),
                    L_safe=info.get('L_safe', float('inf')),
                    min_v=999, in_failure_band=info.get('in_failure_band', False),
                    next_in_failure_band=False, episode_id=self.episode_counter,
                    step_idx=step, next_safe_actions_mask=next_safe_actions_mask
                )
                self.buffer_deon.add(transition)
                obs = next_obs
                info = new_info
                step += 1
                step_count += 1
            self.episode_counter += 1

    def collect_rollouts_with_buffers(self):
        """Collect rollouts, populating both replay buffers.

        Returns on-policy data for PPO actor update.
        """
        observations = []
        continuous_actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        modes = []
        stakes_vals = []

        step_count = 0
        episode_rewards = []
        episode_violations = []
        episode_saves = []

        while step_count < self.steps_per_epoch:
            use_bomb = np.random.random() > 0.5
            timer = self.bomb_timer if use_bomb else None

            env = InstrumentedHostageEnv.make(
                layout_id=self.layout_id, bomb_timer=timer)
            obs, info = env.reset()
            ep_reward = 0
            done = False
            step = 0

            while not done and step_count < self.steps_per_epoch:
                obs_t = self._pad_obs(obs)
                observations.append(obs_t)

                stakes = self.agent.compute_stakes(env)
                mode = self.agent.get_mode(stakes)
                stakes_vals.append(stakes)
                modes.append(mode)

                # Get continuous action + log prob
                with torch.no_grad():
                    mean, log_std = self.actor_net(obs_t.unsqueeze(0))
                    mean = mean.squeeze(0)
                    std = log_std.squeeze(0).exp()
                    noise = torch.randn_like(mean)
                    continuous_action = mean + std * noise

                    _log_prob = -0.5 * (((continuous_action - mean) / (std + 1e-8))**2
                                        + 2 * log_std.squeeze(0) + np.log(2 * np.pi))
                    lp = _log_prob.sum()
                    v = self.value_net(obs_t.unsqueeze(0)).item()

                discrete_action = discretize_action(continuous_action)

                # Shield in deontic mode
                if mode == 'deontic':
                    safe = self.shield.get_safe_actions(env)
                    if discrete_action not in safe:
                        discrete_action = self.agent.deontic_selector.select_safe_action(
                            obs, env, safe)[0]

                continuous_actions.append(continuous_action)
                log_probs.append(lp.item())
                values.append(v)

                # Step
                next_obs, reward, terminated, truncated, new_info = env.step(discrete_action)
                done = terminated or truncated
                rewards.append(reward)
                dones.append(done)
                ep_reward += reward

                # Add to appropriate replay buffer
                next_safe_actions_mask = np.zeros(4, dtype=np.float32)
                for sa in env.get_safe_actions(tuple(env.agent_pos)):
                    next_safe_actions_mask[sa] = 1.0

                transition = Transition(
                    obs=obs, action=discrete_action, reward=reward,
                    next_obs=next_obs, done=done, mode=mode,
                    T_rem=info.get('T_rem', float('inf')),
                    D_min=info.get('D_min', 0),
                    L_safe=info.get('L_safe', float('inf')),
                    min_v=999,
                    in_failure_band=info.get('in_failure_band', False),
                    next_in_failure_band=False,
                    episode_id=self.episode_counter,
                    step_idx=step,
                    next_safe_actions_mask=next_safe_actions_mask
                )
                if mode == 'utilitarian':
                    self.buffer_free.add(transition)

                obs = next_obs
                info = new_info
                step += 1
                step_count += 1

            episode_rewards.append(ep_reward)
            episode_violations.append(len(env.trampled_civilians))
            saved = tuple(env.agent_pos) == env.hostage_pos
            episode_saves.append(saved)
            self.episode_counter += 1

        return {
            'observations': torch.stack(observations),
            'continuous_actions': torch.stack(continuous_actions),
            'rewards': rewards,
            'values': values,
            'log_probs': log_probs,
            'dones': dones,
            'modes': modes,
            'stakes': stakes_vals,
            'episode_rewards': episode_rewards,
            'episode_violations': episode_violations,
            'episode_saves': episode_saves,
        }

    def update_critics(self):
        """Inner-loop critic updates from replay buffers."""
        losses = {'z_free': [], 'z_deon': []}

        for _ in range(self.critic_updates_per_epoch):
            if len(self.buffer_free) >= self.critic_batch_size:
                batch = self.buffer_free.sample(self.critic_batch_size)
                
                with torch.no_grad():
                    mean, _ = self.actor_net(torch.FloatTensor(batch['next_obs']))
                    next_a_free = discretize_action(mean)
                    next_a_free = torch.LongTensor(next_a_free)

                l = self.dual_critics.update_z_free(
                    torch.FloatTensor(batch['obs']), torch.LongTensor(batch['actions']),
                    torch.FloatTensor(batch['rewards']), torch.FloatTensor(batch['next_obs']),
                    torch.FloatTensor(batch['dones']), next_a_free, self.gamma)
                losses['z_free'].append(l)

            if len(self.buffer_deon) >= self.critic_batch_size:
                batch = self.buffer_deon.sample(self.critic_batch_size)
                
                with torch.no_grad():
                    mean, _ = self.actor_net(torch.FloatTensor(batch['next_obs']))
                    next_a_deon = discretize_action(mean)
                    next_a_deon = torch.LongTensor(next_a_deon)

                l = self.dual_critics.update_z_deon(
                    torch.FloatTensor(batch['obs']), torch.LongTensor(batch['actions']),
                    torch.FloatTensor(batch['rewards']), torch.FloatTensor(batch['next_obs']),
                    torch.FloatTensor(batch['dones']), next_a_deon, self.gamma)
                losses['z_deon'].append(l)

        return {k: np.mean(v) if v else 0.0 for k, v in losses.items()}

    def compute_gae(self, rewards, values, dones):
        """GAE advantages + returns."""
        advantages = []
        gae = 0
        T = len(rewards)
        for t in reversed(range(T)):
            next_val = values[t + 1] if t < T - 1 else 0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        return advantages, returns

    def ppo_update(self, data, advantages, returns):
        """PPO actor + value critic update."""
        obs = data['observations']
        actions = data['continuous_actions']
        old_lp = torch.FloatTensor(data['log_probs'])

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        N = len(obs)
        bs = min(64, N)

        total_pl, total_vl, total_ent, n_up = 0, 0, 0, 0

        for _ in range(self.ppo_epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, bs):
                end = min(start + bs, N)
                i = idx[start:end]

                b_obs = obs[i]
                b_act = actions[i]
                b_old = old_lp[i]
                b_adv = advantages[i]
                b_ret = returns[i]

                mean, log_std = self.actor_net(b_obs)
                std = log_std.exp()
                new_lp = -0.5 * (((b_act - mean) / (std + 1e-8))**2
                                 + 2 * log_std + np.log(2 * np.pi))
                new_lp = new_lp.sum(dim=-1)

                ratio = torch.exp(new_lp - b_old)
                s1 = ratio * b_adv
                s2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                 1 + self.clip_epsilon) * b_adv
                pl = -torch.min(s1, s2).mean()
                ent = 0.5 * (1 + torch.log(2 * np.pi * std**2)).sum(-1).mean()
                vl = nn.functional.mse_loss(self.value_net(b_obs), b_ret)

                loss = pl - self.entropy_coef * ent + self.value_coef * vl

                self.actor_optim.zero_grad()
                self.value_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.actor_optim.step()
                self.value_optim.step()

                total_pl += pl.item()
                total_vl += vl.item()
                total_ent += ent.item()
                n_up += 1

        return {'policy_loss': total_pl / max(n_up, 1),
                'value_loss': total_vl / max(n_up, 1),
                'entropy': total_ent / max(n_up, 1)}

    def evaluate(self, n_episodes=50, bomb_timer=None, deterministic=True):
        """Evaluate current policy."""
        saves, violations, total_reward = 0, 0, 0
        modes = {'deontic': 0, 'utilitarian': 0}

        for _ in range(n_episodes):
            env = InstrumentedHostageEnv.make(
                layout_id=self.layout_id, bomb_timer=bomb_timer)
            obs, _ = env.reset()
            done = False
            ep_r = 0
            while not done:
                action, info = self.agent.select_action(obs, env, deterministic=deterministic)
                modes[info['mode']] += 1
                obs, r, term, trunc, _ = env.step(action)
                done = term or trunc
                ep_r += r
            total_reward += ep_r
            violations += len(env.trampled_civilians)
            if tuple(env.agent_pos) == env.hostage_pos:
                saves += 1

        return {
            'save_rate': saves / n_episodes,
            'avg_violations': violations / n_episodes,
            'avg_reward': total_reward / n_episodes,
            'mode_counts': dict(modes),
        }

    def train(self):
        """Full training loop: warmup → main loop."""
        os.makedirs(self.save_dir, exist_ok=True)
        stakes_type = 'oracle' if self.use_oracle_stakes else 'learned'

        print(f"\n{'='*60}")
        print(f"Full CDDP Trainer — Grid {self.layout_id}, T={self.bomb_timer}")
        print(f"τ={self.tau}, stakes={stakes_type}")
        print(f"Save: {self.save_dir}")
        print(f"{'='*60}")

        # Phase I: Deontic warmup
        self.deontic_warmup()

        # Phase II: Main loop
        print(f"\n{'─'*40}")
        print(f"Main Training Loop: {self.total_epochs} epochs")
        print(f"{'─'*40}\n")

        for epoch in range(self.total_epochs):
            t0 = time.time()

            # Collect rollouts (populates buffers)
            self.collect_deontic_rollouts(self.steps_per_epoch // 2)
            data = self.collect_rollouts_with_buffers()

            # Inner loop: critic updates
            critic_losses = {}
            if not self.use_oracle_stakes:
                critic_losses = self.update_critics()

            # Outer loop: actor PPO update
            advantages, returns = self.compute_gae(
                data['rewards'], data['values'], data['dones'])
            actor_losses = self.ppo_update(data, advantages, returns)

            elapsed = time.time() - t0

            # Stats
            avg_r = np.mean(data['episode_rewards'])
            avg_v = np.mean(data['episode_violations'])
            save_r = np.mean(data['episode_saves'])
            deon_f = sum(1 for m in data['modes'] if m == 'deontic') / len(data['modes'])
            avg_s = np.mean(data['stakes'])

            self.history['epoch'].append(epoch)
            self.history['avg_reward'].append(avg_r)
            self.history['avg_violations'].append(avg_v)
            self.history['save_rate'].append(save_r)
            self.history['deontic_frac'].append(deon_f)
            self.history['avg_stakes'].append(avg_s)
            self.history['policy_loss'].append(actor_losses['policy_loss'])
            self.history['value_loss'].append(actor_losses['value_loss'])
            if 'z_free' in critic_losses:
                self.history['z_free_loss'].append(critic_losses['z_free'])
                self.history['z_deon_loss'].append(critic_losses['z_deon'])

            if epoch % 10 == 0:
                crit_str = (f"  Zf={critic_losses.get('z_free', 0):.4f}"
                            f"  Zd={critic_losses.get('z_deon', 0):.4f}"
                            if not self.use_oracle_stakes else "")
                print(f"[{epoch:3d}] R={avg_r:7.2f}  S={save_r*100:5.1f}%  "
                      f"V={avg_v:.2f}  D={deon_f*100:.0f}%  "
                      f"PL={actor_losses['policy_loss']:.4f}"
                      f"{crit_str}  ({elapsed:.1f}s)")

            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            if (epoch + 1) % 50 == 0:
                self._run_evaluation(epoch)

        self._run_evaluation(self.total_epochs - 1)
        self._save_checkpoint(self.total_epochs - 1)
        self._save_history()

        print(f"\n{'='*60}")
        print(f"Training complete! Results: {self.save_dir}")
        print(f"{'='*60}")

    def _run_evaluation(self, epoch):
        print(f"\n  --- Eval (epoch {epoch}) ---")
        for label, timer in [('low', None), ('high', self.bomb_timer)]:
            r = self.evaluate(50, bomb_timer=timer)
            print(f"  {label:>4}: Save={r['save_rate']*100:.0f}%  "
                  f"Viol={r['avg_violations']:.2f}  "
                  f"R={r['avg_reward']:.1f}")
            self.history[f'eval_{label}_save_rate'].append(r['save_rate'])
            self.history[f'eval_{label}_violations'].append(r['avg_violations'])

    def _save_checkpoint(self, epoch):
        path = os.path.join(self.save_dir, f'cddp_full_epoch_{epoch}.pt')
        state = {
            'epoch': epoch,
            'pi': self.actor_net.state_dict(),
            'value': self.value_net.state_dict(),
            'critics': self.dual_critics.state_dict(),
            'tau': self.tau,
            'layout_id': self.layout_id,
        }
        torch.save(state, path)

    def _save_history(self):
        path = os.path.join(self.save_dir, 'training_history.json')
        serializable = {}
        for k, v in self.history.items():
            serializable[k] = [float(x) if isinstance(x, (np.floating, np.integer, bool)) else x
                               for x in v]
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"  ✓ History: {path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Full CDDP Trainer')
    parser.add_argument('--grid', type=int, default=1)
    parser.add_argument('--timer', type=int, default=5)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--quantiles', type=int, default=21)
    parser.add_argument('--oracle', action='store_true', help='Use oracle stakes')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    if args.save_dir is None:
        stype = 'oracle' if args.oracle else 'learned'
        args.save_dir = f'cddp_results/grid{args.grid}_{stype}_tau{args.tau}'

    trainer = FullCDDPTrainer(
        layout_id=args.grid,
        bomb_timer=args.timer,
        tau=args.tau,
        n_quantiles=args.quantiles,
        use_oracle_stakes=args.oracle,
        warmup_episodes=args.warmup,
        total_epochs=args.epochs,
        steps_per_epoch=args.steps,
        lr_actor=args.lr,
        save_dir=args.save_dir,
    )
    trainer.train()
