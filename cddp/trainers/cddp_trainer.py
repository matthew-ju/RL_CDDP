"""
CDDP Trainer — Phase 2b: PPO with CDDP mode switching and oracle stakes.

Training flow:
    1. Rollout: collect episodes using CDDPAgent (mode + shield)
    2. In deontic mode: modulated cost C̃(s,a) = 0 (shield already enforces safety)
    3. In utilitarian mode: modulated cost C̃(s,a) = 0 (constraints lifted)
    4. PPO update on actor + value critic using collected trajectories

This is the **oracle-stakes** version (Phase 2b): stakes are computed from
D_min/T_rem (exact BFS), not from learned distributional critics.

Architecture notes:
    - Actor: 3-layer MLP (obs → 64 → 64 → 2), Gaussian policy
    - Critic: separate 3-layer MLP (obs → 64 → 64 → 1), value function
    - Action: 2D continuous → quadrant discretized → grid movement
    - Observations: padded from 7-dim (env) to 9-dim (OmniSafe compat)
"""

import os
import sys
import time
import json
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
from cddp.agents.ppo_actor import PPOActor, OmniSafeGaussianActor, discretize_action
from cddp.agents.cddp_agent import CDDPAgent


class ValueCritic(nn.Module):
    """Value function V(s) for PPO advantage estimation."""

    def __init__(self, obs_dim: int = 9, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class CDDPTrainer:
    """PPO trainer with CDDP mode switching.

    Args:
        layout_id: Grid layout (1, 2, 3)
        bomb_timer: Bomb timer for high-stakes training
        tau: Mode-switching threshold
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE parameter
        clip_epsilon: PPO clip
        entropy_coef: Entropy bonus
        value_coef: Value loss coefficient
        epochs_per_update: PPO epochs per batch
        steps_per_epoch: Steps to collect per rollout batch
        hidden_dim: Network hidden layer size
        total_epochs: Total training epochs
        save_dir: Directory for checkpoints
        checkpoint_interval: Save every N epochs
    """

    def __init__(
        self,
        layout_id: int = 1,
        bomb_timer: int = 5,
        tau: float = 0.5,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        epochs_per_update: int = 10,
        steps_per_epoch: int = 1000,
        hidden_dim: int = 64,
        total_epochs: int = 300,
        save_dir: str = 'cddp_results',
        checkpoint_interval: int = 50,
    ):
        self.layout_id = layout_id
        self.bomb_timer = bomb_timer
        self.tau = tau
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs_per_update = epochs_per_update
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.save_dir = save_dir
        self.checkpoint_interval = checkpoint_interval

        # Actor and critic
        obs_dim = 9  # 7 env + 2 padding
        self.actor_net = OmniSafeGaussianActor(obs_dim=obs_dim, action_dim=2,
                                                hidden_dim=hidden_dim)
        self.value_net = ValueCritic(obs_dim=obs_dim, hidden_dim=hidden_dim)

        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=lr)
        self.value_optim = optim.Adam(self.value_net.parameters(), lr=lr)

        # CDDP components
        self.actor_wrapper = PPOActor(obs_dim=obs_dim, hidden_dim=hidden_dim)
        self.actor_wrapper.actor = self.actor_net  # share weights
        self.shield = ActionShield()
        self.stakes_fn = AnalyticalStakesDmin()
        self.agent = CDDPAgent(self.actor_wrapper, self.shield, self.stakes_fn, tau=tau)

        # Training history
        self.history = defaultdict(list)

    def _pad_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Pad 7-dim env obs to 9-dim."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[-1] < 9:
            padding = np.zeros(9 - obs.shape[-1], dtype=np.float32)
            obs = np.concatenate([obs, padding])
        return torch.FloatTensor(obs)

    def collect_rollouts(self, env_bomb_timer=None):
        """Collect trajectories using CDDP agent.

        Returns dict of trajectory data (lists of per-step values).
        """
        if env_bomb_timer is None:
            # Alternate: 50% high-stakes, 50% low-stakes episodes
            use_bomb = np.random.random() > 0.5
            timer = self.bomb_timer if use_bomb else None
        else:
            timer = env_bomb_timer

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
            # Alternate stakes per episode
            use_bomb = np.random.random() > 0.5
            timer = self.bomb_timer if use_bomb else None

            env = InstrumentedHostageEnv.make(
                layout_id=self.layout_id,
                bomb_timer=timer,
            )
            obs, info = env.reset()
            ep_reward = 0
            ep_violations = 0
            done = False

            while not done and step_count < self.steps_per_epoch:
                obs_t = self._pad_obs(obs)
                observations.append(obs_t)

                # CDDP mode switching
                stakes = self.agent.compute_stakes(env)
                mode = self.agent.get_mode(stakes)
                stakes_vals.append(stakes)
                modes.append(mode)

                # Get continuous action from Gaussian + log_prob
                with torch.no_grad():
                    mean, log_std = self.actor_net(obs_t.unsqueeze(0))
                    mean = mean.squeeze(0)
                    std = log_std.squeeze(0).exp()
                    noise = torch.randn_like(mean)
                    continuous_action = mean + std * noise

                    # Log prob under Gaussian
                    _log_prob = -0.5 * (((continuous_action - mean) / (std + 1e-8))**2
                                        + 2 * log_std.squeeze(0) + np.log(2 * np.pi))
                    lp = _log_prob.sum()

                    # Value estimate
                    v = self.value_net(obs_t.unsqueeze(0)).item()

                # Discretize
                discrete_action = discretize_action(continuous_action)

                # Shield in deontic mode
                if mode == 'deontic':
                    safe = self.shield.get_safe_actions(env)
                    if discrete_action not in safe:
                        # Deterministic deontic selection (replaces random MC fallback)
                        discrete_action = self.agent.deontic_selector.select_safe_action(
                            obs, env, safe)[0]

                continuous_actions.append(continuous_action)
                log_probs.append(lp.item())
                values.append(v)

                # Step environment
                obs_next, reward, terminated, truncated, info = env.step(discrete_action)
                done = terminated or truncated

                rewards.append(reward)
                dones.append(done)
                ep_reward += reward

                if len(env.trampled_civilians) > 0:
                    ep_violations = len(env.trampled_civilians)

                obs = obs_next
                step_count += 1

            episode_rewards.append(ep_reward)
            episode_violations.append(ep_violations)
            saved = tuple(env.agent_pos) == env.hostage_pos if not env.multi_hostage else len(env.env.hostages_saved) > 0
            episode_saves.append(saved)

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

    def compute_gae(self, rewards, values, dones):
        """Compute GAE advantages and returns."""
        advantages = []
        gae = 0
        T = len(rewards)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        return advantages, returns

    def ppo_update(self, rollout_data, advantages, returns):
        """PPO update on actor and value networks."""
        obs = rollout_data['observations']
        actions = rollout_data['continuous_actions']
        old_log_probs = torch.FloatTensor(rollout_data['log_probs'])

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(obs)
        batch_size = min(64, dataset_size)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(self.epochs_per_update):
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                idx = indices[start:end]

                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_old_lp = old_log_probs[idx]
                batch_adv = advantages[idx]
                batch_ret = returns[idx]

                # Actor: compute new log probs
                mean, log_std = self.actor_net(batch_obs)
                std = log_std.exp()
                new_log_probs = -0.5 * (((batch_actions - mean) / (std + 1e-8))**2
                                        + 2 * log_std + np.log(2 * np.pi))
                new_log_probs = new_log_probs.sum(dim=-1)

                # PPO ratio and clip
                ratio = torch.exp(new_log_probs - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy = 0.5 * (1 + torch.log(2 * np.pi * std**2)).sum(dim=-1).mean()

                # Value loss
                values = self.value_net(batch_obs)
                value_loss = nn.functional.mse_loss(values, batch_ret)

                # Total loss
                loss = (policy_loss
                        - self.entropy_coef * entropy
                        + self.value_coef * value_loss)

                self.actor_optim.zero_grad()
                self.value_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.actor_optim.step()
                self.value_optim.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }

    def evaluate(self, n_episodes: int = 50, bomb_timer=None, deterministic=True):
        """Evaluate current CDDP policy.

        Returns dict of metrics.
        """
        saves = 0
        violations = 0
        total_reward = 0
        mode_counts = {'deontic': 0, 'utilitarian': 0}
        shield_overrides = 0

        for _ in range(n_episodes):
            env = InstrumentedHostageEnv.make(
                layout_id=self.layout_id,
                bomb_timer=bomb_timer,
            )
            obs, _ = env.reset()
            done = False
            ep_reward = 0

            while not done:
                action, info = self.agent.select_action(
                    obs, env, deterministic=deterministic)
                mode_counts[info['mode']] += 1
                if info.get('action_was_shielded'):
                    shield_overrides += 1

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward

            total_reward += ep_reward
            violations += len(env.trampled_civilians)
            if tuple(env.agent_pos) == env.hostage_pos:
                saves += 1

        return {
            'save_rate': saves / n_episodes,
            'avg_violations': violations / n_episodes,
            'avg_reward': total_reward / n_episodes,
            'mode_counts': dict(mode_counts),
            'shield_overrides': shield_overrides,
        }

    def train(self):
        """Main training loop."""
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"CDDP Trainer — Grid {self.layout_id}, T={self.bomb_timer}, τ={self.tau}")
        print(f"Save: {self.save_dir}")
        print(f"{'='*60}\n")

        best_save_rate = -1

        for epoch in range(self.total_epochs):
            t0 = time.time()

            # Collect rollouts
            data = self.collect_rollouts()

            # Compute GAE
            advantages, returns = self.compute_gae(
                data['rewards'], data['values'], data['dones'])

            # PPO update
            losses = self.ppo_update(data, advantages, returns)

            elapsed = time.time() - t0

            # Episode stats
            avg_ep_reward = np.mean(data['episode_rewards'])
            avg_violations = np.mean(data['episode_violations'])
            save_rate = np.mean(data['episode_saves'])
            deontic_frac = sum(1 for m in data['modes'] if m == 'deontic') / len(data['modes'])

            # Log
            self.history['epoch'].append(epoch)
            self.history['avg_reward'].append(avg_ep_reward)
            self.history['avg_violations'].append(avg_violations)
            self.history['save_rate'].append(save_rate)
            self.history['deontic_frac'].append(deontic_frac)
            self.history['policy_loss'].append(losses['policy_loss'])
            self.history['value_loss'].append(losses['value_loss'])
            self.history['entropy'].append(losses['entropy'])

            if epoch % 10 == 0:
                print(f"[Epoch {epoch:3d}] "
                      f"R={avg_ep_reward:7.2f}  "
                      f"Save={save_rate*100:5.1f}%  "
                      f"Viol={avg_violations:.2f}  "
                      f"Deon={deontic_frac*100:.0f}%  "
                      f"PL={losses['policy_loss']:.4f}  "
                      f"VL={losses['value_loss']:.4f}  "
                      f"H={losses['entropy']:.3f}  "
                      f"({elapsed:.1f}s)")

            # Checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0 or epoch == self.total_epochs - 1:
                self._save_checkpoint(epoch)

            # Periodic evaluation
            if (epoch + 1) % 50 == 0:
                self._run_evaluation(epoch)

        # Final evaluation
        self._run_evaluation(self.total_epochs - 1)
        self._save_history()

        print(f"\n{'='*60}")
        print(f"Training complete! Results in {self.save_dir}")
        print(f"{'='*60}")

    def _run_evaluation(self, epoch):
        """Run evaluation on both low and high stakes."""
        print(f"\n  --- Evaluation (epoch {epoch}) ---")

        for label, timer in [('low', None), ('high', self.bomb_timer)]:
            results = self.evaluate(n_episodes=50, bomb_timer=timer)
            print(f"  {label:>4} stakes: "
                  f"Save={results['save_rate']*100:.0f}%  "
                  f"Viol={results['avg_violations']:.2f}  "
                  f"R={results['avg_reward']:.1f}  "
                  f"Modes={results['mode_counts']}")

            self.history[f'eval_{label}_save_rate'].append(results['save_rate'])
            self.history[f'eval_{label}_violations'].append(results['avg_violations'])

    def _save_checkpoint(self, epoch):
        """Save actor + value networks."""
        ckpt_path = os.path.join(self.save_dir, f'cddp_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'pi': self.actor_net.state_dict(),
            'value': self.value_net.state_dict(),
            'tau': self.tau,
            'layout_id': self.layout_id,
        }, ckpt_path)

    def _save_history(self):
        """Save training history as JSON."""
        hist_path = os.path.join(self.save_dir, 'training_history.json')
        serializable = {}
        for k, v in self.history.items():
            serializable[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
        with open(hist_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"  ✓ History saved: {hist_path}")


# ─── CLI ─────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CDDP Trainer (oracle stakes)')
    parser.add_argument('--grid', type=int, default=1)
    parser.add_argument('--timer', type=int, default=5)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f'cddp_results/grid{args.grid}_tau{args.tau}'

    trainer = CDDPTrainer(
        layout_id=args.grid,
        bomb_timer=args.timer,
        tau=args.tau,
        lr=args.lr,
        total_epochs=args.epochs,
        steps_per_epoch=args.steps,
        save_dir=args.save_dir,
    )
    trainer.train()
