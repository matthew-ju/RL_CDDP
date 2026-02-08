"""
Constrained RL Training Script - SIMPLIFIED Deontological Baseline

Goal: Train an agent that NEVER violates constraints (trampling civilians), 
even if it means failing the mission (running out of time).

Method:
1. Train purely on Low Stakes (No Bomb).
2. Use a high, fixed penalty for trampling.
3. The resulting policy will be "Safe but Slow".
4. When evaluated on High Stakes, it will fail safely (Time out).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
try:
    from hostage_gridworld import HostageGridworld
except ImportError:
    from standard_RL.hostage_gridworld import HostageGridworld
import argparse

# --- Reusing PPO Agent Architecture ---
class PPOAgent(nn.Module):
    """PPO agent with actor-critic architecture."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs):
        # Mask bomb features to consistency with PPO baseline
        # (Though here we want it to be rigid regardless of context)
        masked_obs = obs.clone()
        if len(masked_obs.shape) == 1:
            masked_obs[-2:] = 0
        else:
            masked_obs[:, -2:] = 0
            
        features = self.shared(masked_obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_logits, value = self.forward(obs_tensor)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                dist = Categorical(logits=action_logits)
                action = dist.sample()
            
            return action.item()

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    return advantages

def ppo_update(agent, optimizer, observations, actions, old_log_probs, advantages, returns,
                clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.05, epochs=4, batch_size=64):
    """Standard PPO update."""
    
    observations = torch.FloatTensor(np.array(observations))
    actions = torch.LongTensor(actions)
    old_log_probs = torch.FloatTensor(old_log_probs)
    advantages = torch.FloatTensor(advantages)
    returns = torch.FloatTensor(returns)
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    dataset_size = len(observations)
    
    for _ in range(epochs):
        indices = np.random.permutation(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_indices = indices[start:end]
            
            batch_obs = observations[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            
            action_logits, values = agent(batch_obs)
            dist = Categorical(logits=action_logits)
            
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
            
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

def train_agent(env, agent, optimizer, num_episodes=15000, max_steps=200, print_every=500): # max_steps original = 50
    
    episode_rewards = []
    success_count = 0
    
    # Batch buffers
    batch_observations = []
    batch_actions = []
    batch_log_probs = []
    batch_returns = []
    batch_advantages = []
    
    batch_size_episodes = 20
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        ep_observations = []
        ep_actions = []
        ep_log_probs = []
        ep_values = []
        ep_rewards = []
        ep_dones = []
        
        episode_reward = 0
        
        for step in range(max_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action_logits, value = agent(obs_tensor)
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            ep_observations.append(obs)
            ep_actions.append(action.item())
            ep_log_probs.append(log_prob.item())
            ep_values.append(value.item())
            
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            
            # --- HARD CONSTRAINT LOGIC ---
            # Civilians are now walls (handled by env).
            # No extra penalties needed.
            
            ep_rewards.append(reward)
            ep_dones.append(terminated or truncated)
            episode_reward += reward
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        if info['agent_pos'] == env.hostage_pos:
            success_count += 1
            
        ep_advantages = compute_gae(ep_rewards, ep_values, ep_dones)
        ep_returns_computed = [adv + val for adv, val in zip(ep_advantages, ep_values)]
        
        batch_observations.extend(ep_observations)
        batch_actions.extend(ep_actions)
        batch_log_probs.extend(ep_log_probs)
        batch_advantages.extend(ep_advantages)
        batch_returns.extend(ep_returns_computed)
        
        if (episode + 1) % batch_size_episodes == 0:
            ppo_update(agent, optimizer, batch_observations, batch_actions, 
                      batch_log_probs, batch_advantages, batch_returns)
            batch_observations = []
            batch_actions = []
            batch_log_probs = []
            batch_returns = []
            batch_advantages = []
            
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            success_rate = success_count / print_every
            
            print(f"[HARD CONSTRAINT] Ep {episode+1} | Reward: {avg_reward:.1f} | Success: {success_rate*100:.1f}%")
            success_count = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=1, help="Grid layout ID (1-3)")
    args = parser.parse_args()
    
    print("="*60)
    print(f"TRAINING CONSTRAINED BASELINE - GRID {args.grid}")
    print("STRATEGY: HARD CONSTRAINTS (Civilians = Walls)")
    print("="*60)
    
    # Train on Low Stakes (No Bomb) with HARD CONSTRAINTS (Civilians = Walls)
    env = HostageGridworld(bomb_timer=None, layout_id=args.grid, hard_constraint=True)
    
    # Scale parameters based on grid size (10x10 needs more capacity)
    grid_size = env.grid_size
    hidden_dim = 128 if grid_size <= 5 else 256  # Larger network for larger grids
    num_episodes = 15000 if grid_size <= 5 else 40000  # Even more episodes for constrained learning
    
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Hidden dim: {hidden_dim}, Episodes: {num_episodes}")
    print(f"Civilians treated as walls: {env.hard_constraint}")
    
    # BFS-GUIDED STRATEGY
    # The environment now uses BFS distance, so shaping guides AROUND civilians.
    env.DISTANCE_REWARD_SCALE = 2.0
    
    obs_dim = env.observation_space.shape[0]
    # Standard PPO configuration
    agent = PPOAgent(obs_dim=obs_dim, action_dim=4, hidden_dim=hidden_dim)
    
    # Use low entropy (0.05) to stabilize learned policy
    def ppo_update_wrapper(*args, **kwargs):
        return ppo_update(*args, **kwargs, entropy_coef=0.05)
        
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    
    # Train directly (it's just a maze now) - scaled episodes with stable policy
    print("\n--- TRAINING (Solving the Maze - Stable Policy) ---")
    train_agent(env, agent, optimizer, num_episodes=num_episodes)
    
    # Save as both Low and High stakes model files
    torch.save(agent.state_dict(), f'constrained_low_stakes_grid{args.grid}.pt')
    torch.save(agent.state_dict(), f'constrained_high_stakes_grid{args.grid}.pt')
    
    print(f"\n✓ Saved models for Grid {args.grid}")
    print("="*60)
