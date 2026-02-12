"""
PPO Training Script - IMPROVED Utilitarian Baseline

Improvements:
1. Higher entropy coefficient for better exploration
2. Train ONLY on High Stakes to enforce trampling
3. Batched updates
4. Best model checkpointing
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
try:
    from hostage_gridworld import HostageGridworld
except ImportError:
    from standard_RL.hostage_gridworld import HostageGridworld


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
        # Mask bomb features to force "context-blind" behavior
        # Obs: [x, y, steps, bomb_active, time_left] -> mask last 2
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
    """Perform PPO update."""
    
    observations = torch.FloatTensor(np.array(observations))
    actions = torch.LongTensor(actions)
    old_log_probs = torch.FloatTensor(old_log_probs)
    advantages = torch.FloatTensor(advantages)
    returns = torch.FloatTensor(returns)
    
    # Normalize advantages
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
            
            # Forward pass
            action_logits, values = agent(batch_obs)
            dist = Categorical(logits=action_logits)
            
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            # PPO objective
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
            
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()


def train_ppo(env, agent, optimizer, num_episodes=500, max_steps=50, print_every=100):
    """Train PPO agent with BATCHED UPDATES."""
    
    episode_rewards = []
    episode_civilian_violations = []
    episode_lengths = []
    success_count = 0
    best_success_rate = 0.0
    
    # Batch buffers
    batch_observations = []
    batch_actions = []
    batch_log_probs = []
    batch_returns = []
    batch_advantages = []
    
    batch_size_episodes = 20
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        # Episode buffers
        ep_observations = []
        ep_actions = []
        ep_log_probs = []
        ep_values = []
        ep_rewards = []
        ep_dones = []
        
        episode_reward = 0
        
        for step in range(max_steps):
            # Get action
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
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            
            ep_rewards.append(reward)
            ep_dones.append(terminated or truncated)
            episode_reward += reward
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Track success
        if info['agent_pos'] == env.hostage_pos:
            success_count += 1
        
        # Compute advantages
        ep_advantages = compute_gae(ep_rewards, ep_values, ep_dones)
        ep_returns_computed = [adv + val for adv, val in zip(ep_advantages, ep_values)]
        
        # Accumulate to batch
        batch_observations.extend(ep_observations)
        batch_actions.extend(ep_actions)
        batch_log_probs.extend(ep_log_probs)
        batch_advantages.extend(ep_advantages)
        batch_returns.extend(ep_returns_computed)
        
        # PPO update periodically
        if (episode + 1) % batch_size_episodes == 0:
            # reduced entropy for stability
            ppo_update(agent, optimizer, batch_observations, batch_actions, 
                      batch_log_probs, batch_advantages, batch_returns,
                      entropy_coef=0.3)
            
            # Reset batch buffers
            batch_observations = []
            batch_actions = []
            batch_log_probs = []
            batch_returns = []
            batch_advantages = []
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_civilian_violations.append(len(info['civilians_trampled_this_episode']))
        episode_lengths.append(step + 1)
        
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_violations = np.mean(episode_civilian_violations[-print_every:])
            avg_length = np.mean(episode_lengths[-print_every:])
            success_rate = success_count / print_every
            
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Violations: {avg_violations:.2f} | "
                  f"Avg Length: {avg_length:.2f} | "
                  f"Success Rate: {success_rate*100:.1f}%")
            
            # Save if best
            if success_rate >= best_success_rate and success_rate > 0:
                best_success_rate = success_rate
                torch.save(agent.state_dict(), 'best_temp_ppo.pt')
                print(f"  * New Best Model (Success: {success_rate*100:.1f}%) saved!")
            
            success_count = 0
            
    # Load best model if found
    if best_success_rate > 0:
        print(f"\nLoading best model with success rate: {best_success_rate*100:.1f}%")
        agent.load_state_dict(torch.load('best_temp_ppo.pt'))
    
    return episode_rewards, episode_civilian_violations, episode_lengths


def evaluate_agent(env, agent, num_episodes=100):
    """Evaluate trained agent."""
    total_rewards = []
    civilian_violations = []
    bomb_explosions = 0
    hostages_saved = 0
    paths_taken = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        path = [tuple(info['agent_pos'])]
        
        for _ in range(50):
            action = agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            path.append(tuple(info['agent_pos']))
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        civilian_violations.append(len(info['civilians_trampled_this_episode']))
        
        if info['agent_pos'] == env.hostage_pos:
            hostages_saved += 1
        
        if env.bomb_timer_max is not None and info['bomb_time_left'] == 0 and info['agent_pos'] != env.hostage_pos:
            bomb_explosions += 1
        
        paths_taken.append(path)
    
    results = {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_civilian_violations': np.mean(civilian_violations),
        'hostages_saved': hostages_saved,
        'hostage_save_rate': hostages_saved / num_episodes,
        'bomb_explosions': bomb_explosions,
        'bomb_explosion_rate': bomb_explosions / num_episodes,
        'paths': paths_taken[:5]
    }
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=1, help="Grid layout ID (1-3)")
    args = parser.parse_args()
    
    print("="*60)
    print(f"TRAINING UTILITARIAN BASELINE (PPO) - GRID {args.grid}")
    print("HIGH STAKES ONLY")
    print("="*60)
    
    # Get dynamic timer
    optimal_timer = HostageGridworld.GRID_LAYOUTS[args.grid]['bomb_timer']
    print(f"Using dynamic bomb timer: {optimal_timer} steps (for Grid {args.grid})")
    
    # Use bomb_timer=optimal_timer
    env_high_stakes = HostageGridworld(bomb_timer=optimal_timer, layout_id=args.grid)
    
    obs_dim = env_high_stakes.observation_space.shape[0]
    
    # Scale parameters based on grid size
    grid_size = env_high_stakes.grid_size
    hidden_dim = 128 if grid_size <= 5 else 256  # Larger network for larger grids
    num_episodes = 15000 if grid_size <= 5 else 30000  # More episodes for larger grids
    
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Hidden dim: {hidden_dim}, Episodes: {num_episodes}")
    
    agent = PPOAgent(obs_dim=obs_dim, action_dim=4, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    
    # Train directly on high stakes with scaled episodes
    train_ppo(env_high_stakes, agent, optimizer, num_episodes=num_episodes, print_every=500)
    
    print("\n[Evaluation] High Stakes...")
    results_high = evaluate_agent(env_high_stakes, agent, num_episodes=100)
    print(f"High Stakes Results:")
    print(f"  Success Rate: {results_high['hostage_save_rate']*100:.1f}%")
    print(f"  Avg Violations: {results_high['avg_civilian_violations']:.2f}")
    print(f"  Avg Reward: {results_high['avg_reward']:.2f}")
    print(f"  Bomb Explosion Rate: {results_high['bomb_explosion_rate']*100:.1f}%")
    
    # Save final model
    filename = f'ppo_high_stakes_grid{args.grid}.pt'
    torch.save(agent.state_dict(), filename)
    print(f"\n✓ Saved final model: {filename}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
 
