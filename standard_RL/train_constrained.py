"""
Constrained RL Training Script - IMPROVED Deontological Baseline

Improvements:
1. Phase 0: Train with regular PPO (no constraint interference)
2. Transfer weights to Constrained agent
3. Phase 1-2: Add constraints gradually
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from hostage_gridworld import HostageGridworld
from train_ppo import PPOAgent, train_ppo  # Import for Phase 0


class ConstrainedPPOAgent(nn.Module):
    """PPO agent with constrained optimization."""
    
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
        self.cost_critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs):
        features = self.shared(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        cost_value = self.cost_critic(features)
        return action_logits, value, cost_value
    
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_logits, value, cost_value = self.forward(obs_tensor)
            
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


def constrained_ppo_update(agent, optimizer, observations, actions, old_log_probs, 
                           advantages, returns, cost_advantages, cost_returns,
                           lambda_penalty, constraint_threshold=0.1,
                           clip_epsilon=0.2, value_coef=0.5, cost_coef=0.5, 
                           entropy_coef=0.1, epochs=4, batch_size=64):
    """Constrained PPO update with HIGHER ENTROPY."""
    
    observations = torch.FloatTensor(np.array(observations))
    actions = torch.LongTensor(actions)
    old_log_probs = torch.FloatTensor(old_log_probs)
    advantages = torch.FloatTensor(advantages)
    returns = torch.FloatTensor(returns)
    cost_advantages = torch.FloatTensor(cost_advantages)
    cost_returns = torch.FloatTensor(cost_returns)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
    
    # Lagrangian combination
    combined_advantages = advantages - lambda_penalty * cost_advantages
    
    dataset_size = len(observations)
    
    for _ in range(epochs):
        indices = np.random.permutation(dataset_size)
        
        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_indices = indices[start:end]
            
            batch_obs = observations[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_combined_advantages = combined_advantages[batch_indices]
            batch_returns = returns[batch_indices]
            batch_cost_returns = cost_returns[batch_indices]
            
            # Forward pass
            action_logits, values, cost_values = agent(batch_obs)
            dist = Categorical(logits=action_logits)
            
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            # PPO objective
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_combined_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_combined_advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
            cost_value_loss = nn.MSELoss()(cost_values.squeeze(), batch_cost_returns)
            
            # Higher entropy
            loss = policy_loss + value_coef * value_loss + cost_coef * cost_value_loss - entropy_coef * entropy
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()


def train_constrained_ppo(env, agent, optimizer, num_episodes=500, max_steps=50, 
                          constraint_threshold=0.0, lambda_lr=0.01, print_every=50,
                          initial_lambda=0.1):
    """Train constrained PPO with adaptive lambda."""
    
    episode_rewards = []
    episode_civilian_violations = []
    episode_lengths = []
    lambda_values = []
    success_count = 0
    
    # Start with LOW lambda to allow exploration
    lambda_penalty = initial_lambda
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        observations = []
        actions_taken = []
        rewards_received = []
        costs_received = []
        values = []
        cost_values = []
        log_probs = []
        dones = []
        costs = []
        
        episode_reward = 0
        episode_cost = 0
        prev_violations = 0
        
        for step in range(max_steps):
            # Get action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action_logits, value, cost_value = agent(obs_tensor)
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            observations.append(obs)
            actions_taken.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
            cost_values.append(cost_value.item())
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            
            # Cost = 1 if civilian trampled this step
            current_violations = len(info['civilians_trampled_this_episode'])
            cost = 1.0 if current_violations > prev_violations else 0.0
            prev_violations = current_violations
            
            rewards_received.append(reward)
            costs_received.append(cost)
            dones.append(terminated or truncated)
            episode_reward += reward
            episode_cost += cost
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Track success
        if info['agent_pos'] == env.hostage_pos:
            success_count += 1
        
        # Compute advantages
        advantages = compute_gae(rewards_received, values, dones)
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        cost_advantages = compute_gae(costs_received, cost_values, dones)
        cost_returns = [adv + val for adv, val in zip(cost_advantages, cost_values)]
        
        # Update agent
        constrained_ppo_update(agent, optimizer, observations, actions_taken, log_probs, 
                              advantages, returns, cost_advantages, cost_returns,
                              lambda_penalty, constraint_threshold)
        
        # Update lambda (slowly increase penalty)
        violation_excess = episode_cost - constraint_threshold
        lambda_penalty = max(0.1, lambda_penalty + lambda_lr * violation_excess)
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_civilian_violations.append(int(episode_cost))
        episode_lengths.append(step + 1)
        lambda_values.append(lambda_penalty)
        
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_violations = np.mean(episode_civilian_violations[-print_every:])
            avg_length = np.mean(episode_lengths[-print_every:])
            avg_lambda = np.mean(lambda_values[-print_every:])
            success_rate = success_count / print_every
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Reward: {avg_reward:.2f} | "
                  f"Violations: {avg_violations:.2f} | "
                  f"Length: {avg_length:.2f} | "
                  f"Lambda: {avg_lambda:.2f} | "
                  f"Success: {success_rate*100:.1f}%")
            success_count = 0
    
    return episode_rewards, episode_civilian_violations, episode_lengths, lambda_values


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
    print("="*60)
    print("TRAINING DEONTOLOGICAL BASELINE (Constrained RL)")
    print("WITH CURRICULUM LEARNING")
    print("="*60)
    
    # PHASE 0: Train with regular PPO (no constraint interference)
    print("\n[PHASE 0/3] Training navigation with regular PPO...")
    print("Goal: Learn to reach hostage without constraint penalties")
    env_low_stakes = HostageGridworld(bomb_timer=None)
    obs_dim = env_low_stakes.observation_space.shape[0]
    
    # Use regular PPO agent
    ppo_agent = PPOAgent(obs_dim=obs_dim, action_dim=4, hidden_dim=128)
    ppo_optimizer = torch.optim.Adam(ppo_agent.parameters(), lr=3e-4)
    
    # Train for navigation
    train_ppo(env_low_stakes, ppo_agent, ppo_optimizer, num_episodes=2000, print_every=200)
    
    print("\n[Weight Transfer] Transferring PPO weights to Constrained agent...")
    # Create constrained agent and transfer learned weights
    agent = ConstrainedPPOAgent(obs_dim=obs_dim, action_dim=4, hidden_dim=128)
    agent.shared.load_state_dict(ppo_agent.shared.state_dict())
    agent.actor.load_state_dict(ppo_agent.actor.state_dict())
    agent.critic.load_state_dict(ppo_agent.critic.state_dict())
    # cost_critic starts from scratch (random init)
    print("✓ Transferred shared, actor, and critic weights")
    print("✓ Cost critic initialized randomly\n")
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    
    # PHASE 1: Introduce constraints VERY gradually (preserve navigation!)
    print("\n[PHASE 1/3] Introducing constraint penalty (GENTLE)...")
    print("Goal: Maintain success while GENTLY reducing violations")
    train_constrained_ppo(env_low_stakes, agent, optimizer, 
                         num_episodes=2000, constraint_threshold=2.0,  # Very relaxed
                         lambda_lr=0.005, print_every=200, initial_lambda=0.05)  # Very slow growth
    
    # PHASE 2: Tighten constraint (still gentle)
    print("\n[PHASE 2/3] Tightening constraint (MODERATE)...")
    train_constrained_ppo(env_low_stakes, agent, optimizer,
                         num_episodes=2000, constraint_threshold=0.5,  # More strict but not zero
                         lambda_lr=0.01, print_every=200, initial_lambda=0.2)  # Still conservative
    
    print("\n[Checkpoint] Evaluating on LOW STAKES...")
    results_low = evaluate_agent(env_low_stakes, agent, num_episodes=100)
    print(f"Low Stakes Results:")
    print(f"  Success Rate: {results_low['hostage_save_rate']*100:.1f}%")
    print(f"  Avg Violations: {results_low['avg_civilian_violations']:.2f}")
    print(f"  Avg Reward: {results_low['avg_reward']:.2f}")
    
    torch.save(agent.state_dict(), 'constrained_low_stakes.pt')
    print("\n✓ Saved checkpoint: constrained_low_stakes.pt")
    
    # PHASE 3: Deploy on HIGH STAKES (expect failure - bomb explodes)
    print("\n[PHASE 3/3] Testing on HIGH STAKES...")
    env_high_stakes = HostageGridworld(bomb_timer=5)
    
    # Train briefly on high stakes to see if it adapts (it shouldn't)
    train_constrained_ppo(env_high_stakes, agent, optimizer,
                         num_episodes=2000, constraint_threshold=0.0,
                         lambda_lr=0.05, print_every=200, initial_lambda=2.0)
    
    results_high = evaluate_agent(env_high_stakes, agent, num_episodes=100)
    print(f"\nHigh Stakes Results:")
    print(f"  Success Rate: {results_high['hostage_save_rate']*100:.1f}%")
    print(f"  Avg Violations: {results_high['avg_civilian_violations']:.2f}")
    print(f"  Avg Reward: {results_high['avg_reward']:.2f}")
    print(f"  Bomb Explosion Rate: {results_high['bomb_explosion_rate']*100:.1f}%")
    
    torch.save(agent.state_dict(), 'constrained_high_stakes.pt')
    print("\n✓ Saved final model: constrained_high_stakes.pt")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("Expected: High bomb explosion rate (fanatical deontologist)")
    print("="*60)
