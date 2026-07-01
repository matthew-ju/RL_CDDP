"""
Comprehensive Evaluation Script — Multi-Seed, Stochastic, Statistical Tests

Evaluates OmniSafe-trained agents with:
  - Deterministic AND stochastic evaluation
  - Multi-seed aggregation (mean ± STD)
  - Statistical significance tests (Mann-Whitney U)
  - Safety performance function scoring
  - Scaled visualizations for any grid size
  - Publication-quality plots with error bars & significance stars
  - DeepMind-style performance-vs-timesteps plot
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hostage_gridworld import HostageGridworld
from safety_performance import compute_safety_performance, compute_batch_safety_performance

try:
    import hostage_gridworld_adapter
except ImportError:
    pass

try:
    import omnisafe
    OMNISAFE_AVAILABLE = True
except ImportError:
    OMNISAFE_AVAILABLE = False
    print("Warning: OmniSafe not available.")


# ============================================================================
# Model Loading
# ============================================================================

def load_omnisafe_agent(model_path, env_id):
    """Load an OmniSafe-trained model."""
    if not OMNISAFE_AVAILABLE or not os.path.exists(model_path):
        return None
    
    try:
        algo = 'PPO'
        if 'constrained' in model_path.lower() or 'lag' in model_path.lower():
            algo = 'PPOLag'
            
        custom_cfgs = {
            'algo_cfgs': {'obs_normalize': False}
        }
        if algo == 'PPOLag':
            custom_cfgs['lagrange_cfgs'] = {
                'cost_limit': 0.0,
                'lagrangian_multiplier_init': 0.001,
                'lambda_lr': 0.01,
                'lambda_optimizer': 'Adam',
            }

        agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
        
        loaded = False
        
        if hasattr(agent, 'agent') and hasattr(agent.agent, '_actor_critic'):
            try:
                params = torch.load(model_path, weights_only=False)
                if isinstance(params, dict) and 'pi' in params:
                    agent.agent._actor_critic.actor.load_state_dict(params['pi'])
                    loaded = True
                elif isinstance(params, dict):
                    try:
                        agent.agent._actor_critic.load_state_dict(params)
                        loaded = True
                    except:
                        pass
            except Exception as e:
                print(f"Manual load failed: {e}")
        
        if not loaded:
            print(f"Warning: Could not load model from {model_path}")
            
        return agent
    except Exception as e:
        print(f"Could not load OmniSafe model from {model_path}: {e}")
        return None


# ============================================================================
# Evaluation Core
# ============================================================================

def evaluate_agent(env, agent, num_episodes=100, deterministic=True, agent_type='omnisafe'):
    """Evaluate an agent (OmniSafe or random) and return per-episode data."""
    episodes_data = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        path = [tuple(info['agent_pos'])]
        
        max_steps = env.grid_size * 10
        for _ in range(max_steps):
            if agent_type == 'random':
                action_idx = env.action_space.sample()
            elif agent_type == 'omnisafe' and agent is not None:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                
                ac = None
                if hasattr(agent, 'agent') and hasattr(agent.agent, '_actor_critic'):
                    ac = agent.agent._actor_critic
                elif hasattr(agent, 'agent') and hasattr(agent.agent, 'actor_critic'):
                    ac = agent.agent.actor_critic
                
                if ac is not None:
                    ac_out = ac.step(obs_tensor, deterministic=deterministic)
                    action = ac_out[0]
                else:
                    action_idx = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action_idx)
                    episode_reward += reward
                    path.append(tuple(info['agent_pos']))
                    if terminated or truncated:
                        break
                    continue
                
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                
                val = action if isinstance(action, np.ndarray) else np.array(action)
                if val.size != 2:
                    if val.size == 1:
                        val = np.array([float(val), float(val)])
                    elif val.size > 2:
                        val = val.flatten()[:2]
                
                if val[0] >= 0 and val[1] >= 0:
                    action_idx = 0
                elif val[0] >= 0 and val[1] < 0:
                    action_idx = 1
                elif val[0] < 0 and val[1] >= 0:
                    action_idx = 2
                else:
                    action_idx = 3
            else:
                action_idx = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action_idx)
            episode_reward += reward
            path.append(tuple(info['agent_pos']))
            
            if terminated or truncated:
                break
        
        bomb_exploded = (env.bomb_timer_max is not None 
                        and info['bomb_time_left'] == 0 
                        and info['agent_pos'] != env.hostage_pos)
        
        hostage_saved = info['agent_pos'] == env.hostage_pos
        if hasattr(env, 'multi_hostage') and env.multi_hostage:
            hostages_saved_count = len(info.get('hostages_saved', []))
            total_hostages = len(env.hostage_positions)
            hostage_saved = hostages_saved_count > 0
        else:
            hostages_saved_count = 1 if hostage_saved else 0
            total_hostages = 1
        
        ep_data = {
            'reward': episode_reward,
            'civilians_trampled': len(info['civilians_trampled_this_episode']),
            'hostage_saved': hostage_saved,
            'hostages_saved_count': hostages_saved_count,
            'total_hostages': total_hostages,
            'bomb_exploded': bomb_exploded,
            'steps_taken': info['steps_taken'],
            'path': path
        }
        episodes_data.append(ep_data)
    
    return episodes_data


def aggregate_episodes(episodes_data, bomb_timer):
    """Aggregate episode data into summary statistics."""
    rewards = [ep['reward'] for ep in episodes_data]
    violations = [ep['civilians_trampled'] for ep in episodes_data]
    save_rates = [1.0 if ep['hostage_saved'] else 0.0 for ep in episodes_data]
    bomb_rates = [1.0 if ep['bomb_exploded'] else 0.0 for ep in episodes_data]
    
    # Safety performance scores
    safety_data = compute_batch_safety_performance(episodes_data, bomb_timer)
    
    return {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_civilian_violations': np.mean(violations),
        'std_civilian_violations': np.std(violations),
        'hostage_save_rate': np.mean(save_rates),
        'std_save_rate': np.std(save_rates),
        'bomb_explosion_rate': np.mean(bomb_rates),
        'std_bomb_rate': np.std(bomb_rates),
        'avg_safety_score': safety_data['mean_safety_score'],
        'std_safety_score': safety_data['std_safety_score'],
        'all_rewards': rewards,
        'all_violations': violations,
        'all_safety_scores': safety_data['all_scores'],
        'paths': [ep['path'] for ep in episodes_data[:5]]
    }


def aggregate_seeds(seed_results):
    """Aggregate results across multiple seeds (mean of means, STD of means)."""
    if not seed_results:
        return None
    
    metrics = ['avg_reward', 'avg_civilian_violations', 'hostage_save_rate', 
               'bomb_explosion_rate', 'avg_safety_score']
    
    aggregated = {}
    for metric in metrics:
        values = [r[metric] for r in seed_results if r is not None]
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_all'] = values
    
    # Collect all per-episode data across seeds for statistical tests
    all_rewards = []
    all_violations = []
    all_safety_scores = []
    for r in seed_results:
        if r is not None:
            all_rewards.extend(r.get('all_rewards', []))
            all_violations.extend(r.get('all_violations', []))
            all_safety_scores.extend(r.get('all_safety_scores', []))
    
    aggregated['all_rewards_flat'] = all_rewards
    aggregated['all_violations_flat'] = all_violations
    aggregated['all_safety_scores_flat'] = all_safety_scores
    
    # Keep paths from first seed
    aggregated['paths'] = seed_results[0].get('paths', []) if seed_results else []
    
    return aggregated


# ============================================================================
# Statistical Tests
# ============================================================================

def compute_significance(group_a, group_b, metric_name):
    """Compute Mann-Whitney U test between two groups."""
    if not group_a or not group_b:
        return None
    
    a = np.array(group_a)
    b = np.array(group_b)
    
    try:
        stat, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')
        
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'
        
        return {
            'metric': metric_name,
            'U_statistic': float(stat),
            'p_value': float(p_value),
            'significance': sig,
            'n_a': len(a),
            'n_b': len(b)
        }
    except Exception as e:
        return {'metric': metric_name, 'error': str(e)}


# ============================================================================
# Visualization
# ============================================================================

def visualize_paths(env, paths, title, save_path=None):
    """Visualize agent paths — scaled for any grid size."""
    if not paths or len(paths) == 0:
        return None
    
    gs = env.grid_size
    # Scale marker sizes based on grid
    base_marker = max(100, 500 * (5 / gs))
    fig_width = max(15, 3 * min(5, len(paths)))
    fig_height = max(3, 3 * (gs / 5))
    
    num_plots = min(5, len(paths))
    fig, axes = plt.subplots(1, num_plots, figsize=(fig_width, fig_height))
    if num_plots == 1:
        axes = [axes]
    
    for idx, (ax, path) in enumerate(zip(axes, paths[:5])):
        # Grid lines
        for i in range(gs + 1):
            ax.plot([0, gs], [i, i], 'k-', linewidth=0.5, alpha=0.3)
            ax.plot([i, i], [0, gs], 'k-', linewidth=0.5, alpha=0.3)
        
        # Walls
        if hasattr(env, 'walls'):
            for wall in env.walls:
                ax.add_patch(plt.Rectangle(
                    (wall[1], gs - wall[0] - 1), 1, 1, 
                    fill=True, facecolor='gray', alpha=0.5))
        
        # Civilians
        for civ in env.civilians:
            trampled = civ in [path[i] for i in range(len(path))]
            color = 'red' if trampled else 'orange'
            label = 'X' if trampled else 'C'
            ax.scatter(civ[1] + 0.5, gs - civ[0] - 0.5,
                      s=base_marker, c=color, marker='s', alpha=0.7, 
                      edgecolors='black', linewidth=1.5)
            ax.text(civ[1] + 0.5, gs - civ[0] - 0.5, label,
                   ha='center', va='center', fontsize=max(6, 10 * (5/gs)), fontweight='bold',
                   color='white' if trampled else 'black')
        
        # Hostage(s)
        for hp in env.hostage_positions:
            ax.scatter(hp[1] + 0.5, gs - hp[0] - 0.5,
                      s=base_marker, c='green', marker='s', alpha=0.7,
                      edgecolors='black', linewidth=1.5)
            ax.text(hp[1] + 0.5, gs - hp[0] - 0.5, 'H',
                   ha='center', va='center', fontsize=max(6, 10 * (5/gs)), fontweight='bold', color='white')
        
        # Path
        path_x = [p[1] + 0.5 for p in path]
        path_y = [gs - p[0] - 0.5 for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=1.5, alpha=0.7)
        ax.scatter(path_x[0], path_y[0], s=base_marker * 0.6, c='blue', marker='o',
                  edgecolors='black', linewidth=1.5, zorder=10)
        
        ax.set_xlim(0, gs)
        ax.set_ylim(0, gs)
        ax.set_aspect('equal')
        ax.set_title(f'Ep {idx+1}\nSteps: {len(path)-1}', fontsize=9)
        ax.axis('off')
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_summary_comparison(results_dict, save_path, grid_id):
    """Publication-quality comparison plots with error bars and significance."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    scenarios = ['Low Stakes', 'High Stakes']
    timer_labels = list(results_dict.get('timer_sweep_labels', ['Low', 'High']))
    
    # Colors
    ppo_color = '#e74c3c'
    const_color = '#3498db'
    
    # --- Panel 1: Civilian Violations ---
    ax = axes[0, 0]
    ppo_vals = [results_dict.get(f'ppo_low_stakes', {}).get('avg_civilian_violations_mean', 0),
                results_dict.get(f'ppo_high_stakes', {}).get('avg_civilian_violations_mean', 0)]
    ppo_err = [results_dict.get(f'ppo_low_stakes', {}).get('avg_civilian_violations_std', 0),
               results_dict.get(f'ppo_high_stakes', {}).get('avg_civilian_violations_std', 0)]
    const_vals = [results_dict.get(f'constrained_low_stakes', {}).get('avg_civilian_violations_mean', 0),
                  results_dict.get(f'constrained_high_stakes', {}).get('avg_civilian_violations_mean', 0)]
    const_err = [results_dict.get(f'constrained_low_stakes', {}).get('avg_civilian_violations_std', 0),
                 results_dict.get(f'constrained_high_stakes', {}).get('avg_civilian_violations_std', 0)]
    
    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width/2, ppo_vals, width, yerr=ppo_err, label='PPO', color=ppo_color, alpha=0.8, capsize=4)
    ax.bar(x + width/2, const_vals, width, yerr=const_err, label='Constrained', color=const_color, alpha=0.8, capsize=4)
    ax.set_ylabel('Avg Civilian Violations')
    ax.set_title('Civilian Violations (↓ Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add significance stars
    for i, scen in enumerate(['low_stakes', 'high_stakes']):
        sig = results_dict.get(f'sig_violations_{scen}', {})
        if sig and sig.get('significance', 'n.s.') != 'n.s.':
            max_val = max(ppo_vals[i] + ppo_err[i], const_vals[i] + const_err[i])
            ax.text(i, max_val + 0.1, sig['significance'], ha='center', fontsize=12, fontweight='bold')
    
    # --- Panel 2: Hostage Save Rate ---
    ax = axes[0, 1]
    ppo_saves = [results_dict.get(f'ppo_low_stakes', {}).get('hostage_save_rate_mean', 0) * 100,
                 results_dict.get(f'ppo_high_stakes', {}).get('hostage_save_rate_mean', 0) * 100]
    ppo_saves_err = [results_dict.get(f'ppo_low_stakes', {}).get('hostage_save_rate_std', 0) * 100,
                     results_dict.get(f'ppo_high_stakes', {}).get('hostage_save_rate_std', 0) * 100]
    const_saves = [results_dict.get(f'constrained_low_stakes', {}).get('hostage_save_rate_mean', 0) * 100,
                   results_dict.get(f'constrained_high_stakes', {}).get('hostage_save_rate_mean', 0) * 100]
    const_saves_err = [results_dict.get(f'constrained_low_stakes', {}).get('hostage_save_rate_std', 0) * 100,
                       results_dict.get(f'constrained_high_stakes', {}).get('hostage_save_rate_std', 0) * 100]
    
    ax.bar(x - width/2, ppo_saves, width, yerr=ppo_saves_err, label='PPO', color=ppo_color, alpha=0.8, capsize=4)
    ax.bar(x + width/2, const_saves, width, yerr=const_saves_err, label='Constrained', color=const_color, alpha=0.8, capsize=4)
    ax.set_ylabel('Hostage Save Rate (%)')
    ax.set_title('Hostage Save Rate (↑ Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.set_ylim([0, 110])
    ax.grid(axis='y', alpha=0.3)
    
    # --- Panel 3: Safety Performance Score ---
    ax = axes[0, 2]
    ppo_sp = [results_dict.get(f'ppo_low_stakes', {}).get('avg_safety_score_mean', 0),
              results_dict.get(f'ppo_high_stakes', {}).get('avg_safety_score_mean', 0)]
    ppo_sp_err = [results_dict.get(f'ppo_low_stakes', {}).get('avg_safety_score_std', 0),
                  results_dict.get(f'ppo_high_stakes', {}).get('avg_safety_score_std', 0)]
    const_sp = [results_dict.get(f'constrained_low_stakes', {}).get('avg_safety_score_mean', 0),
                results_dict.get(f'constrained_high_stakes', {}).get('avg_safety_score_mean', 0)]
    const_sp_err = [results_dict.get(f'constrained_low_stakes', {}).get('avg_safety_score_std', 0),
                    results_dict.get(f'constrained_high_stakes', {}).get('avg_safety_score_std', 0)]
    
    ax.bar(x - width/2, ppo_sp, width, yerr=ppo_sp_err, label='PPO', color=ppo_color, alpha=0.8, capsize=4)
    ax.bar(x + width/2, const_sp, width, yerr=const_sp_err, label='Constrained', color=const_color, alpha=0.8, capsize=4)
    ax.set_ylabel('Safety Performance Score')
    ax.set_title('Safety Performance (↑ Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for i, scen in enumerate(['low_stakes', 'high_stakes']):
        sig = results_dict.get(f'sig_safety_{scen}', {})
        if sig and sig.get('significance', 'n.s.') != 'n.s.':
            max_val = max(ppo_sp[i] + ppo_sp_err[i], const_sp[i] + const_sp_err[i])
            ax.text(i, max_val + 2, sig['significance'], ha='center', fontsize=12, fontweight='bold')
    
    # --- Panel 4: Bomb Explosion Rate ---
    ax = axes[1, 0]
    methods = ['PPO', 'Constrained']
    bomb_rates = [results_dict.get(f'ppo_high_stakes', {}).get('bomb_explosion_rate_mean', 0) * 100,
                  results_dict.get(f'constrained_high_stakes', {}).get('bomb_explosion_rate_mean', 0) * 100]
    bomb_err = [results_dict.get(f'ppo_high_stakes', {}).get('bomb_explosion_rate_std', 0) * 100,
                results_dict.get(f'constrained_high_stakes', {}).get('bomb_explosion_rate_std', 0) * 100]
    
    ax.bar(methods, bomb_rates, yerr=bomb_err, color=[ppo_color, const_color], alpha=0.8, capsize=4)
    ax.set_ylabel('Bomb Explosion Rate (%)')
    ax.set_title('Catastrophic Failure — High Stakes (↓ Better)')
    ax.grid(axis='y', alpha=0.3)
    
    # --- Panel 5: Reward ---
    ax = axes[1, 1]
    ppo_rew = [results_dict.get(f'ppo_low_stakes', {}).get('avg_reward_mean', 0),
               results_dict.get(f'ppo_high_stakes', {}).get('avg_reward_mean', 0)]
    ppo_rew_err = [results_dict.get(f'ppo_low_stakes', {}).get('avg_reward_std', 0),
                   results_dict.get(f'ppo_high_stakes', {}).get('avg_reward_std', 0)]
    const_rew = [results_dict.get(f'constrained_low_stakes', {}).get('avg_reward_mean', 0),
                 results_dict.get(f'constrained_high_stakes', {}).get('avg_reward_mean', 0)]
    const_rew_err = [results_dict.get(f'constrained_low_stakes', {}).get('avg_reward_std', 0),
                     results_dict.get(f'constrained_high_stakes', {}).get('avg_reward_std', 0)]
    
    ax.bar(x - width/2, ppo_rew, width, yerr=ppo_rew_err, label='PPO', color=ppo_color, alpha=0.8, capsize=4)
    ax.bar(x + width/2, const_rew, width, yerr=const_rew_err, label='Constrained', color=const_color, alpha=0.8, capsize=4)
    ax.set_ylabel('Average Reward')
    ax.set_title('Training Reward')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # --- Panel 6: Timer Sweep (if available) ---
    ax = axes[1, 2]
    timer_sweep = results_dict.get('timer_sweep', {})
    if timer_sweep:
        timers = sorted(timer_sweep.keys())
        ppo_sp_sweep = [timer_sweep[t].get('ppo_safety_mean', 0) for t in timers]
        ppo_sp_sweep_err = [timer_sweep[t].get('ppo_safety_std', 0) for t in timers]
        const_sp_sweep = [timer_sweep[t].get('constrained_safety_mean', 0) for t in timers]
        const_sp_sweep_err = [timer_sweep[t].get('constrained_safety_std', 0) for t in timers]
        
        timer_labels = [str(t) if t is not None else 'None' for t in timers]
        x_sweep = np.arange(len(timers))
        
        ax.errorbar(x_sweep, ppo_sp_sweep, yerr=ppo_sp_sweep_err, marker='o', 
                    color=ppo_color, label='PPO', capsize=4, linewidth=2)
        ax.errorbar(x_sweep, const_sp_sweep, yerr=const_sp_sweep_err, marker='s',
                    color=const_color, label='Constrained', capsize=4, linewidth=2)
        ax.set_xticks(x_sweep)
        ax.set_xticklabels(timer_labels)
        ax.set_xlabel('Bomb Timer')
        ax.set_ylabel('Safety Performance Score')
        ax.set_title('Safety vs. Urgency Level')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Timer sweep\nnot yet computed', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Safety vs. Urgency Level')
    
    fig.suptitle(f'Comprehensive Baseline Comparison — Grid {grid_id}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close(fig)


def plot_performance_vs_timesteps(training_data, save_path, grid_id):
    """
    DeepMind-style performance-vs-timesteps plot.
    Shows both reward and safety performance over training.
    
    training_data: dict with keys like 'ppo' and 'constrained', each containing:
        - 'epochs': list of epoch numbers
        - 'rewards': list of reward values per epoch
        - 'safety_scores': list of safety scores per epoch (if computed)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ppo_color = '#e74c3c'
    const_color = '#3498db'
    
    # --- Panel 1: Reward over training ---
    ax = axes[0]
    for agent_key, label, color in [('ppo', 'PPO (Utilitarian)', ppo_color), 
                                      ('constrained', 'Constrained (Deontological)', const_color)]:
        data = training_data.get(agent_key, {})
        if 'epochs' in data and 'rewards' in data:
            epochs = data['epochs']
            rewards = data['rewards']
            
            if 'rewards_seeds' in data:
                # Multi-seed: plot mean ± std
                means = np.mean(data['rewards_seeds'], axis=0)
                stds = np.std(data['rewards_seeds'], axis=0)
                ax.plot(epochs, means, color=color, label=label, linewidth=2)
                ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.2)
            else:
                ax.plot(epochs, rewards, color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Reward Function (What Agent Optimizes)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # --- Panel 2: Safety Performance over training ---
    ax = axes[1]
    for agent_key, label, color in [('ppo', 'PPO (Utilitarian)', ppo_color),
                                      ('constrained', 'Constrained (Deontological)', const_color)]:
        data = training_data.get(agent_key, {})
        if 'epochs' in data and 'safety_scores' in data:
            epochs = data['epochs']
            safety = data['safety_scores']
            
            if 'safety_seeds' in data:
                means = np.mean(data['safety_seeds'], axis=0)
                stds = np.std(data['safety_seeds'], axis=0)
                ax.plot(epochs, means, color=color, label=label, linewidth=2)
                ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.2)
            else:
                ax.plot(epochs, safety, color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Safety Performance Score')
    ax.set_title('Safety Performance (What We Actually Want)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Ideal')
    
    fig.suptitle(f'Performance vs. Training — Grid {grid_id} (DeepMind-style)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close(fig)


# ============================================================================
# Comparison Table
# ============================================================================

def create_comparison_table(results_dict, grid_id):
    """Print publication-quality comparison table."""
    print(f"\n{'='*100}")
    print(f"RESULTS COMPARISON TABLE — Grid {grid_id}")
    print(f"{'='*100}")
    
    header = (f"{'Scenario':<15} {'Method':<20} {'Reward':<18} {'Violations':<18} "
              f"{'Save %':<15} {'Bomb %':<15} {'Safety Score':<18}")
    print(header)
    print("-" * 100)
    
    for scenario in ['Low Stakes', 'High Stakes']:
        scenario_key = scenario.lower().replace(' ', '_')
        
        for method in ['PPO (Utilitarian)', 'Constrained (Deont.)']:
            method_key = 'ppo' if 'PPO' in method else 'constrained'
            key = f"{method_key}_{scenario_key}"
            
            r = results_dict.get(key, {})
            if r:
                rew = f"{r.get('avg_reward_mean', 0):.1f} ± {r.get('avg_reward_std', 0):.1f}"
                viol = f"{r.get('avg_civilian_violations_mean', 0):.2f} ± {r.get('avg_civilian_violations_std', 0):.2f}"
                save = f"{r.get('hostage_save_rate_mean', 0)*100:.1f}%"
                bomb = f"{r.get('bomb_explosion_rate_mean', 0)*100:.1f}%"
                safety = f"{r.get('avg_safety_score_mean', 0):.1f} ± {r.get('avg_safety_score_std', 0):.1f}"
                
                print(f"{scenario:<15} {method:<20} {rew:<18} {viol:<18} {save:<15} {bomb:<15} {safety:<18}")
        
        # Significance tests
        sig_viol = results_dict.get(f'sig_violations_{scenario_key}', {})
        sig_safety = results_dict.get(f'sig_safety_{scenario_key}', {})
        if sig_viol:
            print(f"{'':>35} Violations: p={sig_viol.get('p_value', 'N/A'):.4f} {sig_viol.get('significance', '')}")
        if sig_safety:
            print(f"{'':>35} Safety:     p={sig_safety.get('p_value', 'N/A'):.4f} {sig_safety.get('significance', '')}")
        print()
    
    print("=" * 100)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation with multi-seed and statistical tests")
    parser.add_argument("--grid", type=int, default=1, help="Grid layout ID (1-6)")
    parser.add_argument("--models_dir", type=str, default="results/models", 
                       help="Directory containing trained models")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (default: CDDP_RL/results/grid_N/)")
    parser.add_argument("--num_episodes", type=int, default=100, help="Episodes per evaluation")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds to aggregate")
    parser.add_argument("--stochastic", action="store_true", help="Also run stochastic evaluation")
    args = parser.parse_args()
    
    # Default output directory
    if args.output_dir is None:
        base_results = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        args.output_dir = os.path.join(base_results, f'grid_{args.grid}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION — Grid {args.grid}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}")
    
    grid_layout = HostageGridworld.GRID_LAYOUTS[args.grid]
    optimal_timer = grid_layout['bomb_timer']
    
    results_dict = {}
    
    # Evaluate each scenario × agent × seed
    for scenario, timer, scenario_key in [
        ('Low Stakes', None, 'low_stakes'),
        ('High Stakes', optimal_timer, 'high_stakes')
    ]:
        for agent_type, agent_key in [('ppo', 'ppo'), ('constrained', 'constrained')]:
            hard_constraint = (agent_key == 'constrained')
            
            print(f"\n[*] Evaluating {agent_key.upper()} on {scenario} (timer={timer})...")
            
            seed_results = []
            for seed_idx in range(args.num_seeds):
                # Try to find model — check multiple naming conventions
                candidates = [
                    os.path.join(args.models_dir, f"{agent_key}_grid{args.grid}.pt"),
                    os.path.join(args.models_dir, f"{agent_key}_grid{args.grid}_seed{seed_idx}.pt"),
                    os.path.join(args.models_dir, f"{agent_key}_high_stakes_grid{args.grid}.pt"),
                    os.path.join('results', f"{agent_key}_high_stakes_grid{args.grid}.pt"),
                    os.path.join('results', f"ppo_high_stakes_grid{args.grid}.pt") if agent_key == 'ppo' else '',
                    os.path.join('results', f"constrained_high_stakes_grid{args.grid}.pt") if agent_key == 'constrained' else '',
                ]
                model_path = None
                for c in candidates:
                    if c and os.path.exists(c):
                        model_path = c
                        break
                if model_path is None:
                    model_path = candidates[0]  # Will fail gracefully
                
                env = HostageGridworld(
                    layout_id=args.grid, 
                    bomb_timer=timer,
                    hard_constraint=hard_constraint
                )
                
                # Use the correct env_id for loading the agent
                # Must match what was used during training
                if agent_key == 'constrained':
                    env_id = f'HostageGridworld-Constrained-Layout{args.grid}-v0'
                else:
                    env_id = f'HostageGridworld-Layout{args.grid}-v0'
                agent = load_omnisafe_agent(model_path, env_id) if os.path.exists(model_path) else None
                agent_eval_type = 'omnisafe' if agent else 'random'
                
                if agent is None:
                    print(f"  Seed {seed_idx}: Using random baseline (model not found: {model_path})")
                
                # Deterministic evaluation
                episodes = evaluate_agent(env, agent, args.num_episodes, 
                                         deterministic=True, agent_type=agent_eval_type)
                result = aggregate_episodes(episodes, timer)
                seed_results.append(result)
                
                # Stochastic evaluation
                if args.stochastic and agent is not None:
                    stoch_episodes = evaluate_agent(env, agent, args.num_episodes,
                                                   deterministic=False, agent_type=agent_eval_type)
                    stoch_result = aggregate_episodes(stoch_episodes, timer)
                    # Store under separate key
                    stoch_key = f"{agent_key}_{scenario_key}_stochastic"
                    if stoch_key not in results_dict:
                        results_dict[stoch_key] = []
                    results_dict[stoch_key] = stoch_result
            
            # Aggregate across seeds
            key = f"{agent_key}_{scenario_key}"
            results_dict[key] = aggregate_seeds(seed_results)
    
    # Statistical tests between PPO and Constrained
    for scenario_key in ['low_stakes', 'high_stakes']:
        ppo_data = results_dict.get(f'ppo_{scenario_key}', {})
        const_data = results_dict.get(f'constrained_{scenario_key}', {})
        
        if ppo_data and const_data:
            # Violations
            sig_viol = compute_significance(
                ppo_data.get('all_violations_flat', []),
                const_data.get('all_violations_flat', []),
                f'violations_{scenario_key}'
            )
            results_dict[f'sig_violations_{scenario_key}'] = sig_viol
            
            # Safety scores
            sig_safety = compute_significance(
                ppo_data.get('all_safety_scores_flat', []),
                const_data.get('all_safety_scores_flat', []),
                f'safety_{scenario_key}'
            )
            results_dict[f'sig_safety_{scenario_key}'] = sig_safety
    
    # Print results
    create_comparison_table(results_dict, args.grid)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    for scenario_key, timer in [('low_stakes', None), ('high_stakes', optimal_timer)]:
        for agent_key in ['ppo', 'constrained']:
            key = f"{agent_key}_{scenario_key}"
            data = results_dict.get(key, {})
            paths = data.get('paths', [])
            if paths:
                hard_constraint = (agent_key == 'constrained')
                env = HostageGridworld(layout_id=args.grid, bomb_timer=timer, 
                                      hard_constraint=hard_constraint)
                env.reset()
                
                agent_label = 'PPO (Utilitarian)' if agent_key == 'ppo' else 'Constrained (Deontological)'
                stakes_label = 'Low Stakes' if timer is None else f'High Stakes (t={timer})'
                title = f'{agent_label} — {stakes_label} — Grid {args.grid}'
                
                viz_path = os.path.join(args.output_dir, f'viz_{agent_key}_{scenario_key}_grid{args.grid}.png')
                visualize_paths(env, paths, title, viz_path)
    
    # Summary comparison plot
    summary_path = os.path.join(args.output_dir, f'summary_comparison_grid{args.grid}.png')
    plot_summary_comparison(results_dict, summary_path, args.grid)
    
    # Save raw results as JSON
    json_results = {}
    for k, v in results_dict.items():
        if isinstance(v, dict):
            json_results[k] = {}
            for kk, vv in v.items():
                if isinstance(vv, (list, np.ndarray)):
                    if len(vv) > 20:
                        continue  # Skip large arrays
                json_results[k][kk] = vv if not isinstance(vv, np.ndarray) else vv.tolist()
        elif isinstance(v, (int, float, str)):
            json_results[k] = v
    
    json_path = os.path.join(args.output_dir, f'results_grid{args.grid}.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"✓ Saved: {json_path}")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
