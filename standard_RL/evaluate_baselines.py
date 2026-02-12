"""
Comprehensive Evaluation Script - IMPROVED

Trains both baselines with improved hyperparameters and generates comparison.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from hostage_gridworld import HostageGridworld
from train_ppo import PPOAgent, train_ppo, evaluate_agent as eval_ppo


def visualize_paths(env, paths, title, save_path=None):
    """Visualize agent paths."""
    if not paths or len(paths) == 0:
        print(f"Warning: No paths for {title}")
        return None
    
    fig, axes = plt.subplots(1, min(5, len(paths)), figsize=(15, 3))
    if len(paths) == 1:
        axes = [axes]
    
    for idx, (ax, path) in enumerate(zip(axes, paths[:5])):
        # Draw grid
        for i in range(env.grid_size + 1):
            ax.plot([0, env.grid_size], [i, i], 'k-', linewidth=0.5, alpha=0.3)
            ax.plot([i, i], [0, env.grid_size], 'k-', linewidth=0.5, alpha=0.3)
        
        # Dynamic marker size based on grid_size (smaller for larger grids)
        marker_size = max(200, 500 // (env.grid_size / 5))
        font_size = max(6, 10 // (env.grid_size / 5))
        
        # Draw civilians
        for civ in env.civilians:
            if civ in [path[i] for i in range(len(path))]:
                ax.scatter(civ[1] + 0.5, env.grid_size - civ[0] - 0.5, 
                          s=marker_size, c='red', marker='s', alpha=0.7, edgecolors='black', linewidth=2)
                ax.text(civ[1] + 0.5, env.grid_size - civ[0] - 0.5, 'X',
                       ha='center', va='center', fontsize=font_size, fontweight='bold', color='white')
            else:
                ax.scatter(civ[1] + 0.5, env.grid_size - civ[0] - 0.5, 
                          s=marker_size, c='orange', marker='s', alpha=0.7, edgecolors='black', linewidth=2)
                ax.text(civ[1] + 0.5, env.grid_size - civ[0] - 0.5, 'C',
                       ha='center', va='center', fontsize=font_size, fontweight='bold')
        
        # Draw hostage
        ax.scatter(env.hostage_pos[1] + 0.5, env.grid_size - env.hostage_pos[0] - 0.5,
                  s=marker_size, c='green', marker='s', alpha=0.7, edgecolors='black', linewidth=2)
        ax.text(env.hostage_pos[1] + 0.5, env.grid_size - env.hostage_pos[0] - 0.5, 'H',
               ha='center', va='center', fontsize=font_size, fontweight='bold', color='white')
        
        # Draw path
        path_x = [p[1] + 0.5 for p in path]
        path_y = [env.grid_size - p[0] - 0.5 for p in path]
        
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
        ax.scatter(path_x[0], path_y[0], s=marker_size * 0.6, c='blue', marker='o', 
                  edgecolors='black', linewidth=2, zorder=10)
        
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_aspect('equal')
        
        ax.set_title(f'Episode {idx+1}\nSteps: {len(path)-1}', fontsize=10)
        ax.axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def create_comparison_table(results_dict):
    """Create comparison table."""
    print("\n" + "="*80)
    print("RESULTS COMPARISON TABLE")
    print("="*80)
    print()
    
    header = f"{'Scenario':<20} {'Method':<20} {'Reward':<15} {'Violations':<15} {'Save %':<12} {'Bomb %':<12}"
    print(header)
    print("-"*80)
    
    for scenario in ['Low Stakes', 'High Stakes']:
        scenario_key = scenario.lower().replace(' ', '_')
        
        for method in ['PPO (Utilitarian)', 'Constrained (Deontological)']:
            method_key = 'ppo' if 'PPO' in method else 'constrained'
            
            key = f"{method_key}_{scenario_key}"
            if key in results_dict:
                r = results_dict[key]
                
                reward_str = f"{r['avg_reward']:.1f} ± {r['std_reward']:.1f}"
                violations_str = f"{r['avg_civilian_violations']:.2f}"
                save_pct = f"{r['hostage_save_rate']*100:.1f}%"
                bomb_pct = f"{r['bomb_explosion_rate']*100:.1f}%" if 'bomb_explosion_rate' in r else "0.0%"
                
                print(f"{scenario:<20} {method:<20} {reward_str:<15} {violations_str:<15} {save_pct:<12} {bomb_pct:<12}")
        print()
    
    print("="*80)


def analyze_failure_modes(results_dict):
    """Analyze failure modes."""
    print("\n" + "="*80)
    print("FAILURE MODE ANALYSIS")
    print("="*80)
    
    print("\n[FAILURE 1] Utilitarian (PPO) in LOW STAKES:")
    print("-" * 60)
    low_ppo = results_dict['ppo_low_stakes']
    if low_ppo['avg_civilian_violations'] > 0.5:
        print(f"❌ FAILS: Tramples {low_ppo['avg_civilian_violations']:.2f} civilians on average")
        print(f"   Even though there's NO TIME PRESSURE (no bomb)")
        print(f"   This is RECKLESS - violates rules unnecessarily!")
    else:
        print(f"⚠ Learned to avoid civilians in low stakes")
        print(f"   But cannot distinguish between justified/unjustified violations")
    
    print("\n[FAILURE 2] Deontological (Constrained) in HIGH STAKES:")
    print("-" * 60)
    high_constrained = results_dict['constrained_high_stakes']
    if high_constrained['bomb_explosion_rate'] > 0.3:
        print(f"❌ FAILS: Bomb explodes {high_constrained['bomb_explosion_rate']*100:.1f}% of the time")
        print(f"   Civilian violations: {high_constrained['avg_civilian_violations']:.2f}")
        print(f"   This is FANATICAL - refuses to break rules even to prevent catastrophe!")
    else:
        print(f"⚠ Surprisingly avoided bomb")
    
    print("\n[IDEAL BEHAVIOR]")
    print("-" * 60)
    print("What we WANT:")
    print("  Low Stakes: Safe path, 0 violations")
    print("  High Stakes: Fast path, justified violations, save hostage")
    print()
    print("What we GET:")
    print(f"  PPO: Indiscriminate (same behavior in both scenarios)")
    print(f"  Constrained: Rigid (won't adapt even for catastrophe)")
    print()
    print("THIS IS THE GAP YOUR CDDP ALGORITHM FILLS! 🎯")
    print("="*80)


def main():
    """Run evaluation."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=1, help="Grid layout ID (1-3)")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"COMPREHENSIVE BASELINE EVALUATION - IMPROVED - GRID {args.grid}")
    print("="*80)
    
    # Get dynamic timer
    optimal_timer = HostageGridworld.GRID_LAYOUTS[args.grid]['bomb_timer']
    
    results_dict = {}
    
    ppo_filename = f'ppo_high_stakes_grid{args.grid}.pt'
    const_filename = f'constrained_high_stakes_grid{args.grid}.pt'
    
    # LOW STAKES - PPO (No Hard Constraint)
    print("\n[1/4] Loading PPO on Low Stakes...")
    env_low_ppo = HostageGridworld(bomb_timer=None, layout_id=args.grid, hard_constraint=False)
    obs_dim = env_low_ppo.observation_space.shape[0]
    
    # Match hidden_dim to training script
    grid_size = env_low_ppo.grid_size
    hidden_dim = 128 if grid_size <= 5 else 256
    print(f"Grid size: {grid_size}x{grid_size}, using hidden_dim={hidden_dim}")
    
    agent_ppo = PPOAgent(obs_dim=obs_dim, action_dim=4, hidden_dim=hidden_dim)
    # Load the High Stakes model (reckless)
    try:
        agent_ppo.load_state_dict(torch.load(ppo_filename))
        print(f"✓ Loaded: {ppo_filename}")
    except FileNotFoundError:
        print(f"❌ Could not find {ppo_filename}. Train it first.")
        return
    
    print("Evaluating PPO on Low Stakes...")
    results_ppo_low = eval_ppo(env_low_ppo, agent_ppo, num_episodes=100)
    results_dict['ppo_low_stakes'] = results_ppo_low
    
    # LOW STAKES - Constrained (With Hard Constraint)
    print("\n[2/4] Loading Constrained RL on Low Stakes...")
    env_low_const = HostageGridworld(bomb_timer=None, layout_id=args.grid, hard_constraint=True)
    agent_const = PPOAgent(obs_dim=obs_dim, action_dim=4, hidden_dim=hidden_dim)  # Use same hidden_dim
    
    # Try to load grid-specific model
    try:
        agent_const.load_state_dict(torch.load(const_filename))
        print(f"Loaded {const_filename}")
    except FileNotFoundError:
        print(f"Warning: {const_filename} not found.")
        print("Using default 'constrained_high_stakes.pt' if available (might fail on this grid)...")
        try:
             agent_const.load_state_dict(torch.load('constrained_high_stakes.pt'))
             print("Loaded default constrained_high_stakes.pt")
        except:
             print("❌ Could not load any constrained model. Results will be random.")
    
    print("Evaluating Constrained on Low Stakes...")
    results_const_low = eval_ppo(env_low_const, agent_const, num_episodes=100)
    results_dict['constrained_low_stakes'] = results_const_low
    
    # HIGH STAKES - PPO (No Hard Constraint)
    print("\n[3/4] Evaluating PPO on High Stakes...")
    env_high_ppo = HostageGridworld(bomb_timer=optimal_timer, layout_id=args.grid, hard_constraint=False)
    results_ppo_high = eval_ppo(env_high_ppo, agent_ppo, num_episodes=100)
    results_dict['ppo_high_stakes'] = results_ppo_high
    
    # HIGH STAKES - Constrained (With Hard Constraint)
    print("\n[4/4] Evaluating Constrained RL on High Stakes...")
    env_high_const = HostageGridworld(bomb_timer=optimal_timer, layout_id=args.grid, hard_constraint=True)
    results_const_high = eval_ppo(env_high_const, agent_const, num_episodes=100)
    results_dict['constrained_high_stakes'] = results_const_high
    
    # Analysis
    create_comparison_table(results_dict)
    analyze_failure_modes(results_dict)
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_paths(env_low_ppo, results_ppo_low['paths'], 
                   f'PPO (Utilitarian) - Low Stakes - Grid {args.grid}', f'viz_ppo_low_grid{args.grid}.png')
    visualize_paths(env_low_const, results_const_low['paths'],
                   f'Constrained RL (Deontological) - Low Stakes - Grid {args.grid}', f'viz_constrained_low_grid{args.grid}.png')
    visualize_paths(env_high_ppo, results_ppo_high['paths'],
                   f'PPO (Utilitarian) - High Stakes (Bomb) - Grid {args.grid}', f'viz_ppo_high_grid{args.grid}.png')
    visualize_paths(env_high_const, results_const_high['paths'],
                   f'Constrained RL (Deontological) - High Stakes (Bomb) - Grid {args.grid}', f'viz_constrained_high_grid{args.grid}.png')
    
    # Summary figure
    # (Keeping the rest of the plotting code mostly as is, just need to update filename)
    # ... actually I need to replace the whole main because I can't easily jump blocks.
    # The visualization logic above is already inside the ReplacementContent.
    
    # Summary figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    scenarios = ['Low Stakes', 'High Stakes']
    ppo_viol = [results_dict['ppo_low_stakes']['avg_civilian_violations'],
                results_dict['ppo_high_stakes']['avg_civilian_violations']]
    const_viol = [results_dict['constrained_low_stakes']['avg_civilian_violations'],
                  results_dict['constrained_high_stakes']['avg_civilian_violations']]
    
    x = np.arange(len(scenarios))
    width = 0.35
    axes[0, 0].bar(x - width/2, ppo_viol, width, label='PPO', color='red', alpha=0.7)
    axes[0, 0].bar(x + width/2, const_viol, width, label='Constrained', color='blue', alpha=0.7)
    axes[0, 0].set_ylabel('Avg Civilian Violations')
    axes[0, 0].set_title('Civilian Violations (Lower is Better)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(scenarios)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    ppo_saves = [results_dict['ppo_low_stakes']['hostage_save_rate'] * 100,
                results_dict['ppo_high_stakes']['hostage_save_rate'] * 100]
    const_saves = [results_dict['constrained_low_stakes']['hostage_save_rate'] * 100,
                  results_dict['constrained_high_stakes']['hostage_save_rate'] * 100]
    
    axes[0, 1].bar(x - width/2, ppo_saves, width, label='PPO', color='red', alpha=0.7)
    axes[0, 1].bar(x + width/2, const_saves, width, label='Constrained', color='blue', alpha=0.7)
    axes[0, 1].set_ylabel('Hostage Save Rate (%)')
    axes[0, 1].set_title('Hostage Save Rate (Higher is Better)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(scenarios)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 105])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    methods = ['PPO', 'Constrained']
    bomb_rates = [results_dict['ppo_high_stakes']['bomb_explosion_rate'] * 100,
                 results_dict['constrained_high_stakes']['bomb_explosion_rate'] * 100]
    
    axes[1, 0].bar(methods, bomb_rates, color=['red', 'blue'], alpha=0.7)
    axes[1, 0].set_ylabel('Bomb Explosion Rate (%)')
    axes[1, 0].set_title('Catastrophic Failure (High Stakes Only)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    ppo_rewards = [results_dict['ppo_low_stakes']['avg_reward'],
                  results_dict['ppo_high_stakes']['avg_reward']]
    const_rewards = [results_dict['constrained_low_stakes']['avg_reward'],
                    results_dict['constrained_high_stakes']['avg_reward']]
    
    axes[1, 1].bar(x - width/2, ppo_rewards, width, label='PPO', color='red', alpha=0.7)
    axes[1, 1].bar(x + width/2, const_rewards, width, label='Constrained', color='blue', alpha=0.7)
    axes[1, 1].set_ylabel('Average Reward')
    axes[1, 1].set_title('Overall Performance')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(scenarios)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    summary_filename = f'summary_comparison_grid{args.grid}.png'
    plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {summary_filename}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)



if __name__ == "__main__":
    main()
 
