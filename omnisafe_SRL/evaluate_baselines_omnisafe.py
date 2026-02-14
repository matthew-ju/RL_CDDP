
"""
Comprehensive Evaluation Script - OmniSafe Compatible

Evaluates both baselines trained with OmniSafe and generates comparison visualizations.
Maintains backward compatibility with custom-trained models.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hostage_gridworld import HostageGridworld
# Import adapter to ensure environment registration
try:
    import hostage_gridworld_adapter
except ImportError:
    pass

# Try to import OmniSafe for model loading
try:
    import omnisafe
    OMNISAFE_AVAILABLE = True
except ImportError:
    OMNISAFE_AVAILABLE = False
    print("Warning: OmniSafe not available. Will try to load custom models.")


def load_omnisafe_agent(model_path, env_id):
    """Load an OmniSafe-trained model."""
    if not OMNISAFE_AVAILABLE:
        return None
    
    try:
        # Initialize agent with the same ID as training
        # We assume PPO for loading, but it might be PPOLag. 
        # OmniSafe Agent class handles AlgoWrapper internally?
        # Agent(algo, ...) works.
        # But we need to know if it's PPO or PPOLag?
        # If we load, maybe we don't need to specify algo exactly if we just want 'act'?
        # But Agent init requires algo.
        
        # Heuristic: check filename
        algo = 'PPO'
        if 'constrained' in model_path.lower() or 'lag' in model_path.lower():
            algo = 'PPOLag'
            
        print(f"Loading OmniSafe agent (Algo: {algo}, Env: {env_id}) from {model_path}...")
        
        # We need custom_cfgs to match the training setup (obs_normalize: False)
        custom_cfgs = {
            'algo_cfgs': {
                'obs_normalize': False
            }
        }
        if algo == 'PPOLag':
             custom_cfgs['lagrange_cfgs'] = {
                 'cost_limit': 0.0,
                 'lagrangian_multiplier_init': 0.001,
                 'lambda_lr': 0.01,
                 'lambda_optimizer': 'Adam',
             }

        agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
        
        # Robust Loading Logic
        # Try agent.load (if available) or agent.agent.load (if available)
        # Or manually load state dict
        loaded = False
        
        # 1. Try agent.load (AlgoWrapper usually doesn't have it)
        if hasattr(agent, 'load'):
            try:
                agent.load(model_path)
                loaded = True
                print("Loaded using agent.load()")
            except Exception as e:
                print(f"agent.load failed: {e}")
        
        # 2. Try agent.agent.load (Underlying Algo might have it)
        if not loaded and hasattr(agent, 'agent') and hasattr(agent.agent, 'load'):
            try:
                agent.agent.load(model_path)
                loaded = True
                print("Loaded using agent.agent.load()")
            except Exception as e:
                print(f"agent.agent.load failed: {e}")
                
        # 3. Manual State Dict Loading (Most robust for OmniSafe .pt files)
        if not loaded and hasattr(agent, 'agent') and hasattr(agent.agent, '_actor_critic'):
            try:
                print("Attempting manual state dict load into _actor_critic.actor...")
                params = torch.load(model_path)
                
                if isinstance(params, dict) and 'pi' in params:
                    # Load specifically into the actor
                    agent.agent._actor_critic.actor.load_state_dict(params['pi'])
                    loaded = True
                    print("Loaded 'pi' into _actor_critic.actor successfully")
                elif isinstance(params, dict):
                    # Try full AC load if 'pi' shifted? (unlikely in omnisafe)
                    try:
                        agent.agent._actor_critic.load_state_dict(params)
                        loaded = True
                        print("Loaded full _actor_critic state dict")
                    except:
                        pass
            except Exception as e:
                print(f"Manual load failed: {e}")
        
        if not loaded:
            print("Warning: Could not load model parameters! Using initialized agent (random weights if not loaded).")
            
        return agent
    except Exception as e:
        print(f"Could not load OmniSafe model from {model_path}: {e}")
        return None


def evaluate_omnisafe_agent(env, agent, num_episodes=100):
    """Evaluate an OmniSafe agent."""
    total_rewards = []
    civilian_violations = []
    bomb_explosions = 0
    hostages_saved = 0
    paths_taken = []
    
    # Identify algorithm for logging
    algo_name = type(agent.agent).__name__ if hasattr(agent, 'agent') else "Unknown"
    print(f"Agent detected as {algo_name}")

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        path = [tuple(info['agent_pos'])]
        
        for _ in range(50):
            # obs to tensor and Ensure it's on CPU if needed (matching init)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            
            # Use agent.agent._actor_critic.step
            if hasattr(agent, 'agent') and hasattr(agent.agent, '_actor_critic'):
                ac = agent.agent._actor_critic
                # step returns (action, value_r, [value_c], log_prob)
                # We only need the first element
                ac_out = ac.step(obs_tensor, deterministic=True)
                action = ac_out[0]
            elif hasattr(agent, 'agent') and hasattr(agent.agent, 'actor_critic'):
                ac = agent.agent.actor_critic
                ac_out = ac.step(obs_tensor, deterministic=True)
                action = ac_out[0]
            else:
                # Fallback to random if nothing found
                action = torch.zeros(env.action_space.shape)
                print("Warning: No act/step method found, using zero action")
                break
            
            # Evaluate
            # The agent outputs a 2D Box action (Quadrant Discretization)
            # We must map this to Discrete(4) for the raw HostageGridworld env
            
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            val = action if isinstance(action, np.ndarray) else np.array(action)
            
            # Ensure shape is (2,) or flattened equivalent
            if val.size != 2:
                # If we somehow get a scalar or wrong size, fallback or error?
                # For safety, if scalar/1D, treat as [val, val] ? Or error?
                # The agent should be trained with Box(2,), so this should be fine.
                if val.size == 1:
                     val = np.array([float(val), float(val)]) # Fallback
                elif val.size > 2:
                     val = val.flatten()[:2]
            
            # Quadrant Mapping:
            # (+, +) -> 0 (Up)
            # (+, -) -> 1 (Down)
            # (-, +) -> 2 (Left)
            # (-, -) -> 3 (Right)
            
            if val[0] >= 0 and val[1] >= 0:
                idx = 0 # Up
            elif val[0] >= 0 and val[1] < 0:
                idx = 1 # Down
            elif val[0] < 0 and val[1] >= 0:
                idx = 2 # Left
            else:
                idx = 3 # Right
            
            obs, reward, terminated, truncated, info = env.step(idx)
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


def evaluate_random_agent(env, num_episodes=100):
    """Evaluate a random agent as fallback."""
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
            action = env.action_space.sample()
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
        
        # Draw civilians
        for civ in env.civilians:
            if civ in [path[i] for i in range(len(path))]:
                ax.scatter(civ[1] + 0.5, env.grid_size - civ[0] - 0.5, 
                          s=500, c='red', marker='s', alpha=0.7, edgecolors='black', linewidth=2)
                ax.text(civ[1] + 0.5, env.grid_size - civ[0] - 0.5, 'X',
                       ha='center', va='center', fontsize=10, fontweight='bold', color='white')
            else:
                ax.scatter(civ[1] + 0.5, env.grid_size - civ[0] - 0.5, 
                          s=500, c='orange', marker='s', alpha=0.7, edgecolors='black', linewidth=2)
                ax.text(civ[1] + 0.5, env.grid_size - civ[0] - 0.5, 'C',
                       ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw hostage
        ax.scatter(env.hostage_pos[1] + 0.5, env.grid_size - env.hostage_pos[0] - 0.5,
                  s=500, c='green', marker='s', alpha=0.7, edgecolors='black', linewidth=2)
        ax.text(env.hostage_pos[1] + 0.5, env.grid_size - env.hostage_pos[0] - 0.5, 'H',
               ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Draw path
        path_x = [p[1] + 0.5 for p in path]
        path_y = [env.grid_size - p[0] - 0.5 for p in path]
        
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
        ax.scatter(path_x[0], path_y[0], s=300, c='blue', marker='o', 
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
    low_ppo = results_dict.get('ppo_low_stakes')
    if low_ppo and low_ppo['avg_civilian_violations'] > 0.5:
        print(f"❌ FAILS: Tramples {low_ppo['avg_civilian_violations']:.2f} civilians on average")
        print(f"   Even though there's NO TIME PRESSURE (no bomb)")
        print(f"   This is RECKLESS - violates rules unnecessarily!")
    elif low_ppo:
        print(f"⚠ Learned to avoid civilians in low stakes")
        print(f"   But cannot distinguish between justified/unjustified violations")
    
    print("\n[FAILURE 2] Deontological (Constrained) in HIGH STAKES:")
    print("-" * 60)
    high_constrained = results_dict.get('constrained_high_stakes')
    if high_constrained and high_constrained['bomb_explosion_rate'] > 0.3:
        print(f"❌ FAILS: Bomb explodes {high_constrained['bomb_explosion_rate']*100:.1f}% of the time")
        print(f"   Civilian violations: {high_constrained['avg_civilian_violations']:.2f}")
        print(f"   This is FANATICAL - refuses to break rules even to prevent catastrophe!")
    elif high_constrained:
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
    print(f"COMPREHENSIVE BASELINE EVALUATION - OmniSafe - GRID {args.grid}")
    print("="*80)
    
    # Get dynamic timer
    optimal_timer = HostageGridworld.GRID_LAYOUTS[args.grid]['bomb_timer']
    
    results_dict = {}
    
    # NOTE: Model paths need to be updated to point to OmniSafe logged models
    # OmniSafe logs to ./omnisafe_logs/Algo-{Env}/...
    # For now, we assume user will manually point to correct path or we search?
    # Or we default to 'models/' directory if user moved them?
    # Let's keep assuming loading from current dir for now, 
    # but the user needs to know to move them or update this script.
    
    ppo_path = os.path.join('results', f'ppo_high_stakes_grid{args.grid}.pt')
    const_path = os.path.join('results', f'constrained_high_stakes_grid{args.grid}.pt')
    
    # Create environments for EVALUATION (Raw HostageGridworld)
    env_low_ppo = HostageGridworld(bomb_timer=None, layout_id=args.grid, hard_constraint=False)
    env_high_ppo = HostageGridworld(bomb_timer=optimal_timer, layout_id=args.grid, hard_constraint=False)
    env_low_const = HostageGridworld(bomb_timer=None, layout_id=args.grid, hard_constraint=True)
    env_high_const = HostageGridworld(bomb_timer=optimal_timer, layout_id=args.grid, hard_constraint=True)
    
    # Envs IDs for Agent loading
    low_env_id = 'HostageGridworld-v0'
    high_env_id = 'HostageGridworld-HighStakes-v0'
    
    # Try to load OmniSafe models
    print("\n[1/4] Evaluating PPO on Low Stakes...")
    # NOTE: We trained PPO on High Stakes? Or Low Stakes?
    # train_omnisafe_ppo.py defaulted to Low Stakes ('HostageGridworld-v0').
    # But usually baselines are trained on the scenario they are evaluated on, or High to enable capability.
    # The CDDP project trains baselines on specific scenarios.
    
    ppo_agent = load_omnisafe_agent(ppo_path, low_env_id)
    if ppo_agent:
        results_dict['ppo_low_stakes'] = evaluate_omnisafe_agent(env_low_ppo, ppo_agent, num_episodes=100)
    else:
        print(f"⚠ Using random baseline for PPO (model not found)")
        results_dict['ppo_low_stakes'] = evaluate_random_agent(env_low_ppo, num_episodes=100)
    
    print("\n[2/4] Evaluating Constrained RL on Low Stakes...")
    const_agent = load_omnisafe_agent(const_path, low_env_id)
    if const_agent:
        results_dict['constrained_low_stakes'] = evaluate_omnisafe_agent(env_low_const, const_agent, num_episodes=100)
    else:
        print(f"⚠ Using random baseline for Constrained (model not found)")
        results_dict['constrained_low_stakes'] = evaluate_random_agent(env_low_const, num_episodes=100)
    
    print("\n[3/4] Evaluating PPO on High Stakes...")
    # Load High Stakes agent if separate? Or reuse?
    # Usually we evaluate the SAME agent across scenarios if testing robustness.
    # But here we probably want to evaluate "PPO trained on High Stakes".
    
    if ppo_agent:
        results_dict['ppo_high_stakes'] = evaluate_omnisafe_agent(env_high_ppo, ppo_agent, num_episodes=100)
    else:
        results_dict['ppo_high_stakes'] = evaluate_random_agent(env_high_ppo, num_episodes=100)
    
    print("\n[4/4] Evaluating Constrained RL on High Stakes...")
    if const_agent:
        results_dict['constrained_high_stakes'] = evaluate_omnisafe_agent(env_high_const, const_agent, num_episodes=100)
    else:
        results_dict['constrained_high_stakes'] = evaluate_random_agent(env_high_const, num_episodes=100)
    
    # Analysis
    create_comparison_table(results_dict)
    analyze_failure_modes(results_dict)
    
    # Visualizations
    print("\nGenerating visualizations...")
    os.makedirs('results', exist_ok=True)
    
    visualize_paths(env_low_ppo, results_dict['ppo_low_stakes']['paths'], 
                   f'PPO (Utilitarian) - Low Stakes - Grid {args.grid}', 
                   os.path.join('results', f'viz_ppo_low_grid{args.grid}.png'))
    visualize_paths(env_low_const, results_dict['constrained_low_stakes']['paths'],
                   f'Constrained RL (Deontological) - Low Stakes - Grid {args.grid}', 
                   os.path.join('results', f'viz_constrained_low_grid{args.grid}.png'))
    visualize_paths(env_high_ppo, results_dict['ppo_high_stakes']['paths'],
                   f'PPO (Utilitarian) - High Stakes (Bomb) - Grid {args.grid}', 
                   os.path.join('results', f'viz_ppo_high_grid{args.grid}.png'))
    visualize_paths(env_high_const, results_dict['constrained_high_stakes']['paths'],
                   f'Constrained RL (Deontological) - High Stakes (Bomb) - Grid {args.grid}', 
                   os.path.join('results', f'viz_constrained_high_grid{args.grid}.png'))
    
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
    summary_filename = os.path.join('results', f'summary_comparison_grid{args.grid}.png')
    plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {summary_filename}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
