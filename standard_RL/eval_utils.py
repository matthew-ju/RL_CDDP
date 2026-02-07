"""
Improved Evaluation with Deterministic AND Stochastic Testing

This will reveal if the issue is deterministic argmax vs stochastic sampling.
"""

import numpy as np
import torch


def evaluate_agent_dual_mode(env, agent, num_episodes=100):
    """
    Evaluate agent in BOTH deterministic and stochastic modes.
    
    Returns two result dicts: deterministic_results, stochastic_results
    """
    
    results = {'deterministic': {}, 'stochastic': {}}
    
    for mode in ['deterministic', 'stochastic']:
        is_deterministic = (mode == 'deterministic')
        
        total_rewards = []
        civilian_violations = []
        bomb_explosions = 0
        hostages_saved = 0
        paths_taken = []
        step_counts = []
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            path = [tuple(info['agent_pos'])]
            
            for step_num in range(50):
                action = agent.get_action(obs, deterministic=is_deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                path.append(tuple(info['agent_pos']))
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            civilian_violations.append(len(info['civilians_trampled_this_episode']))
            step_counts.append(info['steps_taken'])
            
            # Check outcomes
            if info['agent_pos'] == env.hostage_pos:
                hostages_saved += 1
            
            if env.bomb_timer_max is not None and info['bomb_time_left'] == 0 and info['agent_pos'] != env.hostage_pos:
                bomb_explosions += 1
            
            if len(paths_taken) < 5:
                paths_taken.append(path)
        
        results[mode] = {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_civilian_violations': np.mean(civilian_violations),
            'std_civilian_violations': np.std(civilian_violations),
            'avg_steps': np.mean(step_counts),
            'hostages_saved': hostages_saved,
            'hostage_save_rate': hostages_saved / num_episodes,
            'bomb_explosions': bomb_explosions,
            'bomb_explosion_rate': bomb_explosions / num_episodes,
            'paths': paths_taken,
            'timeout_rate': np.mean([s >= 30 for s in step_counts])
        }
    
    return results['deterministic'], results['stochastic']


def print_dual_eval_results(det_results, stoch_results, scenario_name):
    """Print comparison of deterministic vs stochastic."""
    
    print(f"\n{scenario_name} - DUAL MODE EVALUATION")
    print("="*70)
    print(f"{'Metric':<30} {'Deterministic':<20} {'Stochastic':<20}")
    print("-"*70)
    
    metrics = [
        ('Hostage Save Rate', 'hostage_save_rate', '%'),
        ('Avg Civilian Violations', 'avg_civilian_violations', ''),
        ('Avg Reward', 'avg_reward', ''),
        ('Avg Steps', 'avg_steps', ''),
        ('Timeout Rate', 'timeout_rate', '%'),
    ]
    
    if 'bomb_explosion_rate' in det_results:
        metrics.append(('Bomb Explosion Rate', 'bomb_explosion_rate', '%'))
    
    for label, key, unit in metrics:
        det_val = det_results[key]
        stoch_val = stoch_results[key]
        
        if unit == '%':
            det_str = f"{det_val*100:.1f}%"
            stoch_str = f"{stoch_val*100:.1f}%"
        else:
            det_str = f"{det_val:.2f}"
            stoch_str = f"{stoch_val:.2f}"
        
        # Highlight mismatches
        if abs(det_val - stoch_val) > 0.1:
            marker = " ⚠️ MISMATCH"
        else:
            marker = ""
        
        print(f"{label:<30} {det_str:<20} {stoch_str:<20}{marker}")
    
    print("="*70)
    
    # Diagnosis
    if det_results['hostage_save_rate'] == 0 and stoch_results['hostage_save_rate'] > 0:
        print("❌ CRITICAL: Deterministic evaluation fails but stochastic succeeds!")
        print("   This means the policy has NO coherent strategy, only random luck.")
    elif det_results['timeout_rate'] > 0.9:
        print("❌ CRITICAL: Agent timing out >90% of episodes")
        print("   Agent never learned to reach the hostage at all.")


if __name__ == "__main__":
    print("Dual-mode evaluation tester")
    print("This will be used in the main evaluation script")
