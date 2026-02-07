"""
Diagnostic Script - Find Out Why Agents Aren't Learning

This script will:
1. Test if paths are even reachable manually
2. Check if reward structure makes sense
3. Test a single training run with verbose output
4. Compare deterministic vs stochastic evaluation
"""

import torch
import numpy as np
from hostage_gridworld import HostageGridworld
from train_ppo import PPOAgent
from eval_utils import evaluate_agent_dual_mode, print_dual_eval_results


def test_manual_paths():
    """Manually test that both paths work."""
    print("="*70)
    print("TEST 1: Manual Path Verification")
    print("="*70)
    
    env = HostageGridworld(bomb_timer=None)
    
    # Test fast path
    print("\nFAST PATH (Up, Right, Right):")
    obs, info = env.reset()
    fast_path_actions = [0, 3, 3]  # up, right, right
    total_reward = 0
    
    for i, action in enumerate(fast_path_actions):
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        print(f"  Step {i+1}: {['Up','Down','Left','Right'][action]:>5} -> "
              f"pos={info['agent_pos']}, reward={reward:6.1f}, total={total_reward:6.1f}")
        if done:
            break
    
    if info['agent_pos'] == env.hostage_pos:
        print(f"  ✓ SUCCESS: Reached hostage in {info['steps_taken']} steps")
        print(f"  Civilians trampled: {len(info['civilians_trampled_this_episode'])}")
        print(f"  Final reward: {total_reward:.1f}")
    else:
        print(f"  ❌ FAILED: Did not reach hostage")
    
    # Test safe path
    print("\nSAFE PATH (Down, Right, Right, Right, Up, Up):")
    obs, info = env.reset()
    safe_path_actions = [1, 3, 3, 3, 0, 0]  # down, right x3, up x2
    total_reward = 0
    
    for i, action in enumerate(safe_path_actions):
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        print(f"  Step {i+1}: {['Up','Down','Left','Right'][action]:>5} -> "
              f"pos={info['agent_pos']}, reward={reward:6.1f}, total={total_reward:6.1f}")
        if done:
            break
    
    if info['agent_pos'] == env.hostage_pos:
        print(f"  ✓ SUCCESS: Reached hostage in {info['steps_taken']} steps")
        print(f"  Civilians trampled: {len(info['civilians_trampled_this_episode'])}")
        print(f"  Final reward: {total_reward:.1f}")
    else:
        print(f"  ❌ FAILED: Did not reach hostage")
    
    print()


def test_reward_structure():
    """Verify reward structure makes sense."""
    print("="*70)
    print("TEST 2: Reward Structure Analysis")
    print("="*70)
    
    env = HostageGridworld(bomb_timer=None)
    
    # Calculate expected rewards
    fast_steps = 3
    fast_civilians = 2
    fast_reward = (fast_steps * env.STEP_REWARD + 
                   fast_civilians * env.CIVILIAN_PENALTY + 
                   env.HOSTAGE_REWARD)
    
    safe_steps = 6
    safe_civilians = 0
    safe_reward = (safe_steps * env.STEP_REWARD + 
                   safe_civilians * env.CIVILIAN_PENALTY + 
                   env.HOSTAGE_REWARD)
    
    # Note: distance shaping adds to this
    print(f"\nReward components:")
    print(f"  STEP_REWARD: {env.STEP_REWARD}")
    print(f"  CIVILIAN_PENALTY: {env.CIVILIAN_PENALTY}")
    print(f"  HOSTAGE_REWARD: {env.HOSTAGE_REWARD}")
    print(f"  DISTANCE_REWARD_SCALE: {env.DISTANCE_REWARD_SCALE}")
    
    print(f"\nExpected rewards (WITHOUT distance shaping):")
    print(f"  Fast path (3 steps, 2 civilians): {fast_reward}")
    print(f"  Safe path (6 steps, 0 civilians): {safe_reward}")
    
    if fast_reward > safe_reward:
        print(f"\n  ⚠️  WARNING: Fast path has higher reward!")
        print(f"     Agent will ALWAYS prefer trampling in low stakes")
    elif safe_reward > fast_reward:
        print(f"\n  ✓ Safe path has higher reward in low stakes")
    
    # Check with bomb
    env_bomb = HostageGridworld(bomb_timer=5)
    timeout_penalty = env_bomb.BOMB_EXPLODE_PENALTY
    
    safe_with_bomb = (safe_steps * env.STEP_REWARD + timeout_penalty)
    
    print(f"\nWith bomb (timer=5):")
    print(f"  Fast path (3 steps): Can reach in time ✓")
    print(f"  Safe path (6 steps): Bomb explodes at step 5 ❌")
    print(f"  Safe path reward with bomb: {safe_with_bomb}")
    
    print()


def test_short_training_run():
    """Run a short training session with verbose output."""
    print("="*70)
    print("TEST 3: Short Training Run (100 episodes)")
    print("="*70)
    
    env = HostageGridworld(bomb_timer=None)
    agent = PPOAgent(obs_dim=5, action_dim=4, hidden_dim=64)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    
    from train_ppo import train_ppo
    
    print("\nTraining for 100 episodes...")
    train_ppo(env, agent, optimizer, num_episodes=100, print_every=25)
    
    print("\nEvaluating after 100 episodes...")
    det_results, stoch_results = evaluate_agent_dual_mode(env, agent, num_episodes=50)
    
    print_dual_eval_results(det_results, stoch_results, "After 100 Episodes")
    
    if det_results['hostage_save_rate'] == 0:
        print("\n❌ Agent learned NOTHING in 100 episodes")
        print("   This suggests reward shaping is broken or learning rate is wrong")
    
    return agent, det_results, stoch_results


def main():
    """Run all diagnostics."""
    print("\n" + "="*70)
    print("COMPREHENSIVE DIAGNOSTIC")
    print("="*70)
    print()
    
    # Test 1: Manual paths
    test_manual_paths()
    
    # Test 2: Reward structure
    test_reward_structure()
    
    # Test 3: Short training
    agent, det_results, stoch_results = test_short_training_run()
    
    # Final diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    if det_results['hostage_save_rate'] > 0:
        print("✓ Agent CAN learn to reach hostage with current setup")
        print("  Problem is likely: need more episodes or better hyperparameters")
    else:
        print("❌ Agent CANNOT learn with current setup")
        print("  Possible issues:")
        print("    1. Reward shaping is still too weak")
        print("    2. Hidden dimension too small")
        print("    3. Learning rate wrong")
        print("    4. Entropy too high (explores forever, never commits)")
    
    print()


if __name__ == "__main__":
    main()
