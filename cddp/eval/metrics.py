"""
CDDP Evaluation Metrics and Experiment Runner.

Computes the key metrics for comparing CDDP against baselines:
    1. Save Rate: fraction of episodes where hostage is saved
    2. Violation Rate: average civilian violations per episode
    3. Safety Score: episodes with zero violations / total
    4. Switch Delay: for failure-band states, how many extra steps before switching
    5. Urgency Gap: V_free - V_deon (distributional difference)
"""

import os
import sys
import json
import glob
import numpy as np
import torch

_omnisafe_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'omnisafe_SRL')
sys.path.insert(0, os.path.abspath(_omnisafe_dir))

from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.shield.action_shield import ActionShield
from cddp.stakes.analytical import AnalyticalStakesDmin
from cddp.agents.ppo_actor import PPOActor
from cddp.agents.cddp_agent import CDDPAgent


def min_violations_needed(env_wrapper, start_pos=None, start_t_rem=None) -> int:
    """Compute the minimum number of civilians that must be trampled to reach the hostage."""
    import heapq
    
    env_base = env_wrapper.env if hasattr(env_wrapper, 'env') else env_wrapper
    start = tuple(env_base.agent_pos) if start_pos is None else tuple(start_pos)
    
    targets = [hp for hp in env_base.hostage_positions if hp not in env_base.hostages_saved]
    if not targets:
        targets = env_base.hostage_positions
        
    target_set = set(targets)
    active_civilians = set(env_base.civilians) - env_base.trampled_civilians
    walls = env_base.walls
    
    # Expanded state space for time-aware feasibility: (r, c, time_remaining)
    if start_t_rem is None:
        start_t_rem = env_wrapper.get_T_rem() if hasattr(env_wrapper, 'get_T_rem') else env_base.bomb_time_left
    
    if start_t_rem == float('inf') or start_t_rem < 0:
        start_t_rem = 999  # effectively no limit
    
    pq = [(0, start[0], start[1], start_t_rem)] # cost, r, c, t_rem
    visited = {}
    
    while pq:
        cost, r, c, t_rem = heapq.heappop(pq)
        
        if (r, c) in target_set:
            if t_rem >= 0:
                return cost
            
        if t_rem <= 0:
            continue
            
        # Standardize representation
        state_key = (r, c, t_rem)
        if state_key in visited and visited[state_key] <= cost:
            continue
        visited[state_key] = cost
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env_base.grid_size and 0 <= nc < env_base.grid_size:
                if (nr, nc) not in walls:
                    step_cost = 1 if (nr, nc) in active_civilians else 0
                    heapq.heappush(pq, (cost + step_cost, nr, nc, t_rem - 1))
                        
    return float('inf')  # Unreachable or impossible in time

def evaluate_agent(agent, layout_id, bomb_timer, n_episodes=100,
                   deterministic=True, max_steps=50, random_start=False, p_slip=0.0,
                   episode_seeds=None):
    """Full CDDP evaluation with detailed metrics.

    Args:
        episode_seeds: optional list[int] of length n_episodes.  When provided,
            each episode calls env.reset(seed=episode_seeds[ep]) so that the
            start-state distribution is IDENTICAL across timers and models.
            L_safe(s0) is also measured per episode and returned in results.

    Returns:
        dict with: save_rate, violation_rate, safety_score, avg_reward,
                   mode_counts, zero_violation_rate, avg_violations_per_rescue,
                   deontic_violation_count, utilitarian_violation_count,
                   failure_band_entries, l_safe_per_episode,
                   step_data (per-step metadata)
    """
    saves = 0
    violations_per_ep = []
    rewards_per_ep = []
    safe_episodes = 0
    mode_counts = {'deontic': 0, 'utilitarian': 0}
    shield_overrides = 0
    switch_delays = []
    lateness_list = []
    earliness_list = []
    switched_ever_count = 0
    step_data = []  # For detailed analysis
    start_states_data = []

    # Enriched tracking
    deontic_violations = 0
    utilitarian_violations = 0
    failure_band_entries = 0
    violations_in_rescues = []
    excess_violations_per_ep = []
    episodes_with_excess = 0
    
    # New switching metrics
    false_positive_switches = 0
    late_switches = 0
    episodes_with_oscillation = 0
    
    infeasible_count = 0
    unique_start_positions = set()
    l_safe_per_episode = []

    for ep in range(n_episodes):
        env = InstrumentedHostageEnv.make(
            layout_id=layout_id, bomb_timer=bomb_timer,
            random_start=random_start, p_slip=p_slip)
        print(f"[DEBUG] grid_id={env.grid_id}, layout_hash={hash(str(env.walls) + str(set(env.civilians)))}, n_unsafe={len(env.civilians)}")
        
        # Parameterization checks
        if hasattr(agent, 'grid_id') and agent.grid_id is not None:
            assert agent.grid_id == env.grid_id, f"Parameterization Bug: Agent expects grid {agent.grid_id} but env is grid {env.grid_id}"

        # Seed-pinned reset: ensures identical start-state distribution across timers
        ep_seed = int(episode_seeds[ep]) if episode_seeds is not None else None
        obs, info = env.reset(seed=ep_seed)
        
        unique_start_positions.add(tuple(env.agent_pos))
        d_min = env.get_D_min()
        t_rem = env.get_T_rem()
        # Record L_safe at start state for determinism validation
        l_safe_s0 = env.get_L_safe()
        l_safe_per_episode.append(float(l_safe_s0) if l_safe_s0 != float('inf') else float('nan'))
        if d_min > t_rem or t_rem <= 0:
            infeasible_count += 1
            if hasattr(agent, 'reset_mode_tracking'):
                agent.reset_mode_tracking()
            continue
        
        # Calculate optimal min violations needed from start state
        min_v = min_violations_needed(env)
        if hasattr(agent, 'reset_mode_tracking'):
            agent.reset_mode_tracking()
        done = False
        ep_reward = 0
        ep_steps = 0
        first_utilitarian = None
        first_failure_band = None
        entered_failure_band = False

        # Per-step mode and violation tracking for this episode
        ep_deontic_viols = 0
        ep_util_viols = 0
        
        mode_history = []
        has_oscillated = False

        while not done and ep_steps < max_steps:
            action, act_info = agent.select_action(
                obs, env, deterministic=deterministic)
            current_mode = act_info['mode']
            mode_counts[current_mode] += 1
            if current_mode != mode_history[-1] if mode_history else True:
                mode_history.append(current_mode)
                
            # Check for generic D->U->D oscillation
            if len(mode_history) >= 3 and not has_oscillated:
                # If we see any subsequence of 'deontic', 'utilitarian', 'deontic'
                for i in range(len(mode_history) - 2):
                    if mode_history[i] == 'deontic' and mode_history[i+1] == 'utilitarian' and mode_history[i+2] == 'deontic':
                        has_oscillated = True
                        break
            if act_info.get('action_was_shielded'):
                shield_overrides += 1

            # Track failure band / utilitarian switching
            if info.get('in_failure_band') and first_failure_band is None:
                first_failure_band = ep_steps
                entered_failure_band = True
            if act_info['mode'] == 'utilitarian' and first_utilitarian is None:
                first_utilitarian = ep_steps

            # Track per-mode violations
            cost = env.get_constraint_cost(action)
            if cost > 0:
                if act_info['mode'] == 'deontic':
                    ep_deontic_viols += 1
                else:
                    ep_util_viols += 1

            p_cat_hat_val = float(act_info.get('p_cat_hat', 0.0))
            if not (0.0 <= p_cat_hat_val <= 1.0):
                # We expect small numerical noise occasionally, clip it if it's very close
                if -1e-5 <= p_cat_hat_val <= 1.0 + 1e-5:
                    p_cat_hat_val = float(np.clip(p_cat_hat_val, 0.0, 1.0))
                else:
                    print(f"INVARIANT VIOLATION: p_cat_hat = {p_cat_hat_val} is out of bounds [0, 1]!")
                    print(f"Step: {ep_steps}")
                    print(f"Obs: {obs}")
                    print(f"Env state: D_min={info.get('D_min')}, T_rem={info.get('T_rem')}")
                    sys.exit(1)

            step_data.append({
                'episode': ep,
                'step': ep_steps,
                'mode': act_info['mode'],
                'stakes': act_info['stakes'],
                'p_cat_hat': p_cat_hat_val,
                'in_failure_band': info.get('in_failure_band', False),
                'D_min': info.get('D_min', -1),
                'T_rem': info.get('T_rem', -1),
                'pos': tuple(env.agent_pos),
            })
            if ep_steps == 0:
                start_states_data.append({
                    'episode': ep,
                    'p_cat_hat': p_cat_hat_val,
                    'min_v': min_v,
                    'pos': tuple(env.agent_pos),
                })

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

        # Per-episode metrics
        ep_violations = len(env.trampled_civilians)
        saved = tuple(env.agent_pos) == env.hostage_pos
        
        if saved and min_v != float('inf'):
            excess = max(0, ep_violations - min_v)
            excess_violations_per_ep.append(excess)
            if excess > 0:
                episodes_with_excess += 1
            
        violations_per_ep.append(ep_violations)
        rewards_per_ep.append(ep_reward)
        deontic_violations += ep_deontic_viols
        utilitarian_violations += ep_util_viols
        if entered_failure_band:
            failure_band_entries += 1
        if ep_violations == 0:
            safe_episodes += 1

        saved = tuple(env.agent_pos) == env.hostage_pos
        if saved:
            saves += 1
            violations_in_rescues.append(ep_violations)

        # Switch delay
        if first_utilitarian is not None:
            switched_ever_count += 1
            
        if first_failure_band is not None and first_utilitarian is not None:
            delay = first_utilitarian - first_failure_band
            switch_delays.append(delay)
            lateness_list.append(max(0, delay))
            earliness_list.append(max(0, -delay))
            
            if first_utilitarian > first_failure_band:
                late_switches += 1
        elif first_failure_band is not None and first_utilitarian is None:
            # Entered failure band but never switched
            # The "first_utilitarian" is effectively max_steps or ep_steps
            delay = ep_steps - first_failure_band
            switch_delays.append(delay)
            lateness_list.append(max(0, delay))
            late_switches += 1
            
        if first_utilitarian is not None and first_failure_band is None:
            false_positive_switches += 1
            
        if has_oscillated:
            episodes_with_oscillation += 1

    feasible_eps = n_episodes - infeasible_count
    den = feasible_eps if feasible_eps > 0 else 1

    results = {
        'feasible_success_rate': saves / den,
        'save_rate': saves / den,   # Overwritten to mean feasible save rate
        'infeasible_rate': infeasible_count / n_episodes,
        'unique_start_positions': len(unique_start_positions),
        'violation_rate': np.mean(violations_per_ep) if violations_per_ep else np.nan,
        'excess_violation_rate': np.mean(excess_violations_per_ep) if excess_violations_per_ep else np.nan,
        'episodes_with_excess_rate': (episodes_with_excess / len(excess_violations_per_ep)) if excess_violations_per_ep else np.nan,
        'safety_score': safe_episodes / den,
        'zero_violation_rate': safe_episodes / den,  # alias
        'avg_switch_delay': np.mean(switch_delays) if switch_delays else np.nan,
        'std_switch_delay': np.std(switch_delays) if switch_delays else np.nan,
        'avg_lateness': np.mean(lateness_list) if lateness_list else np.nan,
        'avg_earliness': np.mean(earliness_list) if earliness_list else np.nan,
        'switched_ever_rate': switched_ever_count / den,
        'avg_reward': np.mean(rewards_per_ep) if rewards_per_ep else np.nan,
        'std_reward': np.std(rewards_per_ep) if rewards_per_ep else np.nan,
        'mode_counts': dict(mode_counts),
        'shield_overrides': shield_overrides,
        'n_episodes': n_episodes,
        # Enriched metrics
        'deontic_violation_count': deontic_violations,
        'utilitarian_violation_count': utilitarian_violations,
        'failure_band_entries': failure_band_entries,
        'avg_violations_per_rescue': (np.mean(violations_in_rescues)
                                       if violations_in_rescues else 0.0),
        # Switching quality metrics
        'false_positive_switch_rate': false_positive_switches / den,
        'late_switch_rate': late_switches / failure_band_entries if failure_band_entries > 0 else 0.0,
        'oscillation_rate': episodes_with_oscillation / den,
    }

    if switch_delays:
        results['avg_switch_delay'] = np.mean(switch_delays)
        results['std_switch_delay'] = np.std(switch_delays)
        
    results['start_states_data'] = start_states_data
    results['l_safe_per_episode'] = l_safe_per_episode

    return results, step_data


def evaluate_ppo_baseline(layout_id, bomb_timer, actor_path, n_episodes=100, deterministic=True, random_start=False, p_slip=0.0):
    """Evaluate raw PPO baseline and track performance and excess violations."""
    actor = PPOActor.from_checkpoint(actor_path, obs_dim=7)

    saves, violations, rewards = 0, [], []
    excess_violations = []
    episodes_with_excess = 0
    infeasible_count = 0
    unique_start_positions = set()
    
    for _ in range(n_episodes):
        env = InstrumentedHostageEnv.make(
            layout_id=layout_id, bomb_timer=bomb_timer, 
            random_start=random_start, p_slip=p_slip)
            
        if hasattr(actor, 'grid_id') and actor.grid_id is not None:
            assert actor.grid_id == env.grid_id, f"Parameterization Bug: Actor expects grid {actor.grid_id} but env is grid {env.grid_id}"

        obs, _ = env.reset()
        
        unique_start_positions.add(tuple(env.agent_pos))
        d_min = env.get_D_min()
        t_rem = env.get_T_rem()
        if d_min > t_rem or t_rem <= 0:
            infeasible_count += 1
            continue
            
        min_v = min_violations_needed(env)
        
        done = False
        ep_r = 0
        while not done:
            action = actor.get_action(obs, deterministic=deterministic)
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_r += r
        rewards.append(ep_r)
        
        ep_violations = len(env.trampled_civilians)
        violations.append(ep_violations)
        
        saved = tuple(env.agent_pos) == env.hostage_pos
        if saved:
            saves += 1
            if min_v != float('inf'):
                excess = max(0, ep_violations - min_v)
                excess_violations.append(excess)
                if excess > 0:
                    episodes_with_excess += 1

    feasible_eps = n_episodes - infeasible_count
    den = feasible_eps if feasible_eps > 0 else 1

    return {
        'feasible_success_rate': saves / den,
        'save_rate': saves / den,
        'infeasible_rate': infeasible_count / n_episodes,
        'unique_start_positions': len(unique_start_positions),
        'violation_rate': np.mean(violations) if violations else np.nan,
        'excess_violation_rate': np.mean(excess_violations) if excess_violations else np.nan,
        'episodes_with_excess_rate': (episodes_with_excess / len(excess_violations)) if excess_violations else np.nan,
        'safety_score': sum(1 for v in violations if v == 0) / den,
        'avg_reward': np.mean(rewards) if rewards else np.nan,
        'std_reward': np.std(rewards) if rewards else np.nan,
        'label': 'PPO (baseline)',
    }


def evaluate_constrained_baseline(layout_id, bomb_timer, checkpoint_path,
                                  n_episodes=100, deterministic=True):
    """Evaluate constrained PPOLag (no CDDP wrapping)."""
    actor = PPOActor.from_checkpoint(checkpoint_path)

    saves, violations, rewards = 0, [], []
    for _ in range(n_episodes):
        env = InstrumentedHostageEnv.make(
            layout_id=layout_id, bomb_timer=bomb_timer)
        obs, _ = env.reset()
        done = False
        ep_r = 0
        while not done:
            action = actor.get_action(obs, deterministic=deterministic)
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_r += r
        rewards.append(ep_r)
        violations.append(len(env.trampled_civilians))
        if tuple(env.agent_pos) == env.hostage_pos:
            saves += 1

    return {
        'save_rate': saves / n_episodes,
        'violation_rate': np.mean(violations),
        'safety_score': sum(1 for v in violations if v == 0) / n_episodes,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'label': 'Constrained (PPOLag)',
    }


def print_comparison_table(results_dict):
    """Print a comparison table of all agents.

    Args:
        results_dict: {'agent_name': results_dict, ...}
    """
    print(f"\n{'Agent':<25} {'Save%':>7} {'Viol':>6} {'Safety%':>8} "
          f"{'Reward':>8} {'Switch':>7}")
    print("─" * 65)

    for name, r in results_dict.items():
        switch = f"{r.get('avg_switch_delay', 'N/A'):>5.1f}" if 'avg_switch_delay' in r else "  N/A"
        print(f"{name:<25} {r['save_rate']*100:>6.1f}% {r['violation_rate']:>6.2f} "
              f"{r['safety_score']*100:>7.1f}% {r['avg_reward']:>8.1f} {switch}")


def save_results(results_dict, save_path):
    """Save results as JSON."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    serializable = {}
    for k, v in results_dict.items():
        sv = {}
        for kk, vv in v.items():
            if isinstance(vv, (np.floating, np.integer)):
                sv[kk] = float(vv)
            else:
                sv[kk] = vv
        serializable[k] = sv
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Saved: {save_path}")
