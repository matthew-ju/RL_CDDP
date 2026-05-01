import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ttest_rel, pearsonr
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'omnisafe_SRL'))

from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.agents.ppo_actor import PPOActor
from cddp.agents.deontic_oracle import DeonticOraclePlanner
from cddp.critics.qr_critic import DualCritics
from cddp.stakes.prospective import ProspectiveStakes
from hostage_gridworld import HostageGridworld

# Constants
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
JSON_PATH = os.path.join(RESULTS_DIR, 'mechanism_analysis.json')
PNG_PATH = os.path.join(RESULTS_DIR, 'mechanism_analysis_v2.png')
OS_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'cddp_results')
CRITICS_DIR = os.path.join(OS_RESULTS_DIR, 'phase3_critics_v2')

# Cosmetics
COLOR_PI_D = '#769958'      # Unified Green for S_d / pi_d
COLOR_PI_THETA = '#2D6FBC'  # Unified Blue for S_theta / pi_theta / pi_u
TAU = 0.3

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_dilemma_states(grid_id, T, margin_range=(-1.0, 1.0)):
    env = InstrumentedHostageEnv.make(layout_id=grid_id, bomb_timer=T)
    grid_size = env.grid_size
    blocked = set(env.env.walls) | set(env.env.civilians) | set(env.env.hostage_positions)
    
    dilemma_states = []
    l_safes = []
    margins = []
    
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) not in blocked:
                L_safe = env.get_L_safe((r, c))
                if L_safe == float('inf'):
                    continue
                margin = T - L_safe
                if margin_range[0] <= margin <= margin_range[1]:
                    dilemma_states.append((r, c))
                    l_safes.append(L_safe)
                    margins.append(margin)
    
    if not dilemma_states:
        print(f"ERROR: No dilemma states found for Grid {grid_id} with margin {margin_range}.")
        return [], 0, 0
        
    indices = np.random.choice(len(dilemma_states), 200, replace=True)
    sampled_states = [dilemma_states[i] for i in indices]
    
    print(f"Grid {grid_id} (T={T}): {len(dilemma_states)} unique dilemma states found.")
    print(f"  Mean L_safe: {np.mean(l_safes):.2f}, Mean margin: {np.mean(margins):.2f}")
    
    return sampled_states, np.mean(l_safes), np.mean(margins)

def run_rollouts(grid_id, T, states, policy_type='deontic', episodes_per_state=50):
    env = InstrumentedHostageEnv.make(layout_id=grid_id, bomb_timer=T)
    if policy_type == 'deontic':
        policy = DeonticOraclePlanner()
    else:
        actor_path = os.path.join(project_root, 'checkpoints', f'ppo_grid{grid_id}_finetuned_actor.pt')
        policy = PPOActor.from_checkpoint(actor_path, obs_dim=7)
    
    results = []
    max_steps = env.grid_size * 10
    
    for s0 in tqdm(states, desc=f"Rollouts {grid_id} {policy_type}"):
        for _ in range(episodes_per_state):
            env.env.random_start = False
            env.env.agent_start = s0
            obs, info = env.reset()
            
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                if policy_type == 'deontic':
                    action = policy.get_action(env)
                else:
                    action = policy.get_action(obs)
                
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                done = terminated or truncated
            
            reached_goal = (reward >= 10.0)
            res = {
                'steps_to_goal': steps if reached_goal else None,
                'reached_goal': reached_goal,
                'n_violations': info['trampled_civilians'],
                'T_rem_at_goal': info['T_rem'] if reached_goal else None
            }
            results.append(res)
    return results

def collect_observations(grid_id, T, states, policy_type='deontic'):
    env = InstrumentedHostageEnv.make(layout_id=grid_id, bomb_timer=T)
    if policy_type == 'deontic':
        policy = DeonticOraclePlanner()
    else:
        actor_path = os.path.join(project_root, 'checkpoints', f'ppo_grid{grid_id}_finetuned_actor.pt')
        policy = PPOActor.from_checkpoint(actor_path, obs_dim=7)
    
    observations = []
    max_steps = env.grid_size * 10
    
    for s0 in tqdm(states, desc=f"Obs collection {grid_id} {policy_type}"):
        env.env.random_start = False
        env.env.agent_start = s0
        obs, info = env.reset()
        
        observations.append((info['agent_pos'], info['T_rem']))
        
        done = False
        steps = 0
        while not done and steps < max_steps:
            if policy_type == 'deontic':
                action = policy.get_action(env)
            else:
                action = policy.get_action(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
            observations.append((info['agent_pos'], info['T_rem']))
            
    return observations

def get_critic_scores(grid_id, T, states, critics_sd, critics_st):
    env = InstrumentedHostageEnv.make(layout_id=grid_id, bomb_timer=T)
    oracle = DeonticOraclePlanner()
    
    sd_stakes_fn = ProspectiveStakes(critics=critics_sd, oracle=oracle)
    st_stakes_fn = ProspectiveStakes(critics=critics_st, oracle=oracle)
    
    sd_scores = []
    st_scores = []
    
    for s0 in states:
        env.env.random_start = False
        env.env.agent_start = s0
        env.reset()
        
        sd_val = sd_stakes_fn.compute(env)
        sd_scores.append(sd_val)
        
        st_val = st_stakes_fn.compute(env)
        st_scores.append(st_val)
        
    return np.array(sd_scores), np.array(st_scores)

def bin_trem(observations):
    bins = {'0-3': 0, '4-6': 0, '7-10': 0, '11+': 0}
    total = len(observations)
    if total == 0:
        return bins, 0.0
    
    low_count = 0
    for _, trem in observations:
        if trem <= 3:
            bins['0-3'] += 1
            low_count += 1
        elif trem <= 6:
            bins['4-6'] += 1
            low_count += 1
        elif trem <= 10:
            bins['7-10'] += 1
        else:
            bins['11+'] += 1
            
    fractions = {k: v/total for k, v in bins.items()}
    low_frac = low_count / total
    return fractions, low_frac

def apply_cosmetics(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc='left', fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(color='#CCCCCC', linewidth=0.5, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(42)
    
    # Check if we should skip computation and load from JSON
    if os.path.exists(JSON_PATH):
        print(f"Loading cached data from {JSON_PATH}...")
        with open(JSON_PATH, 'r') as f:
            mechanism_data = json.load(f)
        
        g1_states = mechanism_data['dilemma_states']['grid1']
        g2_states = mechanism_data['dilemma_states']['grid2']
        
        # We still need the scores and rollouts for plotting if not in JSON
        # Actually, let's just re-run the specific parts needed for plotting if missing.
        # For simplicity and correctness with new requests, I'll re-run the analysis 
        # but skip rollouts if they are too many (actually they are handled by tqdm).
    
    print("=== Step 0: Sampling shared starting states from dilemma zone ===")
    g1_states, g1_lsafe, g1_margin = get_dilemma_states(1, 6)
    g2_states, g2_lsafe, g2_margin = get_dilemma_states(2, 10)
    
    print("\n=== Step 1: Measuring rollout distance ===")
    g1_rollouts_d = run_rollouts(1, 6, g1_states, 'deontic', episodes_per_state=25)
    g1_rollouts_theta = run_rollouts(1, 6, g1_states, 'ppo', episodes_per_state=25)
    g2_rollouts_d = run_rollouts(2, 10, g2_states, 'deontic', episodes_per_state=25)
    g2_rollouts_theta = run_rollouts(2, 10, g2_states, 'ppo', episodes_per_state=25)
    
    print("\n=== Step 2: Measuring critic's input distribution ===")
    g1_obs_d = collect_observations(1, 6, g1_states, 'deontic')
    g1_obs_theta = collect_observations(1, 6, g1_states, 'ppo')
    g2_obs_d = collect_observations(2, 10, g2_states, 'deontic')
    g2_obs_theta = collect_observations(2, 10, g2_states, 'ppo')
    
    all_obs_d = g1_obs_d + g2_obs_d
    all_obs_theta = g1_obs_theta + g2_obs_theta
    d_fracs, d_low_frac = bin_trem(all_obs_d)
    theta_fracs, theta_low_frac = bin_trem(all_obs_theta)

    print("\n=== Step 3: Measuring critic scores on shared states ===")
    def load_critics(grid_id, ppo_trained=False):
        suffix = "_ppo_trained" if ppo_trained else ""
        path = os.path.join(CRITICS_DIR, f'grid{grid_id}_dual_critics{suffix}.pt')
        critics = DualCritics(obs_dim=10, num_actions=4)
        critics.load_state_dict(torch.load(path, map_location='cpu'))
        critics.z_free.eval(); critics.z_deon.eval(); critics.failure_predictor.eval()
        return critics

    critics_g1_sd = load_critics(1, False)
    critics_g1_st = load_critics(1, True)
    critics_g2_sd = load_critics(2, False)
    critics_g2_st = load_critics(2, True)
    
    g1_sd_scores, g1_st_scores = get_critic_scores(1, 6, g1_states, critics_g1_sd, critics_g1_st)
    g2_sd_scores, g2_st_scores = get_critic_scores(2, 10, g2_states, critics_g2_sd, critics_g2_st)
    
    sd_scores = np.concatenate([g1_sd_scores, g2_sd_scores])
    st_scores = np.concatenate([g1_st_scores, g2_st_scores])
    
    print("\n=== Step 4: Producing the mechanism figure v2 ===")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    # Panel (a) & (b): Steps-to-goal
    for i, (grid_id, rolls_d, rolls_theta) in enumerate([(1, g1_rollouts_d, g1_rollouts_theta), (2, g2_rollouts_d, g2_rollouts_theta)]):
        ax = axes[0, i]
        steps_d = [r['steps_to_goal'] for r in rolls_d if r['reached_goal']]
        steps_theta = [r['steps_to_goal'] for r in rolls_theta if r['reached_goal']]
        
        bins = range(min(steps_d + steps_theta), max(steps_d + steps_theta)+2)
        ax.hist(steps_d, bins=bins, alpha=0.5, color=COLOR_PI_D, label='$\pi_d$', density=True)
        ax.hist(steps_theta, bins=bins, alpha=0.5, color=COLOR_PI_THETA, label='$\pi_\\theta$', density=True)
        
        ax.axvline(np.mean(steps_d), color=COLOR_PI_D, linestyle='--', linewidth=1.5)
        ax.axvline(np.mean(steps_theta), color=COLOR_PI_THETA, linestyle='--', linewidth=1.5)
        
        apply_cosmetics(ax, title=f"({chr(97+i)}) Steps to Goal — Grid {grid_id} Dilemma Zone", 
                        xlabel="Episode Length (steps)", ylabel="Density")
        ax.legend(fontsize=8, frameon=False)

    # Panel (c): Critic Input Distribution
    ax = axes[1, 0]
    x = np.arange(len(d_fracs))
    width = 0.35
    ax.bar(x - width/2, [d_fracs[k] for k in ['0-3', '4-6', '7-10', '11+']], width, color=COLOR_PI_D, label='$\pi_d$')
    ax.bar(x + width/2, [theta_fracs[k] for k in ['0-3', '4-6', '7-10', '11+']], width, color=COLOR_PI_THETA, label='$\pi_\\theta$')
    ax.set_xticks(x)
    ax.set_xticklabels(['0-3', '4-6', '7-10', '11+'])
    apply_cosmetics(ax, title="(c) Critic Input Distribution ($T_{\mathrm{rem}}$ bins)", 
                    ylabel="Fraction of Observations")
    ax.legend(fontsize=8, frameon=False)

    # Panel (d): Score Comparison with Marginals
    ax = axes[1, 1]
    
    # Create marginal axes
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 0.8, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 0.8, pad=0.1, sharey=ax)
    
    # Scatter plot
    ax.scatter(g1_sd_scores, g1_st_scores, color='#5DADE2', alpha=0.4, s=15, label='Grid 1')
    ax.scatter(g2_sd_scores, g2_st_scores, color='#F39C12', alpha=0.4, s=15, label='Grid 2')
    ax.plot([0, 1], [0, 1], 'gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(TAU, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(TAU, color='gray', linestyle='--', linewidth=0.8)
    
    # Quadrant fill (Bottom-right: Sd >= 0.3, S_theta < 0.3)
    ax.add_patch(plt.Rectangle((TAU, 0), 1-TAU, TAU, color=COLOR_PI_THETA, alpha=0.08, transform=ax.transData))
    
    # Centroid calculation and annotation
    br_mask = (sd_scores >= TAU) & (st_scores < TAU)
    n_br = np.sum(br_mask)
    pct_br = (n_br / len(sd_scores)) * 100
    if n_br > 0:
        c_x = np.mean(sd_scores[br_mask])
        c_y = np.mean(st_scores[br_mask])
        ax.text(c_x, c_y, f"N={n_br} ({pct_br:.1f}%)", color=COLOR_PI_THETA, 
                fontweight='bold', ha='center', va='center', fontsize=9)
    
    # Pearson r and p-value
    r_val, p_val = pearsonr(sd_scores, st_scores)
    ax.text(0.05, 0.95, f"Pearson $r={r_val:.3f}$\n$p={p_val:.1e}$", 
            transform=ax.transAxes, verticalalignment='top', fontsize=9, fontweight='bold')
    
    # Marginal Histograms
    bins_h = np.linspace(0, 1, 31)
    ax_histx.hist(sd_scores, bins=bins_h, color=COLOR_PI_D, alpha=0.3, density=True)
    ax_histy.hist(st_scores, bins=bins_h, color=COLOR_PI_THETA, alpha=0.3, density=True, orientation='horizontal')
    
    # Clean up marginal axes
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histx.yaxis.set_tick_params(labelleft=False)
    ax_histx.set_yticks([])
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histx.set_title("(d) Score Comparison on Shared Dilemma States", loc='left', fontsize=10)
    
    ax_histy.yaxis.set_tick_params(labelleft=False)
    ax_histy.xaxis.set_tick_params(labelbottom=False)
    ax_histy.set_xticks([])
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)

    apply_cosmetics(ax, title=None, 
                    xlabel="$S_d(s_0)$", ylabel="$S_\\theta(s_0)$")
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {PNG_PATH}")

if __name__ == '__main__':
    main()
