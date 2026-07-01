import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'omnisafe_SRL'))

from hostage_gridworld import HostageGridworld
from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.agents.ppo_actor import PPOActor
from cddp.agents.deontic_oracle import DeonticOraclePlanner
from cddp.agents.cddp_agent import CDDPAgent
from cddp.critics.qr_critic import DualCritics
from cddp.stakes.prospective import ProspectiveStakes

# Constants for cosmetics
COLOR_SD = '#2D6FBC'
TAUS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def run_eval(grid_id, env_kwargs, tau, n_seeds=15, eps_per_seed=200):
    """Run evaluation for a specific grid and threshold."""
    critics_dir = os.path.join(project_root, 'checkpoints')
    st_path = os.path.join(critics_dir, f'grid{grid_id}_dual_critics_ppo_trained.pt')
    
    # Load PPO Actor
    actor_path = os.path.join(project_root, f'checkpoints/ppo_grid{grid_id}_finetuned_actor.pt')
    actor = PPOActor.from_checkpoint(actor_path, obs_dim=7)
    
    # Load S_theta Critics
    critics = DualCritics(obs_dim=10, num_actions=4)
    critics.load_state_dict(torch.load(st_path))
    critics.z_free.eval()
    critics.z_deon.eval()
    critics.failure_predictor.eval()
    
    oracle_for_stakes = DeonticOraclePlanner()
    stakes_fn = ProspectiveStakes(critics=critics, oracle=oracle_for_stakes)
    
    # Fixed value patch to disable force_util override in CDDPAgent
    stakes_fn.fixed_val = 0.0
    
    shield = DeonticOraclePlanner()
    agent = CDDPAgent(actor, shield, stakes_fn, tau=tau)
    
    all_sr, all_v, all_t, all_never = [], [], [], []
    
    layout_info = HostageGridworld.GRID_LAYOUTS.get(grid_id, HostageGridworld.GRID_LAYOUTS[1])
    max_steps = layout_info.get('grid_size', 5) * 10

    for seed in range(n_seeds):
        sr, v, t_switch, never = 0, 0, 0, 0
        env = InstrumentedHostageEnv.make(layout_id=grid_id, random_start=True, p_slip=0.05, **env_kwargs)
        
        for ep in range(eps_per_seed):
            obs, info = env.reset(seed=seed * 1000 + ep)
            agent.reset_mode_tracking()
            done = False
            switched = False
            t_ep = 0
            ep_success = False
            
            while not done and t_ep < max_steps:
                action, step_info = agent.select_action(obs, env)
                
                # Check if UTIL was triggered for the first time
                if step_info['mode'] == 'utilitarian' and not switched:
                    t_switch += t_ep
                    switched = True
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated and reward >= 10.0:
                    ep_success = True
                
                t_ep += 1
                done = terminated or truncated
            
            if ep_success:
                sr += 1
            v += info.get('trampled_civilians', 0)
            if not switched:
                never += 1
                
        all_sr.append(sr / eps_per_seed)
        all_v.append(v / eps_per_seed)
        all_never.append(never / eps_per_seed)
        if (eps_per_seed - never) > 0:
            all_t.append(t_switch / (eps_per_seed - never))
        else:
            all_t.append(0)
            
    return {
        'sr_mean': np.mean(all_sr), 'sr_std': np.std(all_sr),
        'v_mean': np.mean(all_v), 'v_std': np.std(all_v),
        't_mean': np.mean(all_t), 't_std': np.std(all_t),
        'never_switched': np.mean(all_never)
    }

def apply_cosmetics(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc='left', fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(color='#CCCCCC', linewidth=0.5, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def main():
    print("=== Step 6: Threshold Sweep Analysis (Irreducibility Test) ===")
    
    with open(os.path.join(project_root, 'results/switch_timing_comparison_v2.json'), 'r') as f:
        baseline = json.load(f)
    
    configs = [
        (1, 5, 'tight'), (1, 6, 'tight'), (1, 7, 'tight'),
        (2, 9, 'tight'), (2, 10, 'tight'),
        (2, 13, 'easy'), (2, 14, 'easy'), (2, 15, 'easy')
    ]
    
    results = {}
    for tau in TAUS:
        results[tau] = {1: {}, 2: {}}
        print(f"\nEvaluating tau' = {tau:.2f}...")
        for grid_id, T, regime in configs:
            print(f"  Grid {grid_id}, T={T} ({regime})...")
            res = run_eval(grid_id, {'bomb_timer': T}, tau)
            results[tau][grid_id][T] = res
            print(f"    SR: {res['sr_mean']:.3f}, Viol: {res['v_mean']:.3f}, Never: {res['never_switched']:.2f}")

    os.makedirs('results', exist_ok=True)
    with open(os.path.join(project_root, 'results/threshold_sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("Threshold Sweep Summary Table")
    print("-" * 80)
    print(f"{'tau':<6} | {'Tight SR (avg)':<15} | {'Easy Viol (avg)':<15} | {'Tight Fixed?':<12} | {'Easy Fixed?':<12} | {'Both?'}")
    print("-" * 80)
    
    tight_baseline_sr = []
    for g, t, r in configs:
        if r == 'tight':
            tight_baseline_sr.append(float(baseline[f'grid{g}'][str(t)]['sr_d']))
    target_sr = np.mean(tight_baseline_sr)
    
    easy_baseline_v = []
    for g, t, r in configs:
        if r == 'easy':
            easy_baseline_v.append(float(baseline[f'grid{g}'][str(t)]['v_d']))
    target_v = np.mean(easy_baseline_v)
    
    any_both = False
    for tau in TAUS:
        t_srs, e_vs = [], []
        t_fixed, e_fixed = True, True
        
        for g, t, r in configs:
            val = results[tau][g][t]
            base_val = baseline[f'grid{g}'][str(t)]
            if r == 'tight':
                t_srs.append(val['sr_mean'])
                if val['sr_mean'] < float(base_val['sr_d']) - 0.02:
                    t_fixed = False
            else:
                e_vs.append(val['v_mean'])
                if val['v_mean'] > float(base_val['v_d']) + 0.05:
                    e_fixed = False
        
        mean_t_sr = np.mean(t_srs)
        mean_e_v = np.mean(e_vs)
        both = t_fixed and e_fixed
        if both: any_both = True
        
        print(f"{tau:<6.2f} | {mean_t_sr:15.3f} | {mean_e_v:15.3f} | {str(t_fixed):<12} | {str(e_fixed):<12} | {str(both)}")

    print(f"{'S_d (base)':<6} | {target_sr:15.3f} | {target_v:15.3f} | {'✓':<12} | {'✓':<12} | {'✓'}")
    
    if any_both:
        print("\nRESULT: Miscalibration is correctable. Threshold-calibration hypothesis is NOT ruled out.")
    else:
        print("\nRESULT: Miscalibration is irreducible. No single tau' simultaneously fixes both regimes.")

    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0, 1, len(TAUS)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sort T for plotting
    tight_configs = [c for c in configs if c[2] == 'tight']
    easy_configs = [c for c in configs if c[2] == 'easy']
    
    # Plot Tight SR (Success Rate)
    # Since we have multiple grids, let's just plot the points/means
    # To keep it simple and clean, let's plot T values vs SR
    # We combine them into a single sorted X axis
    all_tight_ts = sorted(list(set([c[1] for c in tight_configs])))
    for i, tau in enumerate(TAUS):
        y_vals = []
        for t in all_tight_ts:
            # Avg SR across grids for this T
            srs = [results[tau][g][t]['sr_mean'] for g, tt, r in tight_configs if tt == t]
            y_vals.append(np.mean(srs))
        ax1.plot(all_tight_ts, y_vals, '-o', color=colors[i], label=f"$\\tau'={tau:.2f}$", markersize=4)
        
    sd_sr_vals = []
    for t in all_tight_ts:
        srs = [float(baseline[f'grid{g}'][str(t)]['sr_d']) for g, tt, r in tight_configs if tt == t]
        sd_sr_vals.append(np.mean(srs))
    ax1.plot(all_tight_ts, sd_sr_vals, '--', color=COLOR_SD, label='$S_d$ (base)', linewidth=2)
    ax1.fill_between(all_tight_ts, [v - 0.02 for v in sd_sr_vals], [v + 0.02 for v in sd_sr_vals], color=COLOR_SD, alpha=0.1)
    apply_cosmetics(ax1, title="(a) Tight Regime — Success Rate by $\\tau'$ ", xlabel="Bomb Timer $T$", ylabel="Success Rate")
    ax1.legend(framealpha=0.9, edgecolor='#CCCCCC', fontsize=8, ncol=2)
    
    # Plot Easy Violations
    all_easy_ts = sorted(list(set([c[1] for c in easy_configs])))
    for i, tau in enumerate(TAUS):
        y_vals = []
        for t in all_easy_ts:
            vs = [results[tau][g][t]['v_mean'] for g, tt, r in easy_configs if tt == t]
            y_vals.append(np.mean(vs))
        ax2.plot(all_easy_ts, y_vals, '-o', color=colors[i], label=f"$\\tau'={tau:.2f}$", markersize=4)
        
    sd_v_vals = []
    for t in all_easy_ts:
        vs = [float(baseline[f'grid{g}'][str(t)]['v_d']) for g, tt, r in easy_configs if tt == t]
        sd_v_vals.append(np.mean(vs))
    ax2.plot(all_easy_ts, sd_v_vals, '--', color=COLOR_SD, label='$S_d$ (base)', linewidth=2)
    ax2.fill_between(all_easy_ts, [v - 0.05 for v in sd_v_vals], [v + 0.05 for v in sd_v_vals], color=COLOR_SD, alpha=0.1)
    apply_cosmetics(ax2, title="(b) Easy Regime — Violations by $\\tau'$ ", xlabel="Bomb Timer $T$", ylabel="Mean Violations")
    ax2.legend(framealpha=0.9, edgecolor='#CCCCCC', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'results/threshold_sweep.png'), dpi=300)
    print("\nPlot saved to results/threshold_sweep.png")

if __name__ == '__main__':
    main()
