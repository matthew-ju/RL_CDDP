import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'omnisafe_SRL'))

from hostage_gridworld import HostageGridworld
from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.agents.deontic_oracle import DeonticOraclePlanner
from cddp.critics.qr_critic import DualCritics

# Constants for cosmetics
COLOR_SD = '#769958'  # Unified Green
COLOR_ST = '#2D6FBC'  # Unified Blue
TAU = 0.3

def get_obs_tensor(env, actor_obs_dim=9):
    obs = env.env._get_obs()
    obs_np = np.array(obs, dtype=np.float32)
    if obs_np.shape[-1] < actor_obs_dim:
        obs_np = np.concatenate([obs_np, np.zeros(actor_obs_dim - obs_np.shape[-1], dtype=np.float32)])
    elif obs_np.shape[-1] > actor_obs_dim:
        obs_np = obs_np[:actor_obs_dim]

    timer = env.get_T_rem()
    T_NORM = 25.0
    t_feat = np.clip(timer / T_NORM, 0.0, 1.0)
    obs_critic_np = np.concatenate([obs_np, [t_feat]])
    return torch.FloatTensor(obs_critic_np).unsqueeze(0), timer

def bootstrap_ci(data, n_resamples=1000, ci=0.95):
    if len(data) == 0:
        return 0, 0, 0
    means = []
    for _ in range(n_resamples):
        resample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(resample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return np.mean(data), lower, upper

def get_lookahead_label(grid_id, current_state_info, oracle, k_steps=5):
    sim_env = InstrumentedHostageEnv.make(layout_id=grid_id, bomb_timer=current_state_info['bomb_timer_max'], random_start=False)
    sim_env.env.reset()
    sim_env.env.agent_pos = list(current_state_info['agent_pos'])
    sim_env.env.prev_agent_pos = list(current_state_info['agent_pos'])
    sim_env.env.bomb_time_left = current_state_info['bomb_time_left']
    sim_env.env.hostages_saved = set(current_state_info['hostages_saved'])
    sim_env.env.trampled_civilians = set(current_state_info['trampled_civilians'])
    sim_env.env.steps_taken = current_state_info['steps_taken']
    _, d = sim_env.env._nearest_unsaved_hostage(tuple(sim_env.env.agent_pos))
    sim_env.env.prev_distance = d

    for k in range(k_steps):
        pos = tuple(sim_env.agent_pos)
        t_rem = sim_env.get_T_rem()
        l_safe = sim_env.get_L_safe(pos)
        if l_safe > t_rem:
            return True
        action = oracle.get_action(sim_env, pos)
        _, _, terminated, truncated, _ = sim_env.step(action)
        if terminated or truncated:
            f_pos = tuple(sim_env.agent_pos)
            if sim_env.get_L_safe(f_pos) > sim_env.get_T_rem():
                return True
            break
    return False

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

def run_analysis():
    critics_dir = os.path.join(project_root, 'checkpoints')
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    grids = [1, 2]
    grid_timers = {1: [5, 6, 7], 2: [9, 10, 11]}
    n_episodes_per_timer = 50 
    
    all_results = []
    calibration_data = [] 
    
    for grid_id in grids:
        sd_path = os.path.join(critics_dir, f'grid{grid_id}_dual_critics.pt')
        st_path = os.path.join(critics_dir, f'grid{grid_id}_dual_critics_ppo_trained.pt')
        
        if not os.path.exists(sd_path) or not os.path.exists(st_path):
            print(f"[WARNING] Missing checkpoints for Grid {grid_id}. Skipping...")
            print(f"  Expected: {sd_path} and {st_path}")
            continue

        # Detect dimensions independently
        def detect_dim(state):
            if 'z_free.net.0.weight' in state:
                return state['z_free.net.0.weight'].shape[1]
            elif 'z_free' in state and 'net.0.weight' in state['z_free']:
                 return state['z_free']['net.0.weight'].shape[1]
            return 10

        sd_state = torch.load(sd_path, map_location='cpu', weights_only=False)
        sd_dim = detect_dim(sd_state)
        sd_critics = DualCritics(obs_dim=sd_dim, num_actions=4)
        sd_critics.load_state_dict(sd_state)
        
        st_state = torch.load(st_path, map_location='cpu', weights_only=False)
        st_dim = detect_dim(st_state)
        st_critics = DualCritics(obs_dim=st_dim, num_actions=4)
        st_critics.load_state_dict(st_state)
        
        print(f"[INFO] Grid {grid_id}: Isolated Dim={sd_dim}, Baseline Dim={st_dim}")
        for c in [sd_critics, st_critics]:
            c.z_free.eval(); c.z_deon.eval(); c.failure_predictor.eval()
        oracle = DeonticOraclePlanner()
        for timer_init in grid_timers[grid_id]:
            for ep in range(n_episodes_per_timer):
                env = InstrumentedHostageEnv.make(layout_id=grid_id, bomb_timer=timer_init, random_start=True, p_slip=0.0)
                obs, _ = env.reset(seed=ep * 777)
                done = False
                while not done:
                    pos = tuple(env.agent_pos); t_rem = env.get_T_rem(); l_safe = env.get_L_safe(pos); margin = t_rem - l_safe
                    
                    obs_th_sd, _ = get_obs_tensor(env, actor_obs_dim=sd_dim-1)
                    obs_th_st, _ = get_obs_tensor(env, actor_obs_dim=st_dim-1)
                    
                    with torch.no_grad():
                        s_d = torch.sigmoid(sd_critics.failure_predictor(obs_th_sd)).item()
                        s_t = torch.sigmoid(st_critics.failure_predictor(obs_th_st)).item()
                    all_results.append({'grid_id': grid_id, 'S_d': s_d, 'S_theta': s_t, 'margin': margin})
                    state_info = {
                        'agent_pos': pos, 'bomb_time_left': env.env.bomb_time_left, 'bomb_timer_max': env.env.bomb_timer_max,
                        'hostages_saved': list(env.env.hostages_saved), 'trampled_civilians': list(env.env.trampled_civilians),
                        'steps_taken': env.env.steps_taken
                    }
                    phi_5 = get_lookahead_label(grid_id, state_info, oracle, k_steps=5)
                    calibration_data.append({'grid_id': grid_id, 'S_d': s_d, 'S_theta': s_t, 'phi_5': phi_5})
                    action = oracle.get_action(env, pos)
                    _, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

    # Plotting Score Gap Analysis v4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bins = [(-float('inf'), -2), (-2, 0), (0, 2), (2, 5), (5, float('inf'))]
    bin_labels = ["<-2", "[-2, 0)", "[0, 2)", "[2, 5)", "5+"]
    sd_m, sd_lo, sd_hi = [], [], []
    st_m, st_lo, st_hi = [], [], []
    for low, high in bins:
        b = [r for r in all_results if low <= r['margin'] < high]
        if b:
            m, lo, hi = bootstrap_ci([r['S_d'] for r in b]); sd_m.append(m); sd_lo.append(lo); sd_hi.append(hi)
            m, lo, hi = bootstrap_ci([r['S_theta'] for r in b]); st_m.append(m); st_lo.append(lo); st_hi.append(hi)
        else:
            sd_m.append(0); sd_lo.append(0); sd_hi.append(0); st_m.append(0); st_lo.append(0); st_hi.append(0)
    ax1.plot(bin_labels, sd_m, label='$S_d$', color=COLOR_SD, marker='o', markersize=4)
    ax1.fill_between(bin_labels, sd_lo, sd_hi, color=COLOR_SD, alpha=0.15)
    ax1.plot(bin_labels, st_m, label='$S_\\theta$', color=COLOR_ST, marker='s', markersize=4)
    ax1.fill_between(bin_labels, st_lo, st_hi, color=COLOR_ST, alpha=0.15)
    apply_cosmetics(ax1, title="(a) Score Miscalibration", xlabel=r"Feasibility Margin ($T_{\mathrm{rem}} - L_{\mathrm{safe}}$)", ylabel=r"Mean Stakes Score $S(s)$")
    ax1.legend(framealpha=0.9, edgecolor='#CCCCCC', fontsize=9)
    
    # Plotting Empirical Classification Errors (Replacing Hardcoded Data)
    # Calculate False Negative Rate (Underestimating Danger) and False Positive Rate (Overestimating)
    # Ground truth: phi_5 == True means danger (needs S >= TAU)
    # Ground truth: phi_5 == False means safe (needs S < TAU)
    
    positives = [d for d in calibration_data if d['phi_5'] == True]
    negatives = [d for d in calibration_data if d['phi_5'] == False]
    
    sd_fnr = np.mean([d['S_d'] < TAU for d in positives]) if positives else 0
    st_fnr = np.mean([d['S_theta'] < TAU for d in positives]) if positives else 0
    
    sd_fpr = np.mean([d['S_d'] >= TAU for d in negatives]) if negatives else 0
    st_fpr = np.mean([d['S_theta'] >= TAU for d in negatives]) if negatives else 0

    regime_labels = ['Tight (FNR)', 'Easy (FPR)']
    sd_conseq = [sd_fnr, sd_fpr]
    st_conseq = [st_fnr, st_fpr]
    width = 0.35
    
    ax2.bar(np.arange(2)-width/2, sd_conseq, width, label='$S_d$', color=COLOR_SD, edgecolor='white', linewidth=0.8)
    ax2.bar(np.arange(2)+width/2, st_conseq, width, label='$S_\\theta$', color=COLOR_ST, edgecolor='white', linewidth=0.8)
    
    # Add value labels on top of bars
    for i, v in enumerate(sd_conseq):
        ax2.text(i - width/2, v + 0.02, f'{v*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for i, v in enumerate(st_conseq):
        ax2.text(i + width/2, v + 0.02, f'{v*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xticks(np.arange(2))
    ax2.set_xticklabels(regime_labels)
    ax2.set_ylim(0, 1.1)
    apply_cosmetics(ax2, title="(b) Estimator Misclassification Error (Internal Critic Failure)", ylabel="Error Rate (supervised prediction at $\\tau=0.3$, NOT RL behavior)")
    ax2.legend(framealpha=0.9, edgecolor='#CCCCCC', fontsize=9)
    plt.tight_layout()
    
    out_file = os.path.join(project_root, 'results/estimator_classification_error.png')
    plt.savefig(out_file, dpi=300)
    print(f"Score gap analysis plot saved to {out_file}")

if __name__ == '__main__':
    run_analysis()
