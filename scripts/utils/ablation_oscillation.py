import os
import sys
import json
import torch
import numpy as np
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'omnisafe_SRL'))

from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.agents.ppo_actor import PPOActor
from cddp.critics.qr_critic import DualCritics
from cddp.agents.deontic_oracle import DeonticOraclePlanner
from cddp.shield.action_shield import ActionShield
from cddp.stakes.factory import build_stakes
from cddp.agents.cddp_agent import CDDPAgent
from cddp.eval.metrics import evaluate_agent

def count_oscillations(step_data):
    ep_modes = defaultdict(list)
    for step in step_data:
        ep_modes[step['episode']].append(step['mode'])
    
    oscillations = 0
    for ep, modes in ep_modes.items():
        if len(modes) < 3:
            continue
        simplified = []
        for m in modes:
            if not simplified or simplified[-1] != m:
                simplified.append(m)
        num_switches = len(simplified) - 1
        if num_switches >= 2:
            oscillations += 1
            
    return oscillations / len(ep_modes)

def analyze_oscillation_fixes():
    print("=== Grid 1 Oscillation Fix Ablations ===")
    
    actor_path = os.path.join(project_root, "checkpoints", "ppo_randomstart_slip005.pt")
    actor = PPOActor.from_checkpoint(actor_path, obs_dim=7)
    
    critics_path = os.path.join(project_root, "checkpoints", "dual_critics.pt")
    critics = DualCritics(obs_dim=10, n_quantiles=21, hidden_dim=128)
    critics.load_state_dict(torch.load(critics_path))
    
    oracle = DeonticOraclePlanner()
    shield = ActionShield()
    
    stakes_fn = build_stakes({'stakes': {'type': 'prospective'}}, critics=critics, oracle=oracle)
    
    variants = {
        'Prospective (Baseline)': None,
        'Prospective (Sticky)': 'sticky',
        'Prospective (Hysteresis)': 'hysteresis'
    }
    
    timers = [4, 5, 6, 7]
    n_episodes = 200
    
    for variant_name, fix_type in variants.items():
        print(f"\n--- Variant: {variant_name} ---")
        agent = CDDPAgent(actor, shield, stakes_fn, tau=0.5, oracle=oracle, util_mode='minharm',
                          oscillation_fix=fix_type, tau_lower=0.3)
        
        for T in timers:
            np.random.seed(42)
            torch.manual_seed(42)
            results, step_data = evaluate_agent(agent, layout_id=1, bomb_timer=T, n_episodes=n_episodes, random_start=True, p_slip=0.05)
            
            osc_rate = count_oscillations(step_data)
            success = results['feasible_success_rate']
            lateness = results.get('avg_lateness', np.nan)
            
            print(f"  T={T:2d} | Success: {success:.3f} | Oscillation Rate: {osc_rate:.3f} | Lateness: {lateness:.2f}")

if __name__ == "__main__":
    analyze_oscillation_fixes()
