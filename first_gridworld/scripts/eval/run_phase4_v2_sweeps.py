"""
Phase 4 v2 Final Sweep Script — True Oracle Upper Bound + Fixed Start States
============================================================================
Changes vs phase4_3:
  - PlannerOracleAgent replaces CDDPAgent+OracleStakes (true BFS/Dijkstra oracle)
  - Pre-sampled episode seeds per (seed, grid) shared across ALL timers and agents
    → L_safe(s0) is constant across timers (no confound)
  - L_safe per episode stored directly from env.reset(); no separate sampling block
  - Oracle regret metric: printed per timer and stored in JSON
  - Output: results/phase4_v2_grid{N}_sweeps.json
"""
import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

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
from cddp.agents.oracle_planner import PlannerOracleAgent
from cddp.eval.metrics import evaluate_agent
from scripts.utils.ablation_oscillation import count_oscillations


def presample_episode_seeds(n_seeds, n_episodes, grid_id, base_rng_seed=987654):
    """Pre-sample gym reset seeds for each (seed, episode) pair.

    The same seeds are reused across ALL timers and ALL agents so that
    start-state distributions are identical.

    Returns:
        dict: seed -> list[int] of length n_episodes (gym reset seeds)
    """
    rng = np.random.default_rng(base_rng_seed + grid_id * 1000)
    return {
        seed: rng.integers(0, 2**31, size=n_episodes).tolist()
        for seed in range(n_seeds)
    }


def oracle_sanity_check(agent, grid_id, T, oracle_ep_seeds, n_episodes=20, p_slip=0.05):
    """Run a small batch with the Oracle agent and print decision details."""
    from cddp.agents.oracle_planner import PlannerOracleAgent
    if not isinstance(agent, PlannerOracleAgent):
        return

    safe_feasible = util_count = total_success = total_viols = 0

    for ep_i in range(min(n_episodes, len(oracle_ep_seeds))):
        ep_seed = int(oracle_ep_seeds[ep_i])
        env = InstrumentedHostageEnv.make(layout_id=grid_id, bomb_timer=T,
                                           random_start=True, p_slip=p_slip)
        obs, _ = env.reset(seed=ep_seed)
        agent.reset_mode_tracking()

        d_min = env.get_D_min()
        t_rem = env.get_T_rem()
        if d_min > t_rem or t_rem <= 0:
            continue
        safe_feasible += 1

        _, act_info = agent.select_action(obs, env, deterministic=True)
        if act_info['mode'] == 'utilitarian':
            util_count += 1

        agent.reset_mode_tracking()
        done = False
        ep_steps = 0
        while not done and ep_steps < 50:
            act, _ = agent.select_action(obs, env, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
            ep_steps += 1

        total_viols += len(env.trampled_civilians)
        total_success += int(tuple(env.agent_pos) in
                             [hp for hp in env.hostage_positions])

    den = safe_feasible if safe_feasible > 0 else 1
    print(f"  [ORACLE SANITY G{grid_id} T={T}] "
          f"util_at_s0={util_count}/{den} | "
          f"success={total_success/den:.2f} | avg_viols={total_viols/den:.2f}")


def run_grid_sweep(grid_id, timers, agents, seeds, episodes_per_seed,
                   episode_seeds_map, output_path,
                   oracle_agent_name='CDDP-Oracle', sanity_timers=None,
                   p_slip=0.05):
    """Run a full grid sweep with fixed episode seeds.

    episode_seeds_map: dict seed -> list[int] of length episodes_per_seed
        Pre-sampled once outside the timer loop so all timers see identical
        start-state distributions.
    """
    print(f"\n=== Starting Grid {grid_id} Sweep ===")
    if os.path.exists(output_path):
        print(f"  [SKIP] Results already cached at {output_path}")
        print("  Delete the file to re-run.")
        return

    sweep_results = {}

    for agent_name, agent in agents.items():
        print(f"\n--- Model: {agent_name} ---")
        sweep_results[agent_name] = {}

        for T in tqdm(timers, desc=f"Evaluating {agent_name}"):
            # Oracle sanity for representative timers
            if (agent_name == oracle_agent_name
                    and sanity_timers is not None and T in sanity_timers):
                # Use seed-0 episode seeds for sanity
                oracle_sanity_check(agent, grid_id, T,
                                    episode_seeds_map[0])

            seed_metrics = {
                'feasible_success_rate': [],
                'infeasible_rate': [],
                'reward': [],
                'total_violations': [],
                'excess_violations': [],
                'excess_incidence': [],
                'oscillation_rate': [],
                'lateness': [],
                'earliness': [],
                'zero_violation_rate': [],
                'l_safe_s0': [],
            }
            per_seed_rows = []
            all_pcats = []

            for seed in seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)

                # Fixed episode seeds for this seed — shared across all T and agents
                ep_seeds = episode_seeds_map[seed]

                res, step_data = evaluate_agent(
                    agent, layout_id=grid_id, bomb_timer=T,
                    n_episodes=episodes_per_seed,
                    random_start=True, p_slip=p_slip,
                    episode_seeds=ep_seeds)

                seed_metrics['feasible_success_rate'].append(res.get('feasible_success_rate', 0))
                seed_metrics['infeasible_rate'].append(res.get('infeasible_rate', 0))
                seed_metrics['reward'].append(res.get('avg_reward', 0))
                seed_metrics['total_violations'].append(res.get('violation_rate', np.nan))
                seed_metrics['excess_violations'].append(res.get('excess_violation_rate', np.nan))
                seed_metrics['excess_incidence'].append(res.get('episodes_with_excess_rate', np.nan))

                osc_rate = count_oscillations(step_data) if 'CDDP' in agent_name else 0.0
                seed_metrics['oscillation_rate'].append(osc_rate)
                lat = res.get('avg_lateness', np.nan)
                ear = res.get('avg_earliness', np.nan)
                seed_metrics['lateness'].append(lat if not np.isnan(lat) else 0.0)
                seed_metrics['earliness'].append(ear if not np.isnan(ear) else 0.0)
                seed_metrics['zero_violation_rate'].append(
                    res.get('zero_violation_rate', res.get('safety_score', 0)))

                # Compute L_safe(s0) directly from per-episode data
                l_safe_vals = [v for v in res.get('l_safe_per_episode', [])
                               if isinstance(v, float) and not np.isnan(v)]
                l_safe_mean = float(np.mean(l_safe_vals)) if l_safe_vals else float('nan')
                seed_metrics['l_safe_s0'].append(l_safe_mean)

                per_seed_rows.append({
                    'feasible_success_rate':     res.get('feasible_success_rate', 0),
                    'violation_rate':            res.get('violation_rate', float('nan')),
                    'excess_violation_rate':     res.get('excess_violation_rate', float('nan')),
                    'episodes_with_excess_rate': res.get('episodes_with_excess_rate', float('nan')),
                    'oscillation_rate':          osc_rate,
                    'avg_reward':                res.get('avg_reward', 0),
                    'l_safe_s0':                 l_safe_mean,
                })

                # Sanity print for first combo
                if seed == seeds[0] and T == timers[0]:
                    vr  = res.get('violation_rate', 'MISSING')
                    zvr = res.get('zero_violation_rate', res.get('safety_score', 'MISSING'))
                    exr = res.get('excess_violation_rate', 'MISSING')
                    print(f"  [SANITY {agent_name} T={T}] "
                          f"viol={vr:.3f}  zero_viol={zvr:.3f}  excess={exr}")

                if 'start_states_data' in res:
                    for s in res['start_states_data']:
                        all_pcats.append(s['p_cat_hat'])

            # Aggregate across seeds
            agg_stats = {}
            for k, vals in seed_metrics.items():
                v_clean = [v for v in vals if isinstance(v, float) and not np.isnan(v)]
                agg_stats[f'mean_{k}'] = float(np.mean(v_clean)) if v_clean else float('nan')
                agg_stats[f'std_{k}']  = float(np.std(v_clean))  if v_clean else float('nan')
            agg_stats['seeds_data'] = per_seed_rows
            agg_stats['mean_pcat_s0'] = float(np.mean(all_pcats)) if all_pcats else 0.0

            # L_safe drift check print
            if agent_name == oracle_agent_name:
                l_safes = [r['l_safe_s0'] for r in per_seed_rows
                           if not np.isnan(r['l_safe_s0'])]
                if l_safes:
                    print(f"    [L_safe T={T}] mean={np.mean(l_safes):.2f} "
                          f"std={np.std(l_safes):.2f}")

            sweep_results[agent_name][str(T)] = agg_stats

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved Grid {grid_id} JSON results to {output_path}")


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=int, default=None)
    parser.add_argument('--seeds', type=int, default=15)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--T_max_only', action='store_true')
    parser.add_argument('--T', type=int, default=None)
    args = parser.parse_args()

    print(f"=== Phase 4 v2 Final Evaluations (Planning Oracle + Fixed Seeds) [grid={args.grid}, seeds={args.seeds}, ep={args.episodes}] ===")

    seeds         = list(range(args.seeds))
    ep_per_seed   = args.episodes

    def load_critic(path):
        if not os.path.exists(path):
            return None
        state = torch.load(path, map_location='cpu', weights_only=False)
        dim = 10
        if 'z_free.net.0.weight' in state:
            dim = state['z_free.net.0.weight'].shape[1]
        elif 'z_free' in state and 'net.0.weight' in state['z_free']:
            dim = state['z_free']['net.0.weight'].shape[1]
        c = DualCritics(obs_dim=dim, n_quantiles=21, hidden_dim=128)
        c.load_state_dict(state)
        c.z_free.eval()
        c.z_deon.eval()
        return c

    critics_path_g1 = os.path.join(project_root, "checkpoints", "grid1_dual_critics.pt")
    critics_g1 = load_critic(critics_path_g1)
    
    critics_path_g2 = os.path.join(project_root, "checkpoints", "grid2_dual_critics.pt")
    critics_g2 = load_critic(critics_path_g2)

    oracle_deontic = DeonticOraclePlanner()
    shield = ActionShield()
    pro_stakes_fn = build_stakes({'stakes': {'type': 'prospective'}},
                                  critics=critics_g1, oracle=oracle_deontic)

    # ── PPO actors ───────────────────────────────────────────────────────────
    ppo_base_path = os.path.join(
        project_root, "checkpoints", "ppo_randomstart_slip005.pt")
    ppo_g1ft_path = os.path.join(
        project_root, "checkpoints", "ppo_grid1_finetuned_actor.pt")
    ppo_g2ft_path = os.path.join(
        project_root, "checkpoints", "ppo_grid2_finetuned_actor.pt")

    actor_g1_ft = PPOActor.from_checkpoint(ppo_g1ft_path, obs_dim=7)
    actor_g2_ft = PPOActor.from_checkpoint(ppo_g2ft_path, obs_dim=7)

    class MockStakes:
        def __init__(self, fixed_val):
            self.fixed_val = fixed_val
            self.last_p_cat = fixed_val
        def compute(self, env):
            return self.fixed_val

    if args.grid is None or args.grid == 1:
        timers_g1  = list(range(3, 13))
        if args.T is not None: timers_g1 = [args.T]
        elif args.T_max_only: timers_g1 = [3, 7, 12]
        out_g1     = os.path.join(project_root, "results", "phase4_v2_grid1_sweeps_debug.json")
        sanity_g1  = [timers_g1[0], timers_g1[len(timers_g1) // 2], timers_g1[-1]]

        ag_ppo_ft_g1 = CDDPAgent(actor_g1_ft, shield, MockStakes(1.0), tau=0.3)
        ag_deontic   = CDDPAgent(actor_g1_ft, shield, MockStakes(0.0), tau=0.3)
        ag_cddp_ora  = PlannerOracleAgent()   # true BFS/Dijkstra oracle upper bound
        ag_cddp_pro  = CDDPAgent(actor_g1_ft, shield, pro_stakes_fn, tau=0.3)

        g1_models = {
            'PPO-Baseline-Finetuned':        ag_ppo_ft_g1,
            'Deontic-Oracle':                ag_deontic,
            'CDDP-Oracle':                   ag_cddp_ora,
            'CDDP-Prospective':              ag_cddp_pro,
        }

        ep_seeds_g1 = presample_episode_seeds(len(seeds), ep_per_seed, grid_id=1)
        print(f"\nPre-sampled episode seeds for Grid 1 (seeds 0..{len(seeds)-1}, {ep_per_seed} ep/seed)")

        run_grid_sweep(1, timers_g1, g1_models, seeds, ep_per_seed,
                       ep_seeds_g1, out_g1,
                       oracle_agent_name='CDDP-Oracle', sanity_timers=sanity_g1)

    if args.grid is None or args.grid == 2:
        timers_g2  = list(range(5, 21))
        if args.T is not None: timers_g2 = [args.T]
        elif args.T_max_only: timers_g2 = [max(timers_g2)]
        out_g2     = os.path.join(project_root, "results", "phase4_v2_grid2_sweeps.json")
        if args.T_max_only or args.T is not None or args.seeds < 15:
            out_g2 = os.path.join(project_root, "results", "phase4_v2_grid2_sweeps_DEBUG.json")

        sanity_g2  = [timers_g2[0], timers_g2[len(timers_g2) // 2], timers_g2[-1]]

        anal_stakes_fn2 = build_stakes({'stakes': {'type': 'analytical', 'analytic_mode': 'lsafe'}})
        pro_stakes_fn2 = build_stakes({'stakes': {'type': 'prospective'}}, critics=critics_g2, oracle=oracle_deontic)

        ag_ppo_ft_g2  = CDDPAgent(actor_g2_ft, shield, MockStakes(1.0), tau=0.3)
        ag_deontic2   = CDDPAgent(actor_g2_ft, shield, MockStakes(0.0), tau=0.3)
        ag_cddp_ora2  = PlannerOracleAgent()   
        ag_cddp_pro2  = CDDPAgent(actor_g2_ft, shield, pro_stakes_fn2, tau=0.3)
        
        g2_models = {
            'PPO-Baseline-Finetuned':        ag_ppo_ft_g2,
            'Deontic-Oracle':                ag_deontic2,
            'CDDP-Oracle':                   ag_cddp_ora2,
            'CDDP-Prospective':              ag_cddp_pro2,
        }

        ep_seeds_g2 = presample_episode_seeds(len(seeds), ep_per_seed, grid_id=2)
        print(f"\nPre-sampled episode seeds for Grid 2 (seeds 0..{len(seeds)-1}, {ep_per_seed} ep/seed)")

        run_grid_sweep(2, timers_g2, g2_models, seeds, ep_per_seed,
                       ep_seeds_g2, out_g2,
                       oracle_agent_name='CDDP-Oracle', sanity_timers=sanity_g2)


if __name__ == "__main__":
    main()
