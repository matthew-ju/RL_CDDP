"""
CDDP Experiment Runner — Compare CDDP vs PPO vs Constrained on all grids.

Usage:
    # Quick comparison (inference-only, no training)
    python -m cddp.eval.run_experiments --mode compare --grid 1

    # Full training + evaluation
    python -m cddp.eval.run_experiments --mode train --grid 1

    # All grids
    python -m cddp.eval.run_experiments --mode compare --grid all
"""

import os
import sys
import glob
import argparse
import json
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'omnisafe_SRL'))

from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.shield.action_shield import ActionShield
from cddp.stakes.analytical import AnalyticalStakesDmin
from cddp.agents.ppo_actor import PPOActor
from cddp.agents.cddp_agent import CDDPAgent
from cddp.eval.metrics import (evaluate_agent, evaluate_ppo_baseline,
                                evaluate_constrained_baseline,
                                print_comparison_table, save_results)
from cddp.plots.theory_plots import (plot_comparison_bars,
                                      plot_training_curves,
                                      plot_stakes_heatmap,
                                      plot_theorem_validation)


def find_best_checkpoint(log_dir):
    """Find the latest epoch checkpoint in a log directory."""
    runs = sorted(os.listdir(log_dir))
    if not runs:
        return None
    latest = runs[-1]
    ckpt_dir = os.path.join(log_dir, latest, 'torch_save')
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch-*.pt')),
                   key=lambda x: int(x.split('epoch-')[1].split('.')[0]))
    return ckpts[-1] if ckpts else None


def run_comparison(grid_id, bomb_timer, tau, n_episodes, save_dir):
    """Compare CDDP (oracle stakes, wrapping PPO) vs raw baselines."""
    print(f"\n{'='*60}")
    print(f"CDDP Comparison — Grid {grid_id}, T={bomb_timer}, τ={tau}")
    print(f"{'='*60}")

    omnisafe_dir = os.path.join(project_root, 'omnisafe_SRL')
    results = {}

    # ── Find PPO checkpoint ──
    ppo_env_name = 'HostageGridworld-HighStakes-v0'
    ppo_log = os.path.join(omnisafe_dir, 'omnisafe_logs',
                           f'PPO-{{{ppo_env_name}}}')
    ppo_ckpt = find_best_checkpoint(ppo_log) if os.path.exists(ppo_log) else None

    # ── Find Constrained checkpoint ──
    con_env_name = 'HostageGridworld-Constrained-v0'
    con_log = os.path.join(omnisafe_dir, 'omnisafe_logs_constrained',
                           f'PPOLag-{{{con_env_name}}}')
    con_ckpt = find_best_checkpoint(con_log) if os.path.exists(con_log) else None

    # ── Evaluate PPO baseline ──
    if ppo_ckpt:
        print(f"\n📊 Evaluating PPO baseline...")
        results['PPO'] = evaluate_ppo_baseline(
            grid_id, bomb_timer, ppo_ckpt, n_episodes)
    else:
        print("⚠ PPO checkpoint not found")

    # ── Evaluate Constrained baseline ──
    if con_ckpt:
        print(f"\n📊 Evaluating Constrained baseline...")
        results['Constrained'] = evaluate_constrained_baseline(
            grid_id, bomb_timer, con_ckpt, n_episodes)
    else:
        print("⚠ Constrained checkpoint not found")

    # ── Evaluate CDDP (oracle stakes wrapping PPO) ──
    if ppo_ckpt:
        print(f"\n📊 Evaluating CDDP (oracle, τ={tau})...")
        actor = PPOActor.from_checkpoint(ppo_ckpt)
        shield = ActionShield()
        stakes_fn = AnalyticalStakesDmin()
        agent = CDDPAgent(actor, shield, stakes_fn, tau=tau)

        cddp_results, step_data = evaluate_agent(
            agent, grid_id, bomb_timer, n_episodes)
        results[f'CDDP (τ={tau})'] = cddp_results

        # ── Also evaluate low stakes ──
        print(f"\n📊 Evaluating CDDP (low stakes)...")
        cddp_low, step_data_low = evaluate_agent(
            agent, grid_id, None, n_episodes)  # no bomb
        results[f'CDDP-low (τ={tau})'] = cddp_low

    # ── Print table ──
    print_comparison_table(results)

    # ── Save results ──
    os.makedirs(save_dir, exist_ok=True)
    save_results(results, os.path.join(save_dir, 'comparison.json'))

    # ── Plots ──
    if ppo_ckpt:
        plot_comparison_bars(
            {k: v for k, v in results.items() if 'low' not in k.lower()},
            os.path.join(save_dir, 'comparison_bars.png'),
            title=f'Grid {grid_id}, T={bomb_timer}')

        if step_data:
            plot_theorem_validation(
                step_data,
                os.path.join(save_dir, 'theorem_validation.png'))

        # Stakes heatmap
        env = InstrumentedHostageEnv.make(
            layout_id=grid_id, bomb_timer=bomb_timer)
        env.reset()
        plot_stakes_heatmap(
            env, AnalyticalStakesDmin(),
            os.path.join(save_dir, 'stakes_heatmap.png'),
            tau=tau)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDDP Experiment Runner')
    parser.add_argument('--mode', type=str, default='compare',
                        choices=['compare', 'train'])
    parser.add_argument('--grid', type=str, default='1',
                        help='Grid ID or "all"')
    parser.add_argument('--timer', type=int, default=5)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='cddp_results')
    args = parser.parse_args()

    grids = [1, 2, 3] if args.grid == 'all' else [int(args.grid)]

    for g in grids:
        grid_save = os.path.join(args.save_dir, f'grid{g}')

        if args.mode == 'compare':
            run_comparison(g, args.timer, args.tau, args.episodes, grid_save)

        elif args.mode == 'train':
            from cddp.trainers.full_cddp_trainer import FullCDDPTrainer
            trainer = FullCDDPTrainer(
                layout_id=g, bomb_timer=args.timer, tau=args.tau,
                total_epochs=300, save_dir=os.path.join(grid_save, 'training'),
            )
            trainer.train()
            plot_training_curves(
                os.path.join(grid_save, 'training', 'training_history.json'),
                os.path.join(grid_save, 'training_curves.png'))
            run_comparison(g, args.timer, args.tau, args.episodes, grid_save)
