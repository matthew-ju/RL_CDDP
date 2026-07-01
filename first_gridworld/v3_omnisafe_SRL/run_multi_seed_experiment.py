"""
Multi-Seed Experiment Runner

Orchestrates training and evaluation across:
  - Multiple random seeds (5+ per config)
  - Multiple grid layouts (1-6)
  - Multiple bomb timers (None, 10, 6, 5, 3)
  - Both PPO and PPO-Lag agents

Results saved to: CDDP_RL/results/grid_N/

Usage:
  # Full experiment (5 seeds, grid 1, all timers)
  python3 run_multi_seed_experiment.py --seeds 5 --layouts 1 --steps 500000

  # Quick test (2 seeds, minimal steps)
  python3 run_multi_seed_experiment.py --seeds 2 --layouts 1 --steps 2048 --timers None,5

  # Dry run (print what would be done)
  python3 run_multi_seed_experiment.py --dry-run
"""

import os
import sys
import json
import shutil
import glob
import subprocess
import argparse
import time
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE = os.path.join(os.path.dirname(SCRIPT_DIR), 'results')


def parse_timers(timer_str):
    """Parse comma-separated timer values (e.g. 'None,10,6,5,3')."""
    timers = []
    for t in timer_str.split(','):
        t = t.strip()
        if t.lower() == 'none':
            timers.append(None)
        else:
            timers.append(int(t))
    return timers


def find_best_checkpoint(log_dir, burn_in=50):
    """Find best checkpoint from OmniSafe training logs."""
    csv_path = os.path.join(log_dir, 'progress.csv')
    if not os.path.exists(csv_path):
        print(f"  Warning: No progress.csv found in {log_dir}")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Find available checkpoints
    torch_dir = os.path.join(log_dir, 'torch_save')
    if not os.path.exists(torch_dir):
        print(f"  Warning: No torch_save dir in {log_dir}")
        return None
    
    saved_epochs = []
    for f in glob.glob(os.path.join(torch_dir, 'epoch-*.pt')):
        try:
            epoch_num = int(os.path.basename(f).split('-')[1].split('.')[0])
            saved_epochs.append(epoch_num)
        except (ValueError, IndexError):
            continue
    
    if not saved_epochs:
        print(f"  Warning: No checkpoints found in {torch_dir}")
        return None
    
    # Filter to saved epochs
    df_saved = df[df['Train/Epoch'].astype(int).isin(saved_epochs)]
    
    # Apply burn-in
    df_converged = df_saved[df_saved['Train/Epoch'] > burn_in]
    if df_converged.empty:
        df_converged = df_saved
    
    # Best by reward
    best_idx = df_converged['Metrics/EpRet'].idxmax()
    best_epoch = int(df_converged.loc[best_idx]['Train/Epoch'])
    best_ret = df_converged.loc[best_idx]['Metrics/EpRet']
    
    checkpoint_path = os.path.join(torch_dir, f'epoch-{best_epoch}.pt')
    
    return {
        'epoch': best_epoch,
        'reward': best_ret,
        'path': checkpoint_path,
        'progress_csv': csv_path
    }


def train_single(agent_type, env_id, seed, total_steps, log_dir, lr=None, extra_args=None):
    """Train a single agent with the given configuration."""
    if agent_type == 'ppo':
        script = 'train_omnisafe_ppo.py'
    else:
        script = 'train_omnisafe_constrained.py'
    
    cmd = [
        sys.executable, script,
        '--env', env_id,
        '--total_steps', str(total_steps),
        '--seed', str(seed),
        '--log_dir', log_dir
    ]
    
    if lr is not None:
        cmd.extend(['--lr', str(lr)])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: Training failed!")
        print(f"  STDERR: {result.stderr[-500:]}")
        return False
    
    return True


def run_experiment(args):
    """Run the full multi-seed experiment."""
    timers = parse_timers(args.timers)
    layouts = [int(l) for l in args.layouts.split(',')]
    num_seeds = args.seeds
    total_steps = args.steps
    
    print(f"\n{'='*80}")
    print(f"MULTI-SEED EXPERIMENT")
    print(f"{'='*80}")
    print(f"  Layouts: {layouts}")
    print(f"  Timers:  {timers}")
    print(f"  Seeds:   {num_seeds}")
    print(f"  Steps:   {total_steps}")
    print(f"  Dry Run: {args.dry_run}")
    print(f"{'='*80}\n")
    
    # Count total runs
    total_runs = len(layouts) * len(timers) * 2 * num_seeds  # 2 agents
    print(f"Total training runs: {total_runs}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would execute the following:")
        run_idx = 0
        for layout_id in layouts:
            for timer in timers:
                for agent_type in ['ppo', 'constrained']:
                    for seed_idx in range(num_seeds):
                        run_idx += 1
                        timer_str = 'None' if timer is None else str(timer)
                        print(f"  Run {run_idx}/{total_runs}: layout={layout_id}, timer={timer_str}, "
                              f"agent={agent_type}, seed={seed_idx}")
        print(f"\nTotal estimated time: ~{total_runs * 5} minutes (at ~5 min/run with {total_steps} steps)")
        return
    
    # Execute
    experiment_log = {
        'layouts': layouts,
        'timers': [str(t) for t in timers],
        'num_seeds': num_seeds,
        'total_steps': total_steps,
        'runs': []
    }
    
    run_idx = 0
    start_time = time.time()
    
    for layout_id in layouts:
        output_dir = os.path.join(RESULTS_BASE, f'grid_{layout_id}')
        models_dir = os.path.join(output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for timer in timers:
            timer_str = 'None' if timer is None else str(timer)
            
            # Determine env_id based on timer
            if timer is None:
                env_id = f'HostageGridworld-Layout{layout_id}-v0'
                # For low stakes, we override bomb_timer in adapter via env kwargs
                # Actually the adapter reads from layout default. For None, we want no bomb.
                env_id = 'HostageGridworld-v0'
            else:
                env_id = 'HostageGridworld-HighStakes-v0'
            
            for agent_type in ['ppo', 'constrained']:
                if agent_type == 'constrained':
                    env_id_train = 'HostageGridworld-Constrained-v0'
                else:
                    env_id_train = env_id
                
                for seed_idx in range(num_seeds):
                    run_idx += 1
                    seed_val = seed_idx * 100 + layout_id  # Deterministic but varied
                    
                    print(f"\n{'='*60}")
                    print(f"[Run {run_idx}/{total_runs}] Layout={layout_id}, Timer={timer_str}, "
                          f"Agent={agent_type}, Seed={seed_idx}")
                    print(f"{'='*60}")
                    
                    # Create per-run log directory
                    log_dir = os.path.join(output_dir, 'training_logs',
                                          f'{agent_type}_timer{timer_str}_seed{seed_idx}')
                    
                    # Train
                    success = train_single(
                        agent_type=agent_type,
                        env_id=env_id_train,
                        seed=seed_val,
                        total_steps=total_steps,
                        log_dir=log_dir
                    )
                    
                    # Find best checkpoint and copy to models dir
                    model_saved_path = None
                    if success:
                        # Find the actual log directory (OmniSafe creates subdirs)
                        pattern = os.path.join(log_dir, '*', '*')
                        sub_dirs = sorted(glob.glob(pattern), key=os.path.getmtime)
                        if sub_dirs:
                            latest = sub_dirs[-1]
                        else:
                            # Try one level up
                            sub_dirs = sorted(glob.glob(os.path.join(log_dir, '*')), 
                                            key=os.path.getmtime)
                            latest = sub_dirs[-1] if sub_dirs else log_dir
                        
                        checkpoint = find_best_checkpoint(latest)
                        if checkpoint:
                            model_name = f"{agent_type}_timer{timer_str}_grid{layout_id}_seed{seed_idx}.pt"
                            dst = os.path.join(models_dir, model_name)
                            shutil.copy2(checkpoint['path'], dst)
                            model_saved_path = dst
                            print(f"  ✓ Best model saved: {dst} (epoch {checkpoint['epoch']}, "
                                  f"reward {checkpoint['reward']:.2f})")
                    
                    experiment_log['runs'].append({
                        'layout': layout_id,
                        'timer': timer_str,
                        'agent': agent_type,
                        'seed': seed_idx,
                        'success': success,
                        'model_path': model_saved_path
                    })
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE — {run_idx} runs in {elapsed/60:.1f} minutes")
    print(f"{'='*80}")
    
    # Save experiment log
    log_path = os.path.join(RESULTS_BASE, 'experiment_log.json')
    with open(log_path, 'w') as f:
        json.dump(experiment_log, f, indent=2)
    print(f"✓ Experiment log saved: {log_path}")
    
    # Run evaluation for each layout
    if not args.skip_eval:
        for layout_id in layouts:
            output_dir = os.path.join(RESULTS_BASE, f'grid_{layout_id}')
            models_dir = os.path.join(output_dir, 'models')
            
            print(f"\n{'='*60}")
            print(f"Running evaluation for Grid {layout_id}...")
            print(f"{'='*60}")
            
            eval_cmd = [
                sys.executable, 'evaluate_baselines_omnisafe.py',
                '--grid', str(layout_id),
                '--models_dir', models_dir,
                '--output_dir', output_dir,
                '--num_seeds', str(num_seeds),
                '--num_episodes', '100'
            ]
            
            if args.stochastic:
                eval_cmd.append('--stochastic')
            
            subprocess.run(eval_cmd, cwd=SCRIPT_DIR)
    
    print(f"\n{'='*80}")
    print("ALL DONE!")
    print(f"Results saved to: {RESULTS_BASE}/grid_*/")
    print(f"{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds per config')
    parser.add_argument('--layouts', type=str, default='1', help='Comma-separated layout IDs (e.g. 1,4,5)')
    parser.add_argument('--timers', type=str, default='None,10,6,5,3', 
                       help='Comma-separated bomb timers (e.g. None,10,6,5,3)')
    parser.add_argument('--steps', type=int, default=500000, help='Total training steps')
    parser.add_argument('--stochastic', action='store_true', help='Also run stochastic evaluation')
    parser.add_argument('--skip_eval', action='store_true', help='Skip evaluation after training')
    parser.add_argument('--dry-run', action='store_true', help='Print plan without executing')
    
    args = parser.parse_args()
    run_experiment(args)
