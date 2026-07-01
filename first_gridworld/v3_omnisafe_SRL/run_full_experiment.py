"""
Full Experiment Runner — Simple Version

Trains PPO and Constrained agents on a specified grid layout,
then evaluates both agents on Low and High Stakes scenarios.

Usage:
  python3 run_full_experiment.py --grid 4 --steps 500000
  python3 run_full_experiment.py --grid 1 --steps 200000 --timer 5
"""

import os
import sys
import shutil
import subprocess
import glob
import argparse
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE = os.path.join(os.path.dirname(SCRIPT_DIR), 'results')


def find_best_checkpoint(log_dir, burn_in=50):
    """Find best checkpoint from OmniSafe training logs."""
    # OmniSafe creates subdirectories like PPO-{EnvID}/timestamp/
    sub_dirs = []
    for root, dirs, files in os.walk(log_dir):
        if 'progress.csv' in files:
            sub_dirs.append(root)
    
    if not sub_dirs:
        print(f"  Warning: No progress.csv found under {log_dir}")
        return None
    
    # Use the most recent
    latest = max(sub_dirs, key=os.path.getmtime)
    csv_path = os.path.join(latest, 'progress.csv')
    
    df = pd.read_csv(csv_path)
    
    torch_dir = os.path.join(latest, 'torch_save')
    if not os.path.exists(torch_dir):
        print(f"  Warning: No torch_save dir in {latest}")
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
    
    df_saved = df[df['Train/Epoch'].astype(int).isin(saved_epochs)]
    df_converged = df_saved[df_saved['Train/Epoch'] > burn_in]
    if df_converged.empty:
        df_converged = df_saved
    
    best_idx = df_converged['Metrics/EpRet'].idxmax()
    best_epoch = int(df_converged.loc[best_idx]['Train/Epoch'])
    best_ret = df_converged.loc[best_idx]['Metrics/EpRet']
    
    return {
        'epoch': best_epoch,
        'reward': best_ret,
        'path': os.path.join(torch_dir, f'epoch-{best_epoch}.pt'),
        'csv_path': csv_path
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate PPO + Constrained on a grid layout")
    parser.add_argument("--grid", type=int, default=1, help="Grid layout ID (1-6)")
    parser.add_argument("--steps", type=int, default=500000, help="Total training steps")
    parser.add_argument("--timer", type=int, default=None, 
                       help="Override bomb timer for high stakes (default: use layout's default)")
    args = parser.parse_args()
    
    grid_id = args.grid
    total_steps = args.steps
    
    # Output directory
    output_dir = os.path.join(RESULTS_BASE, f'grid_{grid_id}')
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Use the per-layout env ID so the adapter knows which layout to use
    # PPO trains on the layout with default bomb timer (unconstrained)
    ppo_env_id = f'HostageGridworld-Layout{grid_id}-v0'
    # Constrained trains with hard_constraint=True and no bomb timer
    constrained_env_id = f'HostageGridworld-Constrained-Layout{grid_id}-v0'
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT — Grid {grid_id}")
    print(f"  Steps: {total_steps}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")
    
    # ------------------------------------------------------------------
    # Step 1: Train PPO
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 1: Training PPO on Grid {grid_id}...")
    print(f"{'='*60}")
    
    ppo_log_dir = os.path.join(output_dir, 'training_logs', 'ppo')
    
    ppo_cmd = [
        sys.executable, 'train_omnisafe_ppo.py',
        '--env', ppo_env_id,
        '--total_steps', str(total_steps),
        '--log_dir', ppo_log_dir
    ]
    print(f"  Command: {' '.join(ppo_cmd)}")
    subprocess.run(ppo_cmd, cwd=SCRIPT_DIR)
    
    # Find and save best PPO checkpoint
    ppo_checkpoint = find_best_checkpoint(ppo_log_dir)
    ppo_model_path = None
    if ppo_checkpoint:
        ppo_model_path = os.path.join(models_dir, f'ppo_grid{grid_id}.pt')
        shutil.copy2(ppo_checkpoint['path'], ppo_model_path)
        print(f"  ✓ Best PPO model: epoch {ppo_checkpoint['epoch']}, "
              f"reward {ppo_checkpoint['reward']:.2f}")
        print(f"  ✓ Saved to: {ppo_model_path}")
    else:
        print("  ✗ No PPO checkpoint found!")
    
    # ------------------------------------------------------------------
    # Step 2: Train Constrained (PPO-Lag)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 2: Training Constrained (PPO-Lag) on Grid {grid_id}...")
    print(f"{'='*60}")
    
    constrained_log_dir = os.path.join(output_dir, 'training_logs', 'constrained')
    
    const_cmd = [
        sys.executable, 'train_omnisafe_constrained.py',
        '--env', constrained_env_id,
        '--total_steps', str(total_steps),
        '--log_dir', constrained_log_dir
    ]
    print(f"  Command: {' '.join(const_cmd)}")
    subprocess.run(const_cmd, cwd=SCRIPT_DIR)
    
    # Find and save best Constrained checkpoint
    const_checkpoint = find_best_checkpoint(constrained_log_dir)
    const_model_path = None
    if const_checkpoint:
        const_model_path = os.path.join(models_dir, f'constrained_grid{grid_id}.pt')
        shutil.copy2(const_checkpoint['path'], const_model_path)
        print(f"  ✓ Best Constrained model: epoch {const_checkpoint['epoch']}, "
              f"reward {const_checkpoint['reward']:.2f}")
        print(f"  ✓ Saved to: {const_model_path}")
    else:
        print("  ✗ No Constrained checkpoint found!")
    
    # ------------------------------------------------------------------
    # Step 3: Evaluate
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 3: Evaluating both agents on Grid {grid_id}...")
    print(f"{'='*60}")
    
    eval_cmd = [
        sys.executable, 'evaluate_baselines_omnisafe.py',
        '--grid', str(grid_id),
        '--models_dir', models_dir,
        '--output_dir', output_dir,
        '--num_seeds', '1',
        '--num_episodes', '100'
    ]
    print(f"  Command: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, cwd=SCRIPT_DIR)
    
    print(f"\n{'='*60}")
    print(f"DONE! Results in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
