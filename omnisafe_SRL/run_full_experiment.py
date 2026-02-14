
import os
import shutil
import subprocess
import glob
import pandas as pd
import numpy as np
import argparse

def run_experiment(grid_id=1, total_steps=200000):
    env_id = "HostageGridworld-HighStakes-v0" if grid_id == 1 else "HostageGridworld-v0"
    # Wait, usually ID is consistent, just internal layout changes?
    # HostageGridworld logic uses layout_id.
    # But HostageGridworldAdapter hardcodes layout=1 for both IDs.
    # And HighStakes ID sets bomb_timer=20.
    # So if grid_id != 1, we might have issues if adapter doesn't support other layouts.
    # Adapter only supports layout=1 currently.
    # Let's assume grid_id=1 for now as per user request.
    if grid_id != 1:
        print("Warning: HostageGridworldAdapter currently hardcodes Layout 1. Other grids might not work as expected in training.")
        
    env_id = "HostageGridworld-HighStakes-v0"
    
    # ------------------------------------------------------------------
    # 1. Train Standard PPO (Baselines)
    # ------------------------------------------------------------------
    print(f"Training PPO (Standard) on {env_id}...")
    subprocess.check_call([
        "python3", "train_omnisafe_ppo.py",
        "--env", env_id,
        "--total_steps", str(total_steps)
    ])
    
    print("\n\n")
    print("="*60)
    print("Step 2: Selecting Best PPO Model...")
    print("="*60)
    
    # Find latest PPO log
    ppo_dirs = glob.glob(f"omnisafe_logs/PPO-{{{env_id}}}/*")
    if not ppo_dirs:
        raise FileNotFoundError("No PPO logs found!")
    
    latest_ppo_dir = max(ppo_dirs, key=os.path.getmtime)
    print(f"Analyzing logs in: {latest_ppo_dir}")
    
    # Read progress.csv
    csv_path = os.path.join(latest_ppo_dir, "progress.csv")
    df = pd.read_csv(csv_path)
    
    # Get available checkpoints
    # Get available checkpoints
    saved_epochs = [int(os.path.basename(f).split('-')[1].split('.')[0]) for f in glob.glob(os.path.join(latest_ppo_dir, "torch_save", "epoch-*.pt"))]
    if not saved_epochs:
        raise FileNotFoundError(f"No saved checkpoints found in {latest_ppo_dir}/torch_save")

    # Filter df to included epochs only
    df_saved = df[df['Train/Epoch'].astype(int).isin(saved_epochs)]
    
    # BURN-IN: Ignore first 50 epochs to avoid selecting random policies
    BURN_IN = 50
    df_converged = df_saved[df_saved['Train/Epoch'] > BURN_IN]
    
    if df_converged.empty:
        print("Warning: Not enough epochs for burn-in. Using all saved epochs.")
        df_converged = df_saved

    # Strategy: Maximize EpRet from converged models
    best_idx = df_converged['Metrics/EpRet'].idxmax()
    best_epoch = int(df_converged.loc[best_idx]['Train/Epoch'])
    best_ret = df_converged.loc[best_idx]['Metrics/EpRet']
    
    print(f"Best PPO Epoch (after burn-in): {best_epoch} (Return: {best_ret:.2f})")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)

    src = os.path.join(latest_ppo_dir, "torch_save", f"epoch-{best_epoch}.pt")
    dst = os.path.join("results", f"ppo_high_stakes_grid{grid_id}.pt")
    shutil.copy2(src, dst)
    print(f"Saved best PPO model to: {dst}")

    # ... (in step 4 block) ...

    src = os.path.join(latest_lag_dir, "torch_save", f"epoch-{best_epoch}.pt")
    dst = os.path.join("results", f"constrained_high_stakes_grid{grid_id}.pt")
    shutil.copy2(src, dst)
    print(f"Saved best Constrained model to: {dst}")

    
    print("\n\n")
    print("="*60)
    print("Step 5: Running Evaluation...")
    print("="*60)
    
    subprocess.check_call([
        "python3", "evaluate_baselines_omnisafe.py",
        "--grid", str(grid_id)
    ])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=1, help="Grid layout ID (1=HighStakes, 2=LowStakes)")
    parser.add_argument("--steps", type=int, default=500000, help="Total training steps")
    args = parser.parse_args()
    
    run_experiment(grid_id=args.grid, total_steps=args.steps)
