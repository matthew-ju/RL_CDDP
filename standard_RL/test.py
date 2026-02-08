import os
import subprocess
import sys
from hostage_gridworld import HostageGridworld

def main():
    print("="*60)
    print("CDDP RL - AUTOMATED TESTING")
    print("="*60)
    
    print("Available Grids:")
    for gid, layout in HostageGridworld.GRID_LAYOUTS.items():
        print(f"  {gid}: {layout.get('name', f'Grid {gid}')} (Timer: {layout['bomb_timer']})")
    
    try:
        grid_input = input("\nEnter Grid ID to test (1-3): ").strip()
        grid_id = int(grid_input)
        if grid_id not in HostageGridworld.GRID_LAYOUTS:
            raise ValueError
    except ValueError:
        print("❌ Invalid Grid ID. Exiting.")
        return

    print(f"\n🚀 Starting workflow for Grid {grid_id}...")
    
    # 1. Train PPO
    print(f"\n[1/3] Training PPO (Utilitarian) on Grid {grid_id}...")
    cmd_ppo = [sys.executable, "train_ppo.py", "--grid", str(grid_id)]
    subprocess.check_call(cmd_ppo)
    
    # 2. Train Constrained
    print(f"\n[2/3] Training Constrained RL (Deontological) on Grid {grid_id}...")
    cmd_const = [sys.executable, "train_constrained.py", "--grid", str(grid_id)]
    subprocess.check_call(cmd_const)
    
    # 3. Evaluate
    print(f"\n[3/3] Evaluating Baselines on Grid {grid_id}...")
    cmd_eval = [sys.executable, "evaluate_baselines.py", "--grid", str(grid_id)]
    subprocess.check_call(cmd_eval)
    
    print("\n✅ WORKFLOW COMPLETE!")
    print(f"Check the generated visualizations: viz_*_grid{grid_id}.png")

if __name__ == "__main__":
    main()
