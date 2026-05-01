import os
import sys
import glob
import shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'omnisafe_SRL'))

import omnisafe
import torch
import hostage_gridworld_adapter

def train():
    env_id = 'HostageGridworld-Fair-v0'
    log_dir = os.path.join(project_root, 'cddp_results', 'phase4_fair_ppo_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 500000,
            'vector_env_nums': 1,
            'parallel': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': 1000,
            'update_iters': 10,
            'entropy_coef': 0.06,
            'target_kl': 0.02,
            'obs_normalize': False,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'log_dir': log_dir
        }
    }
    
    print("Starting fair PPO training...")
    agent = omnisafe.Agent('PPO', env_id, custom_cfgs=custom_cfgs)
    agent.learn()
    
    print(f"Training finished. Logs in {log_dir}")
    
    # Find the latest generated model and copy it to the target location
    base_save_dir = os.path.join(project_root, 'cddp_results', 'phase4_fair_ppo')
    os.makedirs(base_save_dir, exist_ok=True)
    target_path = os.path.join(base_save_dir, 'ppo_fair_start21_slip005.pt')
    
    # OmniSafe saves models inside log_dir/PPO/{env_id}/seed.../torch_save/
    # Find the epoch-500.pt
    search_pattern = os.path.join(log_dir, '**', 'torch_save', 'epoch-500.pt')
    found_models = glob.glob(search_pattern, recursive=True)
    
    if found_models:
        print(f"Copying {found_models[0]} to {target_path}")
        shutil.copy(found_models[0], target_path)
    else:
        print("Could not locate epoch-500.pt automatically. Please find the checkpoint in omnisafe logs and copy manually.")

if __name__ == '__main__':
    train()
