import os
import sys
import glob
import shutil
import torch
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'omnisafe_SRL'))

import omnisafe
import hostage_gridworld_adapter

def train():
    print("=== Grid 2 PPO-Baseline Finetuning ===")
    
    # We use the new explicit ID
    env_id = 'HostageGridworld-Layout2-Finetune-v0'
    
    log_dir = os.path.join(project_root, 'cddp_results', 'phase4_grid2_finetune_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    warm_start_path = os.path.join(project_root, "checkpoints", "ppo_randomstart_slip005.pt")
    
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 500000,
            'vector_env_nums': 1,
            'parallel': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': 2000,
            'update_iters': 10,
            'entropy_coef': 0.05,
            'target_kl': 0.02,
            'obs_normalize': False,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'log_dir': log_dir
        },
    }
    
    print(f"Loading weights from: {warm_start_path}")
    
    agent = omnisafe.Agent('PPO', env_id, custom_cfgs=custom_cfgs)
    
    try:
        checkpoint = torch.load(warm_start_path, map_location='cpu')
        
        if hasattr(agent, 'agent') and hasattr(agent.agent, 'actor_critic'):
            print("Attempting to load weights natively into actor/critic modules...")
            
            actor_state_dict = agent.agent.actor_critic.actor.state_dict()
            matched = 0
            for k in checkpoint.keys():
                if k in actor_state_dict and actor_state_dict[k].shape == checkpoint[k].shape:
                    actor_state_dict[k] = checkpoint[k]
                    matched += 1
            agent.agent.actor_critic.actor.load_state_dict(actor_state_dict)
            print(f"Transferred {matched} layers from actor checkpoint.")
            
    except Exception as e:
        print(f"Warning: partial warm-start failed, training from scratch ({e})")
        
    print("Beginning OmniSafe Finetuning on Grid 2 (T=13)...")
    agent.learn()
    
    print(f"Training finished. Logs in {log_dir}")
    
    base_save_dir = os.path.join(project_root, 'cddp_results', 'phase4_grid2_ppo')
    os.makedirs(base_save_dir, exist_ok=True)
    target_path = os.path.join(base_save_dir, 'ppo_grid2_finetuned.pt')
    
    search_pattern = os.path.join(log_dir, '**', 'torch_save', 'epoch-250.pt')
    found_models = glob.glob(search_pattern, recursive=True)
    
    if found_models:
        print(f"Copying {found_models[0]} to {target_path}")
        shutil.copy(found_models[0], target_path)
        
        act_ckpt = torch.load(target_path, map_location='cpu')
        flat_actor_path = os.path.join(base_save_dir, 'ppo_grid2_finetuned_actor.pt')
        torch.save(act_ckpt['pi'], flat_actor_path)
        print(f"Exported flat actor dict to {flat_actor_path}")
        
    else:
        print("Could not locate epoch-250.pt automatically.")

if __name__ == '__main__':
    train()
