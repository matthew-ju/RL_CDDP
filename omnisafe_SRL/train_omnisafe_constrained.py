
import os
import argparse
import omnisafe
import torch
import numpy as np

# Import the adapter module to register the environment automatically
import hostage_gridworld_adapter

def train(args):
    # OmniSafe configuration for PPO-Lag (Constrained)
    
    env_id = args.env
    
    # Config
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': args.total_steps,
            'vector_env_nums': 1,
            'parallel': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': 1000,
            'update_iters': 10,
            'entropy_coef': 0.0, 
            'target_kl': 0.02,
            'obs_normalize': False,
        },
        'lagrange_cfgs': {
            'cost_limit': 0.0, # Zero tolerance for trampling
            'lagrangian_multiplier_init': 10.0, # Start with strong penalty
            'lambda_lr': 0.02, 
            'lambda_optimizer': 'Adam',
        },
        'logger_cfgs': {
            'use_wandb': False,
            'log_dir': './omnisafe_logs_constrained'
        }
    }
    
    # Initialize Agent
    agent = omnisafe.Agent('PPOLag', env_id, custom_cfgs=custom_cfgs)
    
    # Train
    agent.learn()
    
    print(f"Training finished. Logs in {custom_cfgs['logger_cfgs']['log_dir']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HostageGridworld-Constrained-v0', 
                        choices=['HostageGridworld-v0', 'HostageGridworld-HighStakes-v0', 'HostageGridworld-Constrained-v0'])
    parser.add_argument('--total_steps', type=int, default=500000)
    
    args = parser.parse_args()
    
    train(args)
