
import os
import argparse
import omnisafe
import torch
import numpy as np

# Import the adapter module to register the environment automatically
import hostage_gridworld_adapter

def train(args):
    env_id = args.env
    
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
            'cost_limit': 0.0,
            'lagrangian_multiplier_init': 10.0,
            'lambda_lr': 0.02, 
            'lambda_optimizer': 'Adam',
        },
        'logger_cfgs': {
            'use_wandb': False,
            'log_dir': args.log_dir
        }
    }
    
    # Seed
    if args.seed is not None:
        custom_cfgs['seed'] = args.seed
    
    # Learning rate override
    if args.lr is not None:
        custom_cfgs['algo_cfgs']['actor_lr'] = args.lr
        custom_cfgs['algo_cfgs']['critic_lr'] = args.lr
    
    agent = omnisafe.Agent('PPOLag', env_id, custom_cfgs=custom_cfgs)
    agent.learn()
    
    print(f"Training finished. Logs in {custom_cfgs['logger_cfgs']['log_dir']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HostageGridworld-Constrained-v0')
    parser.add_argument('--total_steps', type=int, default=500000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--log_dir', type=str, default='./omnisafe_logs_constrained')
    
    args = parser.parse_args()
    train(args)
