
import omnisafe
import gymnasium
import torch
import sys
import os

# Add local path
sys.path.insert(0, os.getcwd())

from hostage_gridworld import HostageGridworld

class OmniSafeAdapter(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # OmniSafe requires these?
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = info.get('cost', 0.0)
        return obs, reward, cost, terminated, truncated, info
        
    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

def make_env():
    env = HostageGridworld()
    return OmniSafeAdapter(env)

def main():
    print(f"OmniSafe version: {omnisafe.__version__}")
    print(f"Gymnasium version: {gymnasium.__version__}")
    
    # Try to creating an agent
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 1024,
            'vector_env_nums': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': 1024,
            'update_iters': 1,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'use_tensorboard': False,
            'log_dir': './runs/test_omnisafe'
        }
    }

    try:
        agent = omnisafe.Agent(
            'PPO',
            env_id=make_env,
            custom_cfgs=custom_cfgs
        )
        print("Agent created successfully.")
        
        print("Starting training step...")
        agent.learn()
        print("Training step completed.")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
