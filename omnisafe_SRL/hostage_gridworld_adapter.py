
import torch
import numpy as np
import gymnasium
from gymnasium import spaces
from omnisafe.envs.core import CMDP, env_register
from hostage_gridworld import HostageGridworld

@env_register
class HostageGridworldAdapter(CMDP):
    _support_envs = ['HostageGridworld-v0', 'HostageGridworld-HighStakes-v0', 'HostageGridworld-Constrained-v0']
    
    def __init__(self, env_id, **kwargs):
        self._env_id = env_id
        layout_id = kwargs.get('layout_id', 1)
        
        if env_id == 'HostageGridworld-v0':
            # Low Stakes: Timer 50 (Generous)
            self._env = HostageGridworld(layout_id=layout_id, bomb_timer=50)
        elif env_id == 'HostageGridworld-HighStakes-v0':
            # High Stakes: Bomb (Default ~5 or passed)
            default_timer = HostageGridworld.GRID_LAYOUTS.get(layout_id, {}).get('bomb_timer', 20)
            passed_timer = kwargs.get('bomb_timer', default_timer)
            self._env = HostageGridworld(layout_id=layout_id, bomb_timer=passed_timer)
        elif env_id == 'HostageGridworld-Constrained-v0':
            # CONSTRAINED TRAINING: 
            # 1. bomb_timer=None: No time pressure.
            # 2. hard_constraint=True: Force detour learning.
            self._env = HostageGridworld(
                layout_id=layout_id, 
                bomb_timer=None, 
                hard_constraint=True
            )
        else:
            raise ValueError(f"Unknown env_id: {env_id}")
            
        # Expose Box action space to satisfy OmniSafe PPO (Gaussian learning)
        # Use 2D Box action space
        # We discretize based on signs of the 2 outputs (Quadrants)
        # This is robust to variance explosion (magnitude doesn't matter, only sign)
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self._observation_space = self._env.observation_space
        self._metadata = self._env.metadata
        
        self._num_envs = 1
        self.need_auto_reset_wrapper = True
        self.need_time_limit_wrapper = True

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            
        # Action is array of shape (2,)
        # Map to 4 discrete actions based on signs:
        # (+, +) -> 0 (Up)
        # (+, -) -> 1 (Down)
        # (-, +) -> 2 (Left)
        # (-, -) -> 3 (Right)
        
        val = action if isinstance(action, np.ndarray) else np.array(action)
        if val.shape != (2,):
            # Fallback for scalar (should not happen with Box(2,))
            # or if flattened
            if val.size == 2:
                val = val.flatten()
            else:
                 # Panic fallback
                 val = np.array([1.0, 1.0]) 

        s0 = 1 if val[0] >= 0 else 0
        s1 = 1 if val[1] >= 0 else 0
        
        # 0: (+, +) -> 1, 1 -> 3 (binary 11) -> 3 ? No.
        # Let's use simple logic
        if val[0] >= 0 and val[1] >= 0:
            idx = 0 # Up
        elif val[0] >= 0 and val[1] < 0:
            idx = 1 # Down
        elif val[0] < 0 and val[1] >= 0:
            idx = 2 # Left
        else:
            idx = 3 # Right (-, -)
            
        obs, reward, terminated, truncated, info = self._env.step(idx)
        
        cost = info.get('cost', 0.0)
        
        obs = torch.as_tensor(obs, dtype=torch.float32)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        cost = torch.as_tensor(cost, dtype=torch.float32)
        terminated = torch.as_tensor(terminated, dtype=torch.bool)
        truncated = torch.as_tensor(truncated, dtype=torch.bool)
        
        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        return obs, info

    def set_seed(self, seed):
        self._env.reset(seed=seed) 

    def render(self):
        return self._env.render()
        
    def close(self):
        self._env.close()

    @property
    def max_episode_steps(self) -> int:
        return 200
