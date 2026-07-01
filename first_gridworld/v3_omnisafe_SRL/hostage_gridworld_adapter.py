
import torch
import numpy as np
import gymnasium
from gymnasium import spaces
from omnisafe.envs.core import CMDP, env_register
from hostage_gridworld import HostageGridworld

@env_register
class HostageGridworldAdapter(CMDP):
    _support_envs = [
        # Original env IDs (default to layout 1)
        'HostageGridworld-v0',
        'HostageGridworld-HighStakes-v0', 
        'HostageGridworld-Constrained-v0',
        # Per-layout env IDs (PPO / unconstrained training)
        'HostageGridworld-Layout1-v0',
        'HostageGridworld-Layout2-v0',
        'HostageGridworld-Layout3-v0',
        'HostageGridworld-Layout4-v0',
        'HostageGridworld-Layout5-v0',
        'HostageGridworld-Layout6-v0',
        # Per-layout CONSTRAINED env IDs (PPOLag training: hard_constraint=True, no bomb)
        'HostageGridworld-Constrained-Layout1-v0',
        'HostageGridworld-Constrained-Layout2-v0',
        'HostageGridworld-Constrained-Layout3-v0',
        'HostageGridworld-Constrained-Layout4-v0',
        'HostageGridworld-Constrained-Layout5-v0',
        'HostageGridworld-Constrained-Layout6-v0',
    ]
    
    def __init__(self, env_id, **kwargs):
        self._env_id = env_id
        layout_id = kwargs.get('layout_id', 1)
        bomb_timer = kwargs.get('bomb_timer', None)
        hard_constraint = kwargs.get('hard_constraint', False)
        use_reward_shaping = kwargs.get('use_reward_shaping', True)
        
        # Parse layout from env_id if specified
        if 'Layout' in env_id:
            # Extract layout number from e.g. 'HostageGridworld-Layout4-v0'
            # or 'HostageGridworld-Constrained-Layout4-v0'
            layout_id = int(env_id.split('Layout')[1].split('-')[0])
        
        # Determine bomb_timer and hard_constraint based on env_id
        if 'Constrained' in env_id:
            # CONSTRAINED TRAINING: no time pressure, civilians are walls
            # Works for both 'HostageGridworld-Constrained-v0' and
            # 'HostageGridworld-Constrained-Layout{N}-v0'
            bomb_timer = None
            hard_constraint = True
        elif env_id == 'HostageGridworld-v0':
            # Low Stakes: generous timer
            if bomb_timer is None:
                bomb_timer = 50
        elif env_id == 'HostageGridworld-HighStakes-v0':
            if bomb_timer is None:
                default_timer = HostageGridworld.GRID_LAYOUTS.get(layout_id, {}).get('bomb_timer', 20)
                bomb_timer = default_timer
        elif 'Layout' in env_id:
            # For Layout-specific IDs (unconstrained), use layout's default timer
            if bomb_timer is None:
                bomb_timer = HostageGridworld.GRID_LAYOUTS.get(layout_id, {}).get('bomb_timer', 10)
        
        self._env = HostageGridworld(
            layout_id=layout_id, 
            bomb_timer=bomb_timer,
            hard_constraint=hard_constraint,
            use_reward_shaping=use_reward_shaping
        )
            
        # 2D Box action space for OmniSafe PPO (Gaussian)
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._observation_space = self._env.observation_space
        self._metadata = self._env.metadata
        
        self._num_envs = 1
        self.need_auto_reset_wrapper = True
        self.need_time_limit_wrapper = True

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            
        val = action if isinstance(action, np.ndarray) else np.array(action)
        if val.shape != (2,):
            if val.size == 2:
                val = val.flatten()
            else:
                val = np.array([1.0, 1.0]) 

        if val[0] >= 0 and val[1] >= 0:
            idx = 0  # Up
        elif val[0] >= 0 and val[1] < 0:
            idx = 1  # Down
        elif val[0] < 0 and val[1] >= 0:
            idx = 2  # Left
        else:
            idx = 3  # Right

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
        return self._env.grid_size * 10
