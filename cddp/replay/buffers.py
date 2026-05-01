"""
Metadata-Rich Replay Buffers for CDDP.

Two buffers:
    ReplayBuffer: stores transitions from π_θ (utilitarian mode)
    DeonticReplayBuffer: stores transitions from π_d (deontic mode / warmup)

Each transition stores CDDP metadata for debugging and analysis:
    mode, T_rem, D_min, L_safe, in_failure_band, episode_id, step_idx
"""

import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class Transition:
    """A single transition with CDDP metadata."""
    obs: np.ndarray          # Current observation (7-dim)
    action: int              # Discrete action taken
    reward: float            # Environment reward
    next_obs: np.ndarray     # Next observation
    done: bool               # Episode terminated/truncated
    # CDDP metadata
    mode: str                # 'deontic' or 'utilitarian'
    T_rem: float             # Remaining bomb timer
    D_min: int               # Shortest unsafe path distance
    L_safe: float            # Shortest safe path distance
    min_v: int               # Minimum violations needed to rescue starting from current pos + timer
    in_failure_band: bool    # D_min < T_rem < L_safe
    next_in_failure_band: bool # if true after step is taken
    episode_id: int          # Episode identifier
    step_idx: int            # Step within episode
    next_safe_actions_mask: np.ndarray  # Mask of safe actions for next_obs (1 if safe, 0 if not)
    next_action_oracle: int = -1 # specific SARSA target action for Z_deon


class ReplayBuffer:
    """Fixed-size ring buffer storing Transitions.

    Used for off-policy critic updates (Z_free from π_θ transitions).

    Args:
        capacity: Maximum number of transitions
        obs_dim: Observation dimension (for pre-allocated arrays)
    """

    def __init__(self, capacity: int = 100_000, obs_dim: int = 9):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.pos = 0
        self.size = 0

        # Pre-allocate arrays for efficiency
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Metadata arrays
        self.modes = [''] * capacity
        self.T_rem = np.zeros(capacity, dtype=np.float32)
        self.D_min = np.zeros(capacity, dtype=np.int32)
        self.L_safe = np.zeros(capacity, dtype=np.float32)
        self.min_v = np.zeros(capacity, dtype=np.int32)
        self.in_failure_band = np.zeros(capacity, dtype=np.bool_)
        self.next_in_failure_band = np.zeros(capacity, dtype=np.bool_)
        self.episode_ids = np.zeros(capacity, dtype=np.int32)
        self.step_idxs = np.zeros(capacity, dtype=np.int32)
        self.next_safe_actions_mask = np.zeros((capacity, 4), dtype=np.float32)
        self.next_action_oracle = np.zeros(capacity, dtype=np.int64)

    def _pad_obs(self, obs: np.ndarray) -> np.ndarray:
        """Pad obs to obs_dim if shorter."""
        if obs.shape[-1] < self.obs_dim:
            padding = np.zeros(self.obs_dim - obs.shape[-1], dtype=np.float32)
            return np.concatenate([obs, padding])
        return obs

    def add(self, transition: Transition):
        """Add a transition to the buffer."""
        i = self.pos

        self.obs[i] = self._pad_obs(transition.obs)
        self.next_obs[i] = self._pad_obs(transition.next_obs)
        self.actions[i] = transition.action
        self.rewards[i] = transition.reward
        self.dones[i] = float(transition.done)

        self.modes[i] = transition.mode
        self.T_rem[i] = transition.T_rem if transition.T_rem != float('inf') else 999.0
        self.D_min[i] = transition.D_min
        self.L_safe[i] = transition.L_safe if transition.L_safe != float('inf') else 999.0
        self.min_v[i] = transition.min_v if transition.min_v != float('inf') else 999
        self.in_failure_band[i] = transition.in_failure_band
        self.next_in_failure_band[i] = transition.next_in_failure_band
        self.episode_ids[i] = transition.episode_id
        self.step_idxs[i] = transition.step_idx
        self.next_safe_actions_mask[i] = transition.next_safe_actions_mask
        self.next_action_oracle[i] = transition.next_action_oracle

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """Sample a random batch.

        Returns dict of torch tensors.
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        # Construct 10D critic obs (9D obs + normalized timer)
        t_rem_batch = self.T_rem[indices]
        t_feat = np.where(np.isinf(t_rem_batch), 1.0, np.clip(t_rem_batch / 25.0, 0.0, 1.0))
        t_feat_expand = np.expand_dims(t_feat, axis=-1)
        obs_critic = np.concatenate([self.obs[indices], t_feat_expand], axis=-1)
        
        # Proper decremented future timer for target bootstrapper
        t_rem_next_batch = t_rem_batch - 1.0
        t_rem_next_batch = np.where(t_rem_next_batch < 0, 0.0, t_rem_next_batch)
        t_feat_next = np.where(np.isinf(t_rem_batch), 1.0, np.clip(t_rem_next_batch / 25.0, 0.0, 1.0))
        t_feat_next_expand = np.expand_dims(t_feat_next, axis=-1)
        next_obs_critic = np.concatenate([self.next_obs[indices], t_feat_next_expand], axis=-1)

        return {
            'obs_actor': torch.FloatTensor(self.obs[indices]),
            'obs_critic': torch.FloatTensor(obs_critic),
            'next_obs_critic': torch.FloatTensor(next_obs_critic),
            'actions': torch.LongTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'dones': torch.FloatTensor(self.dones[indices]),
            'next_safe_actions_mask': torch.FloatTensor(self.next_safe_actions_mask[indices]),
            'next_action_oracle': torch.LongTensor(self.next_action_oracle[indices]),
            # Metadata (for analysis, not training)
            'modes': [self.modes[i] for i in indices],
            'T_rem': torch.FloatTensor(self.T_rem[indices]),
            'D_min': torch.IntTensor(self.D_min[indices]),
            'min_v': torch.IntTensor(self.min_v[indices]),
            'in_failure_band': torch.BoolTensor(self.in_failure_band[indices]),
            'next_in_failure_band': torch.BoolTensor(self.next_in_failure_band[indices]),
        }

    def __len__(self):
        return self.size


class DeonticReplayBuffer(ReplayBuffer):
    """Replay buffer specifically for deontic (π_d) transitions.

    Used for training Z_deon. Populated from:
        1. Deontic warmup rollouts (Phase 5)
        2. Deontic-mode steps during CDDP rollouts

    Same structure as ReplayBuffer; separated for logical clarity
    and to ensure Z_deon only trains on π_d data.
    """
    pass
