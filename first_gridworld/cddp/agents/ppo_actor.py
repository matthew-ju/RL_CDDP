"""
PPOActor — Wrapper around OmniSafe-trained Gaussian actor.

OmniSafe trains a Gaussian policy over a 2D continuous action space:
    obs(9) → Linear(64) → Tanh → Linear(64) → Tanh → Linear(2) → μ
    + learned log_std(2)

The 2D continuous action is then discretized to grid movements:
    quadrant (x≥0,y≥0)→Up, (x≥0,y<0)→Down, (x<0,y≥0)→Left, (x<0,y<0)→Right

This module loads OmniSafe checkpoints and provides both:
    - get_continuous_action(): raw Gaussian sample (for OmniSafe compatibility)
    - get_discrete_action(): quadrant-discretized integer action (for CDDP shield)
"""

import numpy as np
import torch
import torch.nn as nn


# Quadrant discretization: continuous 2D → discrete direction
def discretize_action(continuous_action) -> int:
    """Convert 2D continuous action to discrete grid action.
    Matches the OmniSafe adapter convention:
        (x≥0, y≥0) → 0 (Up)
        (x≥0, y<0) → 1 (Down)
        (x<0, y≥0) → 2 (Left)
        (x<0, y<0) → 3 (Right)
    """
    if isinstance(continuous_action, torch.Tensor):
        val = continuous_action.detach().cpu().numpy()
    else:
        val = np.asarray(continuous_action)

    val = val.flatten()
    if val.size < 2:
        val = np.array([float(val), float(val)])

    if val[0] >= 0 and val[1] >= 0:
        return 0  # Up
    elif val[0] >= 0 and val[1] < 0:
        return 1  # Down
    elif val[0] < 0 and val[1] >= 0:
        return 2  # Left
    else:
        return 3  # Right


# Inverse: discrete action → representative continuous action (for shielded sampling)
DISCRETE_TO_CONTINUOUS = {
    0: np.array([1.0, 1.0]),   # Up
    1: np.array([1.0, -1.0]),  # Down
    2: np.array([-1.0, 1.0]),  # Left
    3: np.array([-1.0, -1.0]), # Right
}


class OmniSafeGaussianActor(nn.Module):
    """Gaussian actor matching OmniSafe's architecture.

    Architecture: obs → Linear(64) → Tanh → Linear(64) → Tanh → Linear(2)
    With learned log_std for the 2D action.

    This matches the checkpoint format: pi.mean.{0,2,4}.{weight,bias} + pi.log_std
    """

    def __init__(self, obs_dim: int = 9, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor):
        """Return action mean and log_std."""
        mean = self.mean(obs)
        return mean, self.log_std.expand_as(mean)

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action from Gaussian.

        Args:
            obs: (obs_dim,) or (batch, obs_dim)
            deterministic: if True, return mean; else sample

        Returns:
            action: (action_dim,) or (batch, action_dim)
        """
        mean, log_std = self.forward(obs)
        if deterministic:
            return mean
        std = log_std.exp()
        return mean + std * torch.randn_like(mean)


class PPOActor:
    """CDDP-compatible actor wrapping an OmniSafe-trained Gaussian policy.

    Provides:
        - get_action(obs): discrete action for grid environment
        - get_action_probs(obs): probability of each discrete action
        - raw Gaussian access for training

    The 9-dim observation is: [7 env dims] + [2 time-limit wrapper dims]
    OmniSafe adds the extra 2 dims for its TimeLimit wrapper.

    Args:
        obs_dim: 9 (7 env + 2 OmniSafe padding) — matches training
        hidden_dim: 64 — matches OmniSafe default
    """

    def __init__(self, obs_dim: int = 9, hidden_dim: int = 64):
        self.actor = OmniSafeGaussianActor(obs_dim=obs_dim, action_dim=2,
                                            hidden_dim=hidden_dim)
        self.obs_dim = obs_dim

    def _pad_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Pad 7-dim env obs to 9-dim for OmniSafe compatibility.
        The extra 2 dims are time-limit info: [time_elapsed/max_steps, 1-time_elapsed/max_steps]
        For inference, we zero-pad these (obs_normalize was disabled during training).
        """
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[-1] < self.obs_dim:
            padding = np.zeros(self.obs_dim - obs.shape[-1], dtype=np.float32)
            obs = np.concatenate([obs, padding])
        return torch.FloatTensor(obs)

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Get discrete grid action from observation.

        Args:
            obs: 7-dim numpy array from HostageGridworld
            deterministic: if True, use mean; else sample

        Returns:
            discrete action: int in {0, 1, 2, 3}
        """
        with torch.no_grad():
            obs_t = self._pad_obs(obs).unsqueeze(0)
            continuous = self.actor.sample(obs_t, deterministic=deterministic)
            return discretize_action(continuous.squeeze(0))

    def get_action_distribution(self, obs: np.ndarray):
        """Get the Gaussian distribution parameters for the observation.

        Returns:
            mean: (2,) numpy array
            std: (2,) numpy array
        """
        with torch.no_grad():
            obs_t = self._pad_obs(obs).unsqueeze(0)
            mean, log_std = self.actor.forward(obs_t)
            return mean.squeeze(0).numpy(), log_std.squeeze(0).exp().numpy()

    def get_discrete_probs(self, obs: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """Estimate discrete action probabilities via Monte Carlo sampling.
        This is needed because the Gaussian → quadrant mapping doesn't have
        a nice closed-form probability.

        Args:
            obs: 7-dim observation
            n_samples: number of Gaussian samples

        Returns:
            probs: (4,) array of probabilities for [up, down, left, right]
        """
        with torch.no_grad():
            obs_t = self._pad_obs(obs).unsqueeze(0)
            mean, log_std = self.actor.forward(obs_t)
            mean = mean.squeeze(0)
            std = log_std.squeeze(0).exp()

            samples = mean + std * torch.randn(n_samples, 2)
            counts = np.zeros(4)
            for s in samples:
                counts[discretize_action(s)] += 1
            return counts / n_samples

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, obs_dim: int = 9,
                        hidden_dim: int = 64):
        """Load from OmniSafe checkpoint (.pt file).

        Checkpoint format: {'pi': {state_dict_keys...}}

        Args:
            checkpoint_path: path to epoch-N.pt
            obs_dim: observation dimension (9 for standard OmniSafe training)
            hidden_dim: hidden layer size (64 for OmniSafe default)

        Returns:
            PPOActor with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu',
                                weights_only=False)

        state_dict = checkpoint['pi'] if 'pi' in checkpoint else checkpoint
        
        # Auto-detect obs_dim from weight shape
        if 'mean.0.weight' in state_dict:
            detected_obs_dim = state_dict['mean.0.weight'].shape[1]
            if detected_obs_dim != obs_dim:
                print(f"[INFO] Dimension mismatch detected. Checkpoint has obs_dim={detected_obs_dim}, current model has {obs_dim}. Re-initializing actor with {detected_obs_dim}...")
                obs_dim = detected_obs_dim

        actor_wrapper = cls(obs_dim=obs_dim, hidden_dim=hidden_dim)
        actor_wrapper.checkpoint_path = checkpoint_path
        if 'grid1' in checkpoint_path:
            actor_wrapper.grid_id = 1
        elif 'grid2' in checkpoint_path:
            actor_wrapper.grid_id = 2
        else:
            actor_wrapper.grid_id = None

        print(f"[DEBUG] CDDPProspective loaded: {checkpoint_path}, grid_id={actor_wrapper.grid_id}")

        actor_wrapper.actor.load_state_dict(state_dict)

        actor_wrapper.actor.eval()
        return actor_wrapper

    def __repr__(self):
        return f"PPOActor(obs_dim={self.obs_dim}, checkpoint=OmniSafe)"
