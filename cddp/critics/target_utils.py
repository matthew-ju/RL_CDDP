"""
Utility functions for distributional target computation.
"""
import torch

def compute_distributional_td_target(next_quantiles: torch.Tensor, 
                                     rewards: torch.Tensor, 
                                     dones: torch.Tensor, 
                                     gamma: float = 0.99) -> torch.Tensor:
    """
    Computes the 1-step TD target for a quantile regression critic.
    Unlike standard Q-learning where the target is a scalar max Q(s',a'),
    here we are typically doing policy evaluation (critic), so we use 
    the mean of the next state's quantiles or the next quantiles directly.
    
    Since DualCritics uses the mean for target stability (as per its `update_z_free`):
    target = r + gamma * (1 - done) * mean(Z_target(s'))
    
    Args:
        next_quantiles: (batch_size, N) tensor of next state quantiles
        rewards: (batch_size,) tensor of rewards
        dones: (batch_size,) boolean/float tensor (1 if terminal, 0 otherwise)
        gamma: discount factor
        
    Returns:
        target_values: (batch_size,) scalar targets to be expanded against quantiles
    """
    with torch.no_grad():
        target_v = rewards + gamma * (1.0 - dones) * next_quantiles.mean(dim=-1)
    return target_v

def compute_full_distributional_td_target(next_quantiles: torch.Tensor, 
                                          rewards: torch.Tensor, 
                                          dones: torch.Tensor, 
                                          gamma: float = 0.99) -> torch.Tensor:
    """
    Strict QR-DQN target where each quantile is shifted:
    T_theta_j = r + gamma * (1 - done) * theta_j(s')
    
    Returns:
        target_quantiles: (batch_size, N) tensor
    """
    with torch.no_grad():
        dones = dones.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        target_quantiles = rewards + gamma * (1.0 - dones) * next_quantiles
    return target_quantiles
