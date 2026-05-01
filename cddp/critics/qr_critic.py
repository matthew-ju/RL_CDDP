"""
QR-Based Distributional Critic — Quantile Regression for CDDP.

Estimates the full return distribution Z(s) as a set of N quantiles,
rather than just the expected value V(s). This enables:
    - Computing P_cat = P(Z ≤ threshold) for prospective stakes
    - Richer uncertainty-aware decision making

Two critics are trained:
    Z_free(s):  distribution of returns under π_θ (unshielded policy)
    Z_deon(s):  distribution of returns under π_d (permanently shielded policy)

Architecture: obs → Linear(hidden) → ReLU → Linear(hidden) → ReLU → Linear(N)
where N = number of quantile atoms (21 for dev, 51 for final).

Loss: Quantile Huber Loss (QR-DQN style, but for policy evaluation, not Q-learning)
"""

import numpy as np
import torch
import torch.nn as nn


class QRCritic(nn.Module):
    """Quantile Regression critic for distributional return estimation.

    Outputs N quantile values θ₁, ..., θ_N for the return distribution.
    Each θ_i ≈ F⁻¹(τ_i) where τ_i = (i - 0.5) / N (midpoint quantiles).

    Args:
        obs_dim: Observation dimension (9 for OmniSafe-compatible)
        n_quantiles: Number of quantile atoms
        hidden_dim: Hidden layer size
    """

    def __init__(self, obs_dim: int = 10, num_actions: int = 4, n_quantiles: int = 21,
                 hidden_dim: int = 128):
        super().__init__()

        self.n_quantiles = n_quantiles
        self.num_actions = num_actions

        # Midpoint quantile fractions: τ_i = (i - 0.5) / N
        taus = torch.FloatTensor([(i + 0.5) / n_quantiles
                                  for i in range(n_quantiles)])
        self.register_buffer('taus', taus)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions * n_quantiles),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: (batch, obs_dim) or (obs_dim,)

        Returns:
            quantiles: (batch, num_actions, N)
        """
        out = self.net(obs)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        batch_size = out.size(0)
        return out.view(batch_size, self.num_actions, self.n_quantiles)

    def get_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """V(s) ≈ mean of the quantile distribution.

        Args:
            obs: (batch, obs_dim) or (obs_dim,)

        Returns:
            mean: (batch, num_actions)
        """
        quantiles = self.forward(obs)
        return quantiles.mean(dim=-1)

    def get_cvar(self, obs: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """CVaR_α: expected value of the worst α-fraction of outcomes.
        Useful for risk-sensitive decision making.

        Args:
            obs: (batch, obs_dim)
            alpha: tail fraction (0.1 = worst 10%)

        Returns:
            cvar: (batch,)
        """
        quantiles = self.forward(obs)
        # Sort quantiles ascending
        sorted_q, _ = torch.sort(quantiles, dim=-1)
        k = max(1, int(alpha * self.n_quantiles))
        return sorted_q[:, :k].mean(dim=-1)

    def get_p_below(self, obs: torch.Tensor,
                    threshold: float) -> torch.Tensor:
        """P(Z ≤ threshold): fraction of quantiles below threshold.
        This is the key quantity for P_cat in prospective stakes.

        Args:
            obs: (batch, obs_dim)
            threshold: return threshold

        Returns:
            p_below: (batch, num_actions)
        """
        quantiles = self.forward(obs)
        below = (quantiles <= threshold).float()
        return below.mean(dim=-1)

    def quantile_huber_loss(self, quantiles: torch.Tensor,
                            targets: torch.Tensor,
                            kappa: float = 1.0) -> torch.Tensor:
        """Quantile Huber loss for distributional critic training.
        Computes pairwise TD error between all quantiles and targets.

        Args:
            quantiles: (batch, N) — predicted quantile values
            targets: (batch, N) — TD targets
            kappa: Huber threshold

        Returns:
            loss: scalar
        """
        # Pairwise TD errors: δ_ij = T_j - Q_i
        # targets: (batch, 1, N)
        # quantiles: (batch, N, 1)
        td_errors = targets.unsqueeze(1) - quantiles.unsqueeze(2)  # (batch, N, N)

        # Huber loss element
        abs_td = td_errors.abs()
        huber = torch.where(abs_td <= kappa,
                            0.5 * td_errors ** 2,
                            kappa * (abs_td - 0.5 * kappa))

        # Quantile weighting: |τ - I(δ < 0)|
        taus = self.taus.view(1, -1, 1)  # (1, N, 1)
        weight = torch.abs(taus - (td_errors.detach() < 0).float())

        # Average over targets (j), sum over quantiles (i), mean over batch
        loss = (weight * huber).mean(dim=2).sum(dim=1).mean()
        return loss


class DualCritics:
    """Manager for the two distributional critics Z_free and Z_deon.

    Z_free: evaluates returns under π_θ (unshielded utilitarian policy)
    Z_deon: evaluates returns under π_d (permanently shielded deontic policy)

    Args:
        obs_dim: Observation dimension
        n_quantiles: Number of quantile atoms
        hidden_dim: Hidden layer size
        lr: Learning rate
        target_update_freq: Steps between target network soft updates
        target_tau: Soft update coefficient (polyak averaging)
    """

    def __init__(self, obs_dim: int = 10, num_actions: int = 4, n_quantiles: int = 21,
                 hidden_dim: int = 128, lr: float = 1e-3,
                 target_update_freq: int = 1, target_tau: float = 0.005):

        self.n_quantiles = n_quantiles
        self.num_actions = num_actions

        # Online networks
        self.z_free = QRCritic(obs_dim, num_actions, n_quantiles, hidden_dim)
        self.z_deon = QRCritic(obs_dim, num_actions, n_quantiles, hidden_dim)

        # Target networks
        self.z_free_target = QRCritic(obs_dim, num_actions, n_quantiles, hidden_dim)
        self.z_deon_target = QRCritic(obs_dim, num_actions, n_quantiles, hidden_dim)

        # Initialize targets = online
        self.z_free_target.load_state_dict(self.z_free.state_dict())
        self.z_deon_target.load_state_dict(self.z_deon.state_dict())

        # Freeze target parameters
        for p in self.z_free_target.parameters():
            p.requires_grad = False
        for p in self.z_deon_target.parameters():
            p.requires_grad = False

        self.optimizer_free = torch.optim.Adam(self.z_free.parameters(), lr=lr)
        self.optimizer_deon = torch.optim.Adam(self.z_deon.parameters(), lr=lr)

        # Auxiliary Failure Predictor
        self.failure_predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Binary classification logit
        )
        self.optimizer_failure = torch.optim.Adam(self.failure_predictor.parameters(), lr=lr)
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.target_update_freq = target_update_freq
        self.target_tau = target_tau
        self._update_count = 0

    def update_z_free(self, obs_batch: torch.Tensor,
                      action_batch: torch.Tensor,
                      reward_batch: torch.Tensor,
                      next_obs_batch: torch.Tensor,
                      done_batch: torch.Tensor,
                      next_action_eval: torch.Tensor,
                      gamma: float = 0.99) -> float:
        """One gradient step on Z_free using SARSA from π_θ."""
        with torch.no_grad():
            next_quantiles = self.z_free_target(next_obs_batch) # (batch, a, N)
            
            best_next_actions = next_action_eval.unsqueeze(-1) # (batch, 1)
            idx = best_next_actions.unsqueeze(-1).expand(-1, -1, self.n_quantiles)
            target_quantiles = next_quantiles.gather(1, idx).squeeze(1) # (batch, N)
            
            done_batch_expand = done_batch.unsqueeze(1).expand_as(target_quantiles)
            reward_batch_expand = reward_batch.unsqueeze(1).expand_as(target_quantiles)
            target_values = reward_batch_expand + gamma * (1 - done_batch_expand) * target_quantiles

        quantiles_all = self.z_free(obs_batch)  # (batch, a, N)
        idx = action_batch.unsqueeze(1).unsqueeze(-1).expand(-1, 1, self.n_quantiles)
        quantiles = quantiles_all.gather(1, idx).squeeze(1) # (batch, N)
        
        loss = self.z_free.quantile_huber_loss(quantiles, target_values)

        self.optimizer_free.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.z_free.parameters(), 1.0)
        self.optimizer_free.step()

        self._maybe_update_targets()
        return loss.item()

    def update_z_deon(self, obs_batch: torch.Tensor,
                      action_batch: torch.Tensor,
                      reward_batch: torch.Tensor,
                      next_obs_batch: torch.Tensor,
                      done_batch: torch.Tensor,
                      next_action_eval: torch.Tensor,
                      gamma: float = 0.99) -> float:
        """One gradient step on Z_deon. QR-SARSA policy evaluation on π_d."""
        with torch.no_grad():
            next_quantiles = self.z_deon_target(next_obs_batch)
            
            # Policy Evaluation: strictly use the exact next action chosen by the strict policy
            best_next_actions = next_action_eval.unsqueeze(-1) # (batch, 1)
            
            idx = best_next_actions.unsqueeze(-1).expand(-1, -1, self.n_quantiles)
            target_quantiles = next_quantiles.gather(1, idx).squeeze(1)
            
            done_batch_expand = done_batch.unsqueeze(1).expand_as(target_quantiles)
            reward_batch_expand = reward_batch.unsqueeze(1).expand_as(target_quantiles)
            target_values = reward_batch_expand + gamma * (1 - done_batch_expand) * target_quantiles

        quantiles_all = self.z_deon(obs_batch)
        idx = action_batch.unsqueeze(1).unsqueeze(-1).expand(-1, 1, self.n_quantiles)
        quantiles = quantiles_all.gather(1, idx).squeeze(1)

        loss = self.z_deon.quantile_huber_loss(quantiles, target_values)

        self.optimizer_deon.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.z_deon.parameters(), 1.0)
        self.optimizer_deon.step()

        self._maybe_update_targets()
        return loss.item()

    def update_failure_predictor(self, obs_batch: torch.Tensor,
                                 target_batch: torch.Tensor) -> float:
        """One gradient step on the failure predictor."""
        logits = self.failure_predictor(obs_batch).squeeze(-1)
        loss = self.bce_loss(logits, target_batch.float())

        self.optimizer_failure.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.failure_predictor.parameters(), 1.0)
        self.optimizer_failure.step()

        return loss.item()

    def predict_failure(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict probability of being in/entering failure band.

        Args:
            obs: (batch, obs_dim) or (obs_dim,)

        Returns:
            prob: (batch,) or scalar
        """
        out = self.failure_predictor(obs).squeeze(-1)
        return torch.sigmoid(out)

    def _maybe_update_targets(self):
        """Soft-update target networks (polyak averaging)."""
        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self._soft_update(self.z_free, self.z_free_target)
            self._soft_update(self.z_deon, self.z_deon_target)

    def _soft_update(self, online, target):
        """θ_target = τ * θ_online + (1-τ) * θ_target"""
        for o, t in zip(online.parameters(), target.parameters()):
            t.data.copy_(self.target_tau * o.data + (1 - self.target_tau) * t.data)

    def get_urgency_gap(self, obs: torch.Tensor) -> torch.Tensor:
        """Δ_urg(s) = mean(Z_free(s)) - mean(Z_deon(s)).
        Positive → utilitarian is better than deontic.

        Args:
            obs: (batch, obs_dim) or (obs_dim,)

        Returns:
            urgency_gap: (batch,) or scalar
        """
        with torch.no_grad():
            v_free = self.z_free.get_mean(obs)
            v_deon = self.z_deon.get_mean(obs)
            return v_free - v_deon

    def state_dict(self):
        """Save all critic state dicts."""
        return {
            'z_free': self.z_free.state_dict(),
            'z_deon': self.z_deon.state_dict(),
            'z_free_target': self.z_free_target.state_dict(),
            'z_deon_target': self.z_deon_target.state_dict(),
            'failure_predictor': self.failure_predictor.state_dict(),
        }

    def load_state_dict(self, state):
        """Load all critic state dicts."""
        self.z_free.load_state_dict(state['z_free'])
        self.z_deon.load_state_dict(state['z_deon'])
        self.z_free_target.load_state_dict(state['z_free_target'])
        self.z_deon_target.load_state_dict(state['z_deon_target'])
        if 'failure_predictor' in state:
            self.failure_predictor.load_state_dict(state['failure_predictor'])
