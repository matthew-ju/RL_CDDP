"""
networks.py
===========
Shared neural-network components for all CDDP-MuJoCo algorithms.

Architectures
-------------
  Actor            — deterministic policy π(s) → a  (used by DDPG / CDDP)
  UtilitarianCritic — Q(s, a) or Z_free(s, a)
  DeonticCritic    — state-only head Z_deon(s) → logit of P_cat(s)
  PPOActor         — stochastic Gaussian policy (used by PPO variants)
  PPOCritic        — state-value V(s)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(
    in_dim: int,
    out_dim: int,
    hidden: int = 256,
    n_hidden: int = 3,
    activation: type[nn.Module] = nn.ReLU,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    dims = [in_dim] + [hidden] * n_hidden + [out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# DDPG / CDDP
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Deterministic policy π_θ(s) → action ∈ [-act_limit, act_limit]."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        act_limit: float = 1.0,
        hidden: int = 256,
        n_hidden: int = 3,
    ) -> None:
        super().__init__()
        self.act_limit = float(act_limit)
        self.net = _mlp(state_dim, action_dim, hidden=hidden, n_hidden=n_hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.act_limit * torch.tanh(self.net(s))


class UtilitarianCritic(nn.Module):
    """Z_free(s, a): standard task-return critic Q(s, a)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        n_hidden: int = 3,
    ) -> None:
        super().__init__()
        self.net = _mlp(state_dim + action_dim, 1, hidden=hidden, n_hidden=n_hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


class DeonticCritic(nn.Module):
    """Z_deon(s): P_cat(s) under strict CBF compliance.

    Outputs raw logit; apply sigmoid externally for probability.
    Trained with BCEWithLogitsLoss for numerical stability.
    """

    def __init__(
        self,
        state_dim: int,
        hidden: int = 256,
        n_hidden: int = 3,
    ) -> None:
        super().__init__()
        self.net = _mlp(state_dim, 1, hidden=hidden, n_hidden=n_hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)   # raw logit; no sigmoid here


# ---------------------------------------------------------------------------
# PPO (stochastic Gaussian)
# ---------------------------------------------------------------------------

LOG_STD_MIN = -5.0
LOG_STD_MAX = 0.5


class PPOActor(nn.Module):
    """Gaussian policy: outputs (mean, log_std) for each action dimension."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        n_hidden: int = 3,
    ) -> None:
        super().__init__()
        self.shared = _mlp(state_dim, hidden, hidden=hidden, n_hidden=n_hidden - 1)
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = F.relu(self.shared(s))
        mean = torch.tanh(self.mean_head(feat))
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        return mean, log_std

    def get_dist(self, s: torch.Tensor) -> torch.distributions.Normal:
        mean, log_std = self(s)
        return torch.distributions.Normal(mean, log_std.exp())

    def act(
        self, s: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_dist(s)
        a = dist.mean if deterministic else dist.sample()
        a = a.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(a).sum(dim=-1, keepdim=True)
        return a, log_prob


class PPOCritic(nn.Module):
    """State-value function V(s) for PPO baseline."""

    def __init__(
        self,
        state_dim: int,
        hidden: int = 256,
        n_hidden: int = 3,
    ) -> None:
        super().__init__()
        self.net = _mlp(state_dim, 1, hidden=hidden, n_hidden=n_hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


# ---------------------------------------------------------------------------
# PPO-Lagrangian cost critic
# ---------------------------------------------------------------------------

class CostCritic(nn.Module):
    """V_C(s): expected cumulative constraint cost under current policy."""

    def __init__(
        self,
        state_dim: int,
        hidden: int = 256,
        n_hidden: int = 3,
    ) -> None:
        super().__init__()
        self.net = _mlp(state_dim, 1, hidden=hidden, n_hidden=n_hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


# ---------------------------------------------------------------------------
# Soft-update utility
# ---------------------------------------------------------------------------

def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: θ_target ← τ·θ_source + (1-τ)·θ_target."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)
