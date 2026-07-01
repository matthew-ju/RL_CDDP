"""
cbf_layer.py
============
Analytical Control Barrier Function (CBF) projection for 2D planar robots.

System model
------------
The robot is treated as a 2D point mass with direct velocity control.
The CBF is defined relative to circular obstacles:

    h_i(p) = ||p - c_i||² - R_i²

Since the control input u = [vx, vy] directly updates position (first-order
kinematics for the purpose of the CBF), the relative degree is 1:

    ḣ_i = ∇h_i · u = 2(p - c_i) · u

CBF constraint (one per obstacle):

    ḣ_i + γ · h_i ≥ 0
  → 2(p - c_i) · u ≥ -γ · (||p - c_i||² - R_i²)

CDDP dynamic relaxation
-----------------------
The CBF boundary is expanded by d_soft(S):

    2(p - c_i) · u + γ · h_i + d_soft(S) ≥ 0

where d_soft(S) = (10/α') · log(1 + exp(α'·(S − τ))).

When S = 0  →  d_soft ≈ 0  (strict enforcement)
When S = 1  →  d_soft ≫ 0  (constraint relaxed: agent may enter crowd)

The active constraint is the MOST VIOLATED (smallest slack) obstacle.
An analytical closed-form QP projection is used (no solver required).
"""

from __future__ import annotations

import torch


def compute_d_soft(
    stakes: torch.Tensor,
    tau: float = 0.3,
    alpha_prime: float = 10.0,
    scale: float = 10.0,
) -> torch.Tensor:
    """CDDP C^∞ Softplus boundary expansion.

    d_soft(S) = scale · (1/α') · log(1 + exp(α'·(S − τ)))

    Parameters
    ----------
    stakes      : (B,) tensor, S ∈ [0, 1]
    tau         : relaxation threshold
    alpha_prime : softplus sharpness
    scale       : multiplier so that d_soft(1) fully overrides the CBF push

    Returns
    -------
    d_soft : (B,)
    """
    return scale * (1.0 / alpha_prime) * torch.log(
        1.0 + torch.exp(alpha_prime * (stakes - tau))
    )


def project_action_cbf(
    pos: torch.Tensor,
    u_nom: torch.Tensor,
    obstacle_centers: torch.Tensor,
    obstacle_radii: torch.Tensor,
    stakes: torch.Tensor | None = None,
    gamma: float = 1.0,
    tau: float = 0.3,
    alpha_prime: float = 10.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Project u_nom onto the CDDP-relaxed CBF-safe set (analytical QP).

    Handles multiple obstacles by projecting against the MOST VIOLATED one
    (the single active constraint in the 1-constraint QP approximation).

    Parameters
    ----------
    pos              : (B, 2) current robot XY positions
    u_nom            : (B, 2) nominal action (velocity command)
    obstacle_centers : (N, 2) obstacle centres
    obstacle_radii   : (N,)   obstacle radii
    stakes           : (B,)   S ∈ [0, 1]; None → strict mode (all zeros)
    gamma            : CBF class-K coefficient
    tau              : CDDP threshold
    alpha_prime      : softplus sharpness
    epsilon          : numerical floor for ||A||²

    Returns
    -------
    u_safe : (B, 2) projected (safe) velocity commands
    """
    B = pos.shape[0]
    device = pos.device
    dtype = pos.dtype

    if stakes is None:
        stakes = torch.zeros(B, dtype=dtype, device=device)

    centers = obstacle_centers.to(device=device, dtype=dtype)  # (N, 2)
    radii = obstacle_radii.to(device=device, dtype=dtype)       # (N,)

    # d_soft per environment instance
    d_soft = compute_d_soft(stakes, tau=tau, alpha_prime=alpha_prime)  # (B,)

    # Compute h_i(p) and the CBF gradient ∇h_i = 2(p - c_i) for all obstacles
    # pos: (B, 2) → (B, 1, 2); centers: (N, 2) → (1, N, 2)
    diff = pos.unsqueeze(1) - centers.unsqueeze(0)   # (B, N, 2)
    dist_sq = (diff ** 2).sum(dim=2)                  # (B, N)
    h_all = dist_sq - radii.unsqueeze(0) ** 2         # (B, N)
    grad_h = 2.0 * diff                               # (B, N, 2)  = ∇h_i

    # CBF constraint residual for each obstacle:
    #   slack_i = ∇h_i · u_nom + γ · h_i + d_soft  (CDDP version)
    u_nom_expanded = u_nom.unsqueeze(1).expand(-1, centers.shape[0], -1)  # (B, N, 2)
    d_soft_expanded = d_soft.unsqueeze(1).expand(-1, centers.shape[0])    # (B, N)
    slack_all = (
        (grad_h * u_nom_expanded).sum(dim=2)           # ∇h · u_nom  (B, N)
        + gamma * h_all                                # γ · h       (B, N)
        + d_soft_expanded                              # d_soft(S)   (B, N)
    )

    # Select the most-violated obstacle (smallest slack)
    active_idx = slack_all.argmin(dim=1)              # (B,)
    gather_idx = active_idx.view(B, 1, 1).expand(B, 1, 2)
    A = grad_h.gather(1, gather_idx).squeeze(1)       # (B, 2)   = ∇h_active
    slack = slack_all.gather(1, active_idx.unsqueeze(1)).squeeze(1)  # (B,)

    # Analytical QP projection:
    #   if slack >= 0: u* = u_nom (constraint already satisfied)
    #   else:  u* = u_nom - (slack / ||A||²) * A
    A_norm_sq = (A ** 2).sum(dim=1).clamp(min=epsilon)          # (B,)
    correction = (slack.clamp(max=0.0) / A_norm_sq).unsqueeze(1) * A  # (B, 2)
    u_safe = u_nom - correction

    return u_safe


def cbf_slack(
    pos: torch.Tensor,
    u: torch.Tensor,
    obstacle_centers: torch.Tensor,
    obstacle_radii: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Return the minimum CBF slack across all obstacles (B,).

    Positive = safe direction being maintained; negative = violation.
    """
    centers = obstacle_centers.to(device=pos.device, dtype=pos.dtype)
    radii = obstacle_radii.to(device=pos.device, dtype=pos.dtype)

    diff = pos.unsqueeze(1) - centers.unsqueeze(0)    # (B, N, 2)
    h_all = (diff ** 2).sum(dim=2) - radii.unsqueeze(0) ** 2   # (B, N)
    grad_h = 2.0 * diff                                         # (B, N, 2)

    u_exp = u.unsqueeze(1).expand(-1, centers.shape[0], -1)
    slack = (grad_h * u_exp).sum(dim=2) + gamma * h_all         # (B, N)
    return slack.min(dim=1).values                               # (B,)
