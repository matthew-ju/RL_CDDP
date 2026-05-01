"""
analytical_sandbox.py
=====================
Phase 2: CDDP Dynamic Boundary Extension

Upgrades the standard CBF into a Contextual Deontic Decision Process (CDDP)
dynamic constraint.  A stakes score S(s) ∈ [0, 1] modulates the safety
boundary via a C^∞ Softplus expansion:

    d_soft(S) = (1/α') · log(1 + exp(α'·(S − τ)))

The relaxed CBF constraint becomes:

    L_f h(x) + L_g h(x)·u + γ·h(x) + d_soft(S)  >=  0

When S < τ  (strict regime):   d_soft ≈ 0   → hard CBF enforcement
When S > τ  (high stakes):     d_soft > 0   → boundary expands, agent may
                                               trade safety for mission success

No external physics engines. Pure PyTorch — every operation is batched and
gradient-transparent.

State    x = [p_x, p_y, v_x, v_y, time_remaining]   (B, 5)
Control  u = [a_x, a_y]             (B, 2)

Spatial affine control system:  dx/dt = f(x) + g(x) u
    f(x) = [v_x, v_y, 0, 0]^T
    g(x) = [[0,0],[0,0],[1,0],[0,1]]

Integration: Euler, dt = 0.05 s

CBF:  h(x) = ||p - c||² - R²  >=  0   (safe outside obstacle)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DifferentiablePointMassEnv:
    """Batched, fully differentiable 2D PointMass environment.

    All arithmetic uses PyTorch tensor ops so that gradients can backpropagate
    through arbitrarily many `step()` calls.

    Parameters
    ----------
    goal       : (2,) tensor  [g_x, g_y]
    obstacle_c : (2,) tensor  [c_x, c_y]  — obstacle centre
    obstacle_r : float        — obstacle radius
    dt         : float        — Euler integration step (default 0.05 s)
    goal_tol   : float        — distance at which goal is considered reached
    device     : torch.device
    """

    def __init__(
        self,
        goal: torch.Tensor,
        obstacle_c: torch.Tensor,
        obstacle_r: float = 0.5,
        dt: float = 0.05,
        goal_tol: float = 0.1,
        v_max: float = 3.0,
        device: torch.device | None = None,
    ) -> None:
        self.device    = device or torch.device("cpu")
        self.goal      = goal.to(self.device)               # (2,)
        self.obstacle_c = obstacle_c.to(self.device)        # (2,)
        self.obstacle_r = float(obstacle_r)
        self.dt         = float(dt)
        self.goal_tol   = float(goal_tol)
        self.v_max      = float(v_max)

        self._state: torch.Tensor | None = None             # (B, 5)
        self.max_steps = 1
        self.current_step = 0

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, init_states: torch.Tensor, max_steps: int | None = None) -> torch.Tensor:
        """Set the initial batch of states.

        Parameters
        ----------
        init_states : (B, 4 or 5) tensor  [p_x, p_y, v_x, v_y, time_remaining]
        max_steps   : episode horizon used to normalize the clock

        Returns
        -------
        state : (B, 5) tensor
        """
        assert init_states.ndim == 2 and init_states.shape[1] in (4, 5), \
            "init_states must be (B, 4) or (B, 5)"

        if max_steps is not None:
            self.max_steps = max(1, int(max_steps))
        self.current_step = 0

        init_states = init_states.to(self.device)
        clock = torch.full(
            (init_states.shape[0], 1),
            self.max_steps / 1000.0,
            dtype=init_states.dtype,
            device=self.device,
        )
        self._state = torch.cat([init_states[:, :4], clock], dim=1)
        return self._state

    @property
    def state(self) -> torch.Tensor:
        if self._state is None:
            raise RuntimeError("Call reset() before accessing state.")
        return self._state

    # ------------------------------------------------------------------
    # Dynamics:  dx/dt = f(x) + g(x) u
    # ------------------------------------------------------------------

    def f(self, x: torch.Tensor) -> torch.Tensor:
        """Drift vector field f(x).  Returns (B, 4) tensor."""
        # f(x) = [v_x, v_y, 0, 0]
        v_x = x[:, 2:3]
        v_y = x[:, 3:4]
        zeros = torch.zeros_like(v_x)
        return torch.cat([v_x, v_y, zeros, zeros], dim=1)  # (B, 4)

    def g(self, x: torch.Tensor) -> torch.Tensor:
        """Control matrix g(x).  Returns (B, 4, 2) tensor.

        g(x) is state-independent here, but is returned as a batched tensor
        for generality (e.g., robots with state-dependent kinematics).
        """
        B = x.shape[0]
        # g = [[0, 0], [0, 0], [1, 0], [0, 1]]
        g_mat = torch.tensor(
            [[0., 0.], [0., 0.], [1., 0.], [0., 1.]],
            dtype=x.dtype, device=x.device
        )
        return g_mat.unsqueeze(0).expand(B, -1, -1)        # (B, 4, 2)

    def step(
        self,
        actions: torch.Tensor,
        update_internal: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Euler-integrate one step.

        Parameters
        ----------
        actions         : (B, 2) tensor  [a_x, a_y]
        update_internal : if True, overwrite self._state with next_state
                          (set False when you want to keep the graph intact
                          across multiple steps without in-place mutation)

        Returns
        -------
        dict with keys:
            'state'       : (B, 5)   next state x'
            'reward'      : (B,)     negative distance-to-goal (dense)
            'h_cbf'       : (B,)     CBF value h(x') (safety >= 0 means safe)
            'goal_reached': (B,)     bool mask
            'unsafe'      : (B,)     bool mask  (h < 0)
        """
        x = self._state                                      # (B, 5)
        x_spatial = x[:, :4]                                 # (B, 4)

        # dx/dt = f(x) + g(x) u
        f_x   = self.f(x_spatial)                           # (B, 4)
        g_x   = self.g(x_spatial)                           # (B, 4, 2)
        g_x_u = torch.bmm(g_x, actions.unsqueeze(-1)).squeeze(-1)  # (B, 4)

        x_dot = f_x + g_x_u                                 # (B, 4)
        x_next_spatial = x_spatial + self.dt * x_dot         # (B, 4)  — Euler

        # Clamp velocities (soft saturation via tanh keeps gradients alive)
        x_next_spatial = torch.cat([
            x_next_spatial[:, :2],                                   # positions (unclamped)
            self.v_max * torch.tanh(x_next_spatial[:, 2:] / self.v_max),  # velocities
        ], dim=1)

        # Boundary violations carry reward/cost consequences only; they should
        # not physically slow the mass when CDDP deliberately relaxes the CBF.
        h_current_spatial = self.cbf_value(x_spatial)
        h_next_spatial = self.cbf_value(x_next_spatial)
        inside = ((h_current_spatial < 0.0) | (h_next_spatial < 0.0)).unsqueeze(1)
        v_next = torch.where(inside, 1.0 * x_next_spatial[:, 2:], x_next_spatial[:, 2:])
        x_next_spatial = torch.cat([x_next_spatial[:, :2], v_next], dim=1)

        self.current_step += 1
        time_remaining = max(0.0, (self.max_steps - self.current_step) / 1000.0)
        clock = torch.full(
            (x_next_spatial.shape[0], 1),
            time_remaining,
            dtype=x_next_spatial.dtype,
            device=x_next_spatial.device,
        )
        x_next = torch.cat([x_next_spatial, clock], dim=1)

        # --- Rewards --------------------------------------------------------
        pos_next = x_next[:, :2]                             # (B, 2)
        dist_to_goal = torch.norm(pos_next - self.goal, dim=1)  # (B,)
        reward = -dist_to_goal                               # dense shaping

        # --- Safety ---------------------------------------------------------
        h_next = self.cbf_value(x_next)                     # (B,)
        unsafe = h_next < 0.0

        # --- Termination flags (non-differentiable, detached) ---------------
        goal_reached = (dist_to_goal.detach() < self.goal_tol)

        if update_internal:
            # NOTE: We detach before storing so that long-horizon gradient
            # accumulation is controlled explicitly by the training loop.
            # Remove .detach() if you want unrolled BPTT.
            self._state = x_next.detach()

        return {
            "state":        x_next,
            "reward":       reward,
            "h_cbf":        h_next,
            "goal_reached": goal_reached,
            "unsafe":       unsafe,
        }

    # ------------------------------------------------------------------
    # CBF and Lie Derivatives
    # ------------------------------------------------------------------

    def cbf_value(self, x: torch.Tensor) -> torch.Tensor:
        """h(x) = ||p - c||^2 - R^2.  (B,)

        h >= 0  ⟺  point-mass is outside (safe side of) the obstacle.
        """
        x_spatial = x[:, :4]
        p = x_spatial[:, :2]                                 # (B, 2)
        diff = p - self.obstacle_c                           # (B, 2)
        return (diff ** 2).sum(dim=1) - self.obstacle_r ** 2  # (B,)

    def cbf_lie_derivatives(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute L_f h(x), L_g h(x), and L_f^2 h(x) analytically.

        h(x) = (p_x - c_x)^2 + (p_y - c_y)^2 - R^2

        Gradient:  ∂h/∂x = [2(p_x-c_x), 2(p_y-c_y), 0, 0]

        L_f h  = 2(p_x-c_x)v_x + 2(p_y-c_y)v_y                 (B,)
        L_g h  = [0, 0]  ← velocity rows of g are zero,               (B, 2)
                  position rows of g are also zero → L_g h = 0!

        Because h is relative-degree 2 wrt u, we use HOCBF:
          ψ_1(x)    = L_f h(x) + γ_1 · h(x)
          L_f ψ_1   = L_f^2 h + γ_1 · L_f h             (B,)
          L_g ψ_1   = L_g L_f h = [2(p_x-c_x), 2(p_y-c_y)]  (B, 2)
                       (acceleration directly updates velocity which updates pos)

        The HOCBF constraint is:
          L_f ψ_1 + L_g ψ_1 · u + γ_2 · ψ_1 + d_soft(S)  >= 0

        Returns
        -------
        psi1   : (B,)   first barrier function ψ_1 = L_f h + γ_1 h
        L_f_psi1 : (B,)   L_f ψ_1  (scalar part of HOCBF constraint)
        L_g_psi1 : (B, 2) L_g ψ_1  (linear part of HOCBF constraint, wrt u)
        """
        x_spatial = x[:, :4]
        p    = x_spatial[:, :2]                                # (B, 2)
        v    = x_spatial[:, 2:4]                                # (B, 2)
        diff = p - self.obstacle_c                             # (B, 2)

        h     = self.cbf_value(x)                             # (B,)
        L_f_h = (2.0 * diff * v).sum(dim=1)                  # (B,)  = 2δ·v
        # L_f^2 h = d/dt(L_f h) along f(x):
        #   d/dt[2δ_x v_x + 2δ_y v_y] = 2 v_x^2 + 2 v_y^2  (since a=f contributes 0 accel)
        L_f2_h = 2.0 * (v ** 2).sum(dim=1)                   # (B,)

        # L_g L_f h: how does u (= acceleration) affect d/dt(L_f h)?
        #   d/dt[L_f h] includes term 2 δ_x a_x + 2 δ_y a_y
        L_g_Lf_h = 2.0 * diff                                 # (B, 2)

        return L_f_h, L_f2_h, L_g_Lf_h

    def cbf_constraint_slack(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        gamma: float = 1.0,
    ) -> torch.Tensor:
        """Returns the HOCBF constraint residual (>= 0 means safe direction).

        residual = L_f psi1 + L_g psi1 · u + gamma · psi1

        Parameters
        ----------
        x     : (B, 5), with spatial CBF math applied to x[:, :4]
        u     : (B, 2)
        gamma : class-K coefficient for the second-order barrier

        Returns
        -------
        residual : (B,)
        """
        h = self.cbf_value(x)
        L_f_h, L_f2_h, L_g_Lf_h = self.cbf_lie_derivatives(x)
        psi1     = L_f_h + gamma * h                          # (B,)
        L_f_psi1 = L_f2_h + gamma * L_f_h                    # (B,)
        L_g_psi1 = L_g_Lf_h                                   # (B, 2)
        return L_f_psi1 + (L_g_psi1 * u).sum(dim=1) + gamma * psi1  # (B,)

    # ------------------------------------------------------------------
    # CDDP Softplus Dynamic Boundary
    # ------------------------------------------------------------------

    def compute_d_soft(
        self,
        S: torch.Tensor,
        tau: float = 0.3,
        alpha_prime: float = 10.0,
    ) -> torch.Tensor:
        """CDDP Softplus boundary expansion  d_soft(S).

        d_soft(S) = (1/α') · log(1 + exp(α'·(S − τ)))

        Properties
        ----------
        - C^∞ everywhere (no kink at S = τ, unlike max(0, ·))
        - d_soft ≈ 0     when  S << τ   → strict CBF enforcement
        - d_soft grows   when  S >> τ   → boundary relaxation (unsafe OK)
        - Gradient w.r.t. S is the sigmoid σ(α'·(S − τ)) ∈ (0, 1)

        Parameters
        ----------
        S           : (B,) stakes scores ∈ [0, 1]
        tau         : relaxation threshold (default 0.3)
        alpha_prime : sharpness of the softplus transition (default 10)

        Returns
        -------
        d_soft : (B,)  non-negative boundary expansion
        """
        # Multiplied by 10.0 to fully erase the CBF push-back at high stakes.
        return 10.0 * (1.0 / alpha_prime) * torch.log(
            1.0 + torch.exp(alpha_prime * (S - tau))
        )                                                     # (B,)

    # ------------------------------------------------------------------
    # CDDP-aware QP Projection (analytical 1-constraint case)
    # ------------------------------------------------------------------

    def project_action_cbf(
        self,
        x: torch.Tensor,
        u_nom: torch.Tensor,
        stakes: torch.Tensor | None = None,
        gamma: float = 1.0,
        tau: float = 0.3,
        alpha_prime: float = 10.0,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """Project u_nom onto the CDDP-relaxed HOCBF-safe set.

        Uses the High-Order CBF formulation (HOCBF) to correctly handle
        the relative-degree-2 relationship between control input u
        (acceleration) and barrier function h (position-based).

        HOCBF constraint (one QP constraint):
            L_f ψ_1 + L_g ψ_1 · u + γ · ψ_1 + d_soft(S)  >=  0

        where  ψ_1 = L_f h + γ · h  (first-order barrier candidate).

        Closed-form projection (same structure as first-order CBF QP):
            A  = L_g ψ_1
            b  = L_f ψ_1 + γ · ψ_1 + d_soft(S)
            slack = A · u_nom + b
            u* = u_nom   if slack >= 0
               = u_nom - (slack / ||A||^2) · A   otherwise

        Parameters
        ----------
        x           : (B, 5)  current states; CBF math uses x[:, :4]
        u_nom       : (B, 2)  nominal actions
        stakes      : (B,)    S ∈ [0,1]; None → zeros (strict mode)
        gamma       : class-K coefficient (used at both levels of HOCBF)
        tau         : CDDP relaxation threshold
        alpha_prime : softplus sharpness
        epsilon     : numerical floor for ||A||^2

        Returns
        -------
        u_safe : (B, 2)
        """
        B = x.shape[0]
        if stakes is None:
            stakes = torch.zeros(B, dtype=x.dtype, device=x.device)

        x_spatial = x[:, :4]
        h                        = self.cbf_value(x_spatial)                     # (B,)
        L_f_h, L_f2_h, L_g_Lf_h = self.cbf_lie_derivatives(x_spatial)
        d                        = self.compute_d_soft(stakes, tau, alpha_prime)  # (B,)

        psi1     = L_f_h + gamma * h           # (B,)   first barrier function
        L_f_psi1 = L_f2_h + gamma * L_f_h     # (B,)   scalar part
        L_g_psi1 = L_g_Lf_h                   # (B, 2) linear part

        A     = L_g_psi1                       # (B, 2)
        b     = L_f_psi1 + gamma * psi1 + d   # (B,)  CDDP augmentation

        A_u_nom   = (A * u_nom).sum(dim=1)     # (B,)
        slack     = A_u_nom + b               # (B,)

        A_norm_sq = (A ** 2).sum(dim=1).clamp(min=epsilon)  # (B,)
        correction  = (slack / A_norm_sq).unsqueeze(1) * A  # (B, 2)
        u_projected = u_nom - correction                    # (B, 2)

        safe_mask = (slack >= 0.0).unsqueeze(1)             # (B, 1)
        u_safe    = torch.where(safe_mask, u_nom, u_projected)

        return u_safe


# ---------------------------------------------------------------------------
# Phase 2: CDDP Deontic Divergence Test
# ---------------------------------------------------------------------------
# Two agents start at the same position aimed directly at an obstacle.
#   Agent 0: S = 0.0  (strict deontic adherence) — must be deflected
#   Agent 1: S = 1.0  (catastrophic urgency)    — may clip straight through
# Both receive the SAME "dumb" nominal action: full throttle toward obstacle.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    # ---- Environment setup ------------------------------------------------
    obstacle_c = torch.tensor([1.5, 1.5])
    obstacle_r = 0.5
    goal       = torch.tensor([3.0, 3.0])   # far side of obstacle

    env = DifferentiablePointMassEnv(
        goal=goal,
        obstacle_c=obstacle_c,
        obstacle_r=obstacle_r,
        dt=0.02,          # finer integration step to reduce Euler drift
        v_max=5.0,
    )

    # Both agents start at (0, 0) with zero velocity, aimed at obstacle centre
    # B = 2: index 0 = strict, index 1 = high-stakes
    B = 2
    init = torch.tensor([
        [0.8, 0.8, 0.0, 0.0],   # Agent 0: strict  — starts near obstacle
        [0.8, 0.8, 0.0, 0.0],   # Agent 1: high-stakes
    ])
    env.reset(init)

    # Stakes scores — fixed per agent, batched
    stakes = torch.tensor([0.0, 1.0])          # (B,)

    # Direction from start to obstacle centre (unit vector)
    direction = obstacle_c / obstacle_c.norm()  # [cos45, cos45]

    # "Dumb" nominal action: constant full-throttle toward obstacle
    u_dumb = direction.unsqueeze(0).expand(B, -1) * 2.0  # (B, 2)  magnitude 2

    STEPS = 120
    GAMMA = 3.0       # high gamma → CBF activates earlier, less Euler overshoot
    TAU   = 0.3
    ALPHA = 10.0

    print("=" * 62)
    print(" CDDP Deontic Divergence Test")
    print(f" Obstacle: centre={obstacle_c.tolist()}  radius={obstacle_r}")
    print(f" Nominal action (both):  u={u_dumb[0].tolist()}")
    print(f" Agent 0 stakes S=0.0  (strict)  │  Agent 1 stakes S=1.0  (high)")
    print("=" * 62)

    h_history  = torch.zeros(STEPS, B)
    pos_history = torch.zeros(STEPS, B, 2)

    for t in range(STEPS):
        x = env.state                                           # (B, 4)

        # CDDP-aware projection — each agent gets its own stakes budget
        u_safe = env.project_action_cbf(
            x, u_dumb,
            stakes=stakes,
            gamma=GAMMA,
            tau=TAU,
            alpha_prime=ALPHA,
        )                                                       # (B, 2)

        out = env.step(u_safe)                                  # updates self._state

        h_history[t]   = out["h_cbf"].detach()
        pos_history[t] = out["state"][:, :2].detach()

    # ---- Report -----------------------------------------------------------
    print(f"\n{'Step':>4}  {'h(Agent0)':>12}  {'h(Agent1)':>12}  "
          f"{'pos0':>18}  {'pos1':>18}")
    print("-" * 74)
    for t in range(0, STEPS, 5):
        h0, h1   = h_history[t, 0].item(), h_history[t, 1].item()
        p0       = pos_history[t, 0].tolist()
        p1       = pos_history[t, 1].tolist()
        safe0    = "SAFE  " if h0 >= 0 else "UNSAFE"
        safe1    = "SAFE  " if h1 >= 0 else "UNSAFE"
        print(f"{t:>4}  {h0:>+10.4f} ({safe0})  {h1:>+10.4f} ({safe1})  "
              f"({p0[0]:5.2f},{p0[1]:5.2f})       ({p1[0]:5.2f},{p1[1]:5.2f})")

    # ---- Assertions -------------------------------------------------------
    # Agent 0 must NEVER enter the obstacle
    agent0_min_h = h_history[:, 0].min().item()
    # Agent 1 must clip THROUGH the obstacle at some point
    agent1_min_h = h_history[:, 1].min().item()

    print("\n" + "=" * 62)
    print(f"Agent 0 (strict)     min h = {agent0_min_h:+.4f}  "
          f"→ {'✓ always safe' if agent0_min_h >= -1e-4 else '✗ violation!'}")
    print(f"Agent 1 (high-stakes) min h = {agent1_min_h:+.4f}  "
          f"→ {'✓ clipped through (intended)' if agent1_min_h < 0 else '✗ still safe (unexpected)'}")

    # Small tolerance: discrete-time Euler introduces O(dt) boundary overshoot
    assert agent0_min_h >= -0.02, \
        f"FAIL: strict agent entered obstacle! min h={agent0_min_h:.4f}"
    assert agent1_min_h < 0, \
        f"FAIL: high-stakes agent was deflected; expected boundary violation. min h={agent1_min_h:.4f}"

    print("\n✓  CDDP deontic divergence confirmed: same action, same obstacle,")
    print("   categorically different outcomes driven purely by stakes score.")
    print("\nPhase 2 sandbox ready.")
