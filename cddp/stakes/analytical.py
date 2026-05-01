"""
Analytical Stakes Variants — Oracle stakes functions using environment info.

Provides:
- AnalyticalStakesDmin: S_dmin = clip(D_min / T_rem, 0, 1)
- AnalyticalStakesLsafe: S_lsafe = clip(L_safe / T_rem, 0, 1)
- AnalyticalFeasibilityTrigger: S_feas = 1 if L_safe > T_rem else 0
- AnalyticalOracleStakes: monotone-latching oracle (CDDP-Oracle baseline)

These are "oracle" stakes functions that use exact BFS distances
and the known bomb timer. Used in Phases 1–2 for validating the
CDDP architecture before distributional critics are trained.
"""

import math


class AnalyticalStakesDmin:
    """Analytical / oracle stakes: S(s) = clip(D_min / T_rem, 0, 1).

    Stakes semantics:
        - 0.0: no urgency (no bomb, or plenty of time)
        - 1.0: maximum urgency (must shortcut immediately)
    """

    def compute(self, env) -> float:
        T_rem = env.get_T_rem()

        if math.isinf(T_rem):
            return 0.0
        if T_rem <= 0:
            return 1.0

        D_min = env.get_D_min()
        return min(max(D_min / T_rem, 0.0), 1.0)


class AnalyticalStakesLsafe:
    """Analytical stakes using safe path: S(s) = clip(L_safe / T_rem, 0, 1)."""

    def compute(self, env) -> float:
        T_rem = env.get_T_rem()

        if math.isinf(T_rem):
            return 0.0
        if T_rem <= 0:
            return 1.0

        L_safe = env.get_L_safe()
        return min(max(L_safe / T_rem, 0.0), 1.0)


class AnalyticalFeasibilityTrigger:
    """Feasibility trigger: S(s) = 1.0 if L_safe > T_rem else 0.0."""

    def compute(self, env) -> float:
        T_rem = env.get_T_rem()

        if math.isinf(T_rem):
            return 0.0
        if T_rem <= 0:
            return 1.0

        L_safe = env.get_L_safe()
        return 1.0 if L_safe > T_rem else 0.0


class AnalyticalOracleStakes:
    """Ground-truth oracle stakes with monotone episode latch.

    This is the correct "oracle" baseline for CDDP. It uses ground-truth
    failure-band membership (D_min >= T_rem) and applies a one-way latch:
    once the failure band is entered, the oracle returns 1.0 for the rest
    of the episode — i.e., it never switches back to deontic.

    Rationale: the theoretically optimal switch is monotone. The oracle
    knows the bomb deadline and BFS distances exactly. Once the constraint
    is tight enough that the safe path no longer fits (D_min >= T_rem),
    there is no reason to switch back — the greedy policy remains correct.
    This prevents the oscillation caused by D_min/T_rem fluctuating around τ.

    Attributes:
        last_p_cat: float — tracks last computed value (for metrics compat).
    """

    def __init__(self):
        self._latched: bool = False    # one-way switch: deontic → utilitarian
        self.last_p_cat: float = 0.0  # for CDDPAgent metrics compatibility

    def reset(self) -> None:
        """Reset the latch at episode boundaries."""
        self._latched = False
        self.last_p_cat = 0.0

    def compute(self, env) -> float:
        """Return stakes based on ground-truth L_safe infeasibility + monotone latch.

        Latch condition: L_safe > T_rem (safe path no longer fits in time remaining).
        This is the theoretically correct oracle criterion — once the safe route is
        infeasible the oracle should stay utilitarian. Using D_min would trigger too
        early (an unsafe shortcut may still exist with D_min < T_rem < L_safe).
        """
        if self._latched:
            self.last_p_cat = 1.0
            return 1.0

        T_rem = env.get_T_rem()

        if math.isinf(T_rem):
            self.last_p_cat = 0.0
            return 0.0

        if T_rem <= 0:
            self._latched = True
            self.last_p_cat = 1.0
            return 1.0

        L_safe = env.get_L_safe()
        # Latch when safe route is blocked or takes longer than time remaining
        if math.isinf(L_safe) or L_safe > T_rem:
            self._latched = True
            self.last_p_cat = 1.0
            return 1.0

        self.last_p_cat = 0.0
        return 0.0

