"""
failure_band.py — Single source of truth for the CDDP failure-band predicate.

The failure band F is the set of states where the agent can reach the hostage
in time via an unsafe shortcut (ignoring civilians) but the safe detour takes
too long.  Formally:

    s ∈ F  ⟺  D_min(s) < T_rem(s) < L_safe(s)

This function is imported by:
  - hostage_env._cddp_info()  (env flag in info dict)
  - hostage_env.is_in_failure_band()  (env method)
  - cddp_agent.select_action()  (debug dump: in_failure_band_metric_recomputed)
  - evaluate_agent()  (metrics: in_failure_band_metric_logged)

Do NOT duplicate this logic anywhere else.
"""


def is_failure_band(D_min: int, T_rem: float, L_safe: float) -> bool:
    """Return True iff state is in the CDDP failure band F.

    Args:
        D_min:  Shortest path to hostage ignoring civilians (int steps).
        T_rem:  Remaining time steps before bomb detonates (float or inf).
        L_safe: Shortest path to hostage avoiding all civilians (float or inf).

    Returns:
        True if D_min < T_rem < L_safe, else False.
    """
    return bool(D_min < T_rem < L_safe)
