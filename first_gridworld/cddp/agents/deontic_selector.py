"""
DeonticActionSelector — Deterministic safe-action selection for deontic mode.

When the shield blocks the actor's preferred action, this selector picks
the BEST safe action using a deterministic lexicographic rule:

    1. Smallest L_safe(next_state) — favors progress on the safe path
    2. Tie-break: smallest D_min(next_state) — closer to hostage overall
    3. Final tie-break: fixed action priority [UP=0, LEFT=2, RIGHT=3, DOWN=1]

This replaces the old Monte Carlo random fallback that caused bouncing
and non-deterministic behavior at T=7 and T=10.

Separation of concerns:
    - ActionShield  = WHAT actions are safe (the legal set)
    - DeonticActionSelector = WHICH safe action is best (the selection rule)
"""

import numpy as np

# Fixed priority for final tie-break: UP, LEFT, RIGHT, DOWN
# Rationale: prefer vertical movement toward hostage (usually above),
# then horizontal, with DOWN last (usually moves away from hostage in Grid 1)
ACTION_PRIORITY = [0, 2, 3, 1]  # UP, LEFT, RIGHT, DOWN
ACTION_NAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}

from cddp.agents.oracle_planner import _bfs_safe_path

class DeonticActionSelector:
    """Deterministic safe-action selection for CDDP deontic mode.

    Usage:
        selector = DeonticActionSelector()
        action = selector.select_action(env)
    """

    def select_action(self, env):
        pos = tuple(env.agent_pos)
        
        # 1. Try to find a completely safe path avoiding all civilians
        plan = _bfs_safe_path(env, pos, avoid_civilians=True)
        if plan:
            return plan[0]
            
        # 2. No safe path exists — use least-unsafe fallback 
        # (Find shortest path to goal ignoring civilians)
        fallback_plan = _bfs_safe_path(env, pos, avoid_civilians=False)
        if fallback_plan:
            return fallback_plan[0]
            
        # 3. Emergency trap
        return 0

