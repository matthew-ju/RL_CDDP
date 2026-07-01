"""
PlannerOracleAgent — True Upper-Bound Oracle Controller
=========================================================

Uses exact BFS/Dijkstra planning on the live environment state to select actions,
bypassing any learned actor. This gives a theoretical upper bound on performance.

Decision rule (per step):
  1. Compute L_safe(s) = BFS avoiding all civilians
  2. If L_safe(s) <= T_rem: deontic mode — next step on BFS safe path
  3. Else: utilitarian mode — next step on Dijkstra min-violation path subject to T_rem

No oscillation: once the oracle commits to utilitarian mode, it stays there.
"""
import heapq
from collections import deque


# Discrete action deltas: 0=up, 1=down, 2=left, 3=right
_ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


def _bfs_safe_path(env, pos, avoid_civilians=True):
    """Return list of actions for shortest path pos→hostage, avoiding civilians if requested.
    Returns [] if no path exists (or already at hostage).
    """
    base = env.env if hasattr(env, 'env') else env
    targets = [hp for hp in base.hostage_positions if hp not in base.hostages_saved]
    if not targets:
        targets = list(base.hostage_positions)
    target_set = frozenset(targets)

    if pos in target_set:
        return []

    # Build blocked set
    active_civilians = set(base.civilians) - base.trampled_civilians
    walls = base.walls
    blocked = walls | (active_civilians if avoid_civilians else set())

    grid_val = env.grid_id if hasattr(env, 'grid_id') else getattr(base, 'layout_id', 1)
    # print(f"[DEBUG] DeonticOracle BFS grid_id={grid_val}, ...")

    # BFS — track parent pointers to reconstruct path
    queue = deque([(pos, [])])
    visited = {pos}

    while queue:
        cur, path = queue.popleft()
        for action, (dr, dc) in _ACTION_DELTAS.items():
            r, c = cur[0] + dr, cur[1] + dc
            # Boundary clamp
            gs = base.grid_size
            if not (0 <= r < gs and 0 <= c < gs):
                continue
            nr, nc = r, c
            # Wall bounce — action is safe (agent stays put) so no progress possible
            if (nr, nc) in walls:
                continue
            if (nr, nc) in blocked and (nr, nc) not in target_set:
                continue
            npos = (nr, nc)
            if npos in visited:
                continue
            visited.add(npos)
            npath = path + [action]
            if npos in target_set:
                return npath
            queue.append((npos, npath))

    return []  # unreachable


def _dijkstra_min_violation_path(env, pos, t_rem):
    """Return list of actions for minimum-civilian-violation path from pos to hostage,
    constrained to at most t_rem steps. Uses Dijkstra on (r, c, t_rem) state space.
    Returns [] if no feasible path.
    """
    base = env.env if hasattr(env, 'env') else env
    targets = [hp for hp in base.hostage_positions if hp not in base.hostages_saved]
    if not targets:
        targets = list(base.hostage_positions)
    target_set = frozenset(targets)

    if pos in target_set:
        return []

    active_civilians = set(base.civilians) - base.trampled_civilians
    walls = base.walls
    gs = base.grid_size

    # Track paths separately using a counter as node id to avoid comparing lists in heap
    counter = 0
    paths = {counter: []}

    # Heap entry: (cost, r, c, t_rem_left, path_id)
    pq = [(0, pos[0], pos[1], int(t_rem), counter)]
    visited = {}  # (r, c, t_rem) -> best_cost

    while pq:
        cost, r, c, t, pid = heapq.heappop(pq)

        state = (r, c, t)
        if state in visited and visited[state] <= cost:
            continue
        visited[state] = cost

        path = paths[pid]

        if (r, c) in target_set:
            return path

        if t <= 0:
            continue

        for action, (dr, dc) in _ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            # Boundary clamp — bumping boundary is a wasted step
            if not (0 <= nr < gs and 0 <= nc < gs):
                nr, nc = r, c  # stays put
            # Wall bounce
            if (nr, nc) in walls:
                nr, nc = r, c

            step_cost = 1 if (nr, nc) in active_civilians else 0
            nt = t - 1
            ns = (nr, nc, nt)
            nc_cost = cost + step_cost

            if ns in visited and visited[ns] <= nc_cost:
                continue

            counter += 1
            paths[counter] = path + [action]
            heapq.heappush(pq, (nc_cost, nr, nc, nt, counter))

    return []  # no path within t_rem


class PlannerOracleAgent:
    """True upper-bound agent using exact BFS/Dijkstra planning.

    Implements the interface expected by evaluate_agent():
      - select_action(obs, env, deterministic=True) -> (action, act_info)
      - reset_mode_tracking()

    Decision rule (latched — no oscillation):
      s0: if L_safe(s0) <= T_rem → start deontic; else → start utilitarian
      once latched to utilitarian, stays utilitarian for the episode.
    """

    def __init__(self):
        self._latch_util = False
        self._current_plan = []   # cached planned action sequence

    def reset_mode_tracking(self):
        self._latch_util = False
        self._current_plan = []

    def select_action(self, obs, env, deterministic=True):
        pos = tuple(env.agent_pos)
        t_rem = env.get_T_rem()
        l_safe = env.get_L_safe(pos)

        # Latch: if L_safe > T_rem, switch to utilitarian and stay there
        if l_safe > t_rem:
            self._latch_util = True

        mode = 'utilitarian' if self._latch_util else 'deontic'

        if mode == 'deontic':
            # Re-plan BFS safe path from current position
            plan = _bfs_safe_path(env, pos, avoid_civilians=True)
            if plan:
                action = plan[0]
            else:
                # Infeasible — signal episode failure, do NOT take unsafe action
                action = 6
        else:
            # Utilitarian: Dijkstra min-violation within T_rem
            plan = _dijkstra_min_violation_path(env, pos, t_rem)
            if plan:
                action = plan[0]
            else:
                # Emergency: unconstrained BFS
                plan = _bfs_safe_path(env, pos, avoid_civilians=False)
                action = plan[0] if plan else 0

        stakes = 1.0 if self._latch_util else 0.0
        act_info = {
            'mode': mode,
            'stakes': stakes,
            'p_cat_hat': stakes,
            'action_was_shielded': False,
        }
        return int(action), act_info
