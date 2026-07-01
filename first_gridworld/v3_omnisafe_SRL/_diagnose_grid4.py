"""Diagnostic script for Grid 4 issues."""
import sys
sys.path.insert(0, '.')
from collections import deque
from hostage_gridworld import HostageGridworld

# Analyze Grid 4 with hard constraints (constrained agent perspective)
env = HostageGridworld(layout_id=4, bomb_timer=None, hard_constraint=True)
env.reset()

print("=== Grid 4 (10x10) Analysis ===")
print(f"Agent start:  {env.agent_start}")
print(f"Hostage pos:  {env.hostage_pos}")
print(f"Num civilians: {len(env.civilians)}")
print(f"Max steps:     {env.grid_size * 10}")
print()

# BFS distance with civilians as walls (constrained)
d_constrained = env._get_bfs_distance(env.agent_start, env.hostage_pos, ignore_civilians=False)
print(f"BFS distance (civilians=walls): {d_constrained}")

# BFS distance ignoring civilians (PPO)
d_ppo = env._get_bfs_distance(env.agent_start, env.hostage_pos, ignore_civilians=True)
print(f"BFS distance (ignore civs):     {d_ppo}")
print(f"Max steps available:            {env.grid_size * 10}")
print(f"Constrained path feasible?      {d_constrained < env.grid_size * 10}")
print()

# Check which cells are adjacent to hostage
hx, hy = env.hostage_pos
print(f"Cells adjacent to hostage {env.hostage_pos}:")
for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
    nr, nc = hx+dr, hy+dc
    if 0 <= nr < 10 and 0 <= nc < 10:
        is_civ = (nr,nc) in env.civilians
        print(f"  ({nr},{nc}) -> {'CIVILIAN (WALL)' if is_civ else 'open'}")

# Draw the grid
print()
print("Grid (C=civilian, H=hostage, A=agent):")
print("   " + " ".join(str(i) for i in range(10)))
grid = [["." for _ in range(10)] for _ in range(10)]
for c in env.civilians:
    grid[c[0]][c[1]] = "C"
grid[env.hostage_pos[0]][env.hostage_pos[1]] = "H"
grid[env.agent_start[0]][env.agent_start[1]] = "A"
for i, row in enumerate(grid):
    print(f"{i}: " + " ".join(row))

# BFS path trace for constrained
def bfs_path(start, target, civilians, grid_size):
    q = deque([(start, [start])])
    visited = {start}
    civ_set = set(civilians)
    while q:
        (r,c), path = q.popleft()
        if (r,c) == target:
            return path
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if (nr,nc) not in visited and (nr,nc) not in civ_set:
                    visited.add((nr,nc))
                    q.append(((nr,nc), path + [(nr,nc)]))
    return None

path = bfs_path(env.agent_start, env.hostage_pos, env.civilians, 10)
if path:
    print(f"\nOptimal constrained path (length {len(path)-1}):")
    for i in range(0, len(path), 5):
        chunk = path[i:i+5]
        print("  " + " -> ".join(str(p) for p in chunk))
    
    # Visualize path on grid
    print("\nPath on grid (* = path):")
    grid2 = [["." for _ in range(10)] for _ in range(10)]
    for c in env.civilians:
        grid2[c[0]][c[1]] = "C"
    for p in path:
        grid2[p[0]][p[1]] = "*"
    grid2[env.hostage_pos[0]][env.hostage_pos[1]] = "H"
    grid2[env.agent_start[0]][env.agent_start[1]] = "A"
    print("   " + " ".join(str(i) for i in range(10)))
    for i, row in enumerate(grid2):
        print(f"{i}: " + " ".join(row))
else:
    print("\n!!! NO PATH EXISTS with civilians as walls !!!")

# PPO observation difference analysis
print("\n=== PPO Observation Mismatch ===")
env_high = HostageGridworld(layout_id=4, bomb_timer=20, hard_constraint=False)
env_low = HostageGridworld(layout_id=4, bomb_timer=None, hard_constraint=False)
obs_high, _ = env_high.reset()
obs_low, _ = env_low.reset()
print(f"Feature names:    [agent_y, agent_x, host_y, host_x, rel_y, rel_x, dist, bomb_active, time_norm]")
print(f"High stakes obs:  {[f'{v:.3f}' for v in obs_high]}")
print(f"Low stakes obs:   {[f'{v:.3f}' for v in obs_low]}")
print(f"\nKey difference:")
print(f"  bomb_active: HIGH={obs_high[7]:.1f}  LOW={obs_low[7]:.1f}")
print(f"  time_norm:   HIGH={obs_high[8]:.3f}  LOW={obs_low[8]:.3f}")
print(f"\nPPO trains with bomb_timer=20 -> bomb_active=1.0, time_norm=1.0")
print(f"PPO eval low stakes: bomb_timer=None -> bomb_active=0.0, time_norm=0.0")
print(f"=> DISTRIBUTION SHIFT: agent never saw bomb_active=0 during training!")
