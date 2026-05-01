"""
Tabular Oracle Debugger — Pre-training stakes visualization.

Generates a map over all reachable states showing:
    - D_min (shortest path ignoring civilians)
    - L_safe (shortest path avoiding civilians)
    - S_analytic (oracle stakes)
    - Mode (deontic/utilitarian) for a given τ
    - Failure band membership

Catches BFS bugs, timer indexing, and threshold issues early.
Also produces paper-quality figures.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

_omnisafe_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'omnisafe_SRL')
sys.path.insert(0, os.path.abspath(_omnisafe_dir))
from hostage_gridworld import HostageGridworld
from cddp.env.hostage_env import InstrumentedHostageEnv


def generate_stakes_map(layout_id: int, bomb_timer: int, tau: float = 0.5) -> dict:
    """Compute CDDP quantities for every cell in the grid.

    Args:
        layout_id: Grid layout ID (1, 2, or 3)
        bomb_timer: Bomb timer value (or None for low stakes)
        tau: Mode-switching threshold

    Returns:
        dict mapping (row, col) → {
            'D_min': int,
            'L_safe': int|float,
            'T_rem': float,
            'S_analytic': float,
            'mode': 'deontic'|'utilitarian',
            'in_failure_band': bool,
            'safe_actions': list[int],
            'is_civilian': bool,
            'is_hostage': bool,
        }
    """
    base = HostageGridworld(bomb_timer=bomb_timer, layout_id=layout_id)
    env = InstrumentedHostageEnv(base)
    env.reset()

    grid_size = env.grid_size
    civilians = set(env.civilians)
    hostage = env.hostage_pos

    stakes_map = {}

    for r in range(grid_size):
        for c in range(grid_size):
            pos = (r, c)

            # Temporarily place the agent at this position for BFS
            env.env.agent_pos = list(pos)
            if bomb_timer is not None:
                env.env.bomb_time_left = bomb_timer

            D = env.get_D_min(pos)
            L = env.get_L_safe(pos)
            T = env.get_T_rem()
            safe = env.get_safe_actions(pos)

            import math
            if math.isinf(T):
                S = 0.0
            elif T <= 0:
                S = 1.0
            else:
                S = min(max(D / T, 0.0), 1.0)

            in_F = D < T < L

            stakes_map[pos] = {
                'D_min': D,
                'L_safe': L if not math.isinf(L) else '∞',
                'T_rem': T if not math.isinf(T) else '∞',
                'S_analytic': round(S, 3),
                'mode': 'utilitarian' if S >= tau else 'deontic',
                'in_failure_band': in_F,
                'safe_actions': safe,
                'is_civilian': pos in civilians,
                'is_hostage': pos == hostage,
            }

    return stakes_map


def print_stakes_map(stakes_map: dict, grid_size: int, tau: float = 0.5):
    """Print a human-readable stakes map to console."""
    print(f"\n{'='*60}")
    print(f"CDDP TABULAR ORACLE  (τ = {tau})")
    print(f"{'='*60}")

    print(f"\n--- Stakes S_analytic ---")
    for r in range(grid_size):
        row_str = ""
        for c in range(grid_size):
            cell = stakes_map[(r, c)]
            if cell['is_hostage']:
                row_str += "  H   "
            elif cell['is_civilian']:
                row_str += "  C   "
            else:
                row_str += f" {cell['S_analytic']:.2f} "
        print(row_str)

    print(f"\n--- Mode (D=deontic, U=utilitarian) ---")
    for r in range(grid_size):
        row_str = ""
        for c in range(grid_size):
            cell = stakes_map[(r, c)]
            if cell['is_hostage']:
                row_str += " H "
            elif cell['is_civilian']:
                row_str += " C "
            else:
                m = 'U' if cell['mode'] == 'utilitarian' else 'D'
                row_str += f" {m} "
        print(row_str)

    print(f"\n--- Failure Band (F = in band) ---")
    for r in range(grid_size):
        row_str = ""
        for c in range(grid_size):
            cell = stakes_map[(r, c)]
            if cell['is_hostage']:
                row_str += " H "
            elif cell['is_civilian']:
                row_str += " C "
            else:
                f = 'F' if cell['in_failure_band'] else '·'
                row_str += f" {f} "
        print(row_str)

    print(f"\n--- D_min / L_safe ---")
    for r in range(grid_size):
        row_str = ""
        for c in range(grid_size):
            cell = stakes_map[(r, c)]
            if cell['is_hostage']:
                row_str += "   H    "
            elif cell['is_civilian']:
                row_str += "   C    "
            else:
                L_str = str(cell['L_safe']) if cell['L_safe'] != '∞' else '∞'
                row_str += f" {cell['D_min']}/{L_str}  "
        print(row_str)


def visualize_stakes_map(stakes_map: dict, grid_size: int, layout_id: int,
                         bomb_timer: int, tau: float = 0.5,
                         save_path: str = None):
    """Render a colored grid visualization of stakes.

    Produces a 2x2 figure:
        [0,0] S_analytic heatmap
        [0,1] Mode map (deontic=blue, utilitarian=red)
        [1,0] D_min / L_safe annotation
        [1,1] Failure band overlay
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'CDDP Oracle Map — Grid {layout_id}, T={bomb_timer}, τ={tau}',
                 fontsize=14, fontweight='bold')

    # Collect grid data
    civilians = set()
    hostage = None
    S_grid = np.zeros((grid_size, grid_size))
    mode_grid = np.zeros((grid_size, grid_size))
    F_grid = np.zeros((grid_size, grid_size))
    D_grid = np.zeros((grid_size, grid_size))
    L_grid = np.zeros((grid_size, grid_size))

    for (r, c), cell in stakes_map.items():
        S_grid[r, c] = cell['S_analytic']
        mode_grid[r, c] = 1.0 if cell['mode'] == 'utilitarian' else 0.0
        F_grid[r, c] = 1.0 if cell['in_failure_band'] else 0.0
        D_grid[r, c] = cell['D_min']
        L_val = cell['L_safe']
        L_grid[r, c] = L_val if isinstance(L_val, (int, float)) and L_val != '∞' else grid_size * 2
        if cell['is_civilian']:
            civilians.add((r, c))
        if cell['is_hostage']:
            hostage = (r, c)

    # ─── Panel 1: Stakes heatmap ───
    ax = axes[0, 0]
    cmap = LinearSegmentedColormap.from_list('stakes', ['#2ecc71', '#f39c12', '#e74c3c'])
    im = ax.imshow(S_grid, cmap=cmap, vmin=0, vmax=1, origin='upper')
    plt.colorbar(im, ax=ax, label='S_analytic')
    ax.set_title('Analytical Stakes')
    _annotate_grid(ax, stakes_map, grid_size, 'S_analytic')

    # ─── Panel 2: Mode map ───
    ax = axes[0, 1]
    mode_cmap = LinearSegmentedColormap.from_list('mode', ['#3498db', '#e74c3c'])
    ax.imshow(mode_grid, cmap=mode_cmap, vmin=0, vmax=1, origin='upper')
    ax.set_title(f'Mode (τ={tau})')
    _annotate_grid(ax, stakes_map, grid_size, 'mode')
    ax.legend(handles=[
        mpatches.Patch(color='#3498db', label='Deontic'),
        mpatches.Patch(color='#e74c3c', label='Utilitarian'),
    ], loc='lower right', fontsize=8)

    # ─── Panel 3: D_min / L_safe ───
    ax = axes[1, 0]
    ax.imshow(D_grid, cmap='YlOrRd', origin='upper')
    ax.set_title('D_min (unsafest path)')
    for (r, c), cell in stakes_map.items():
        L_str = str(cell['L_safe']) if cell['L_safe'] != '∞' else '∞'
        if cell['is_civilian']:
            ax.text(c, r, 'C', ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
        elif cell['is_hostage']:
            ax.text(c, r, 'H', ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))
        else:
            ax.text(c, r, f"{cell['D_min']}/{L_str}", ha='center', va='center',
                    fontsize=7)

    # ─── Panel 4: Failure band ───
    ax = axes[1, 1]
    F_cmap = LinearSegmentedColormap.from_list('fband', ['#ecf0f1', '#e74c3c'])
    ax.imshow(F_grid, cmap=F_cmap, vmin=0, vmax=1, origin='upper')
    ax.set_title('Failure Band (F)')
    _annotate_grid(ax, stakes_map, grid_size, 'in_failure_band')

    # Grid lines for all panels
    for ax in axes.flat:
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which='minor', color='black', linewidth=0.5, alpha=0.3)
        ax.tick_params(which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    return fig


def _annotate_grid(ax, stakes_map, grid_size, field):
    """Add text annotations to grid cells."""
    for (r, c), cell in stakes_map.items():
        if cell['is_civilian']:
            ax.text(c, r, 'C', ha='center', va='center', fontsize=9,
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
        elif cell['is_hostage']:
            ax.text(c, r, 'H', ha='center', va='center', fontsize=9,
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))
        else:
            val = cell[field]
            if isinstance(val, bool):
                label = 'F' if val else '·'
            elif isinstance(val, float):
                label = f'{val:.2f}'
            else:
                label = str(val)[0].upper()  # D or U for mode
            ax.text(c, r, label, ha='center', va='center', fontsize=8)


# ─── CLI ────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CDDP Tabular Oracle Debugger')
    parser.add_argument('--grid', type=int, default=1, help='Grid layout (1-3)')
    parser.add_argument('--timer', type=int, default=5, help='Bomb timer')
    parser.add_argument('--tau', type=float, default=0.5, help='Threshold τ')
    parser.add_argument('--save', type=str, default=None, help='Save path for figure')
    args = parser.parse_args()

    smap = generate_stakes_map(args.grid, args.timer, args.tau)
    grid_size = HostageGridworld.GRID_LAYOUTS[args.grid]['grid_size']

    print_stakes_map(smap, grid_size, args.tau)
    visualize_stakes_map(smap, grid_size, args.grid, args.timer, args.tau,
                         save_path=args.save)
    if args.save is None:
        plt.show()
