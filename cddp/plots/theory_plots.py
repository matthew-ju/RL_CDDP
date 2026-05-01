"""
CDDP vs Baselines Plotting.

Generates the key comparison plots:
    1. Performance bars: save rate, violation rate, safety score across agents
    2. Training curves: reward, violations, save rate over epochs
    3. Mode switching map: stakes heatmap + mode switching boundaries
    4. Theorem validation: deontic violation count over steps
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_comparison_bars(results_dict, save_path=None, title='CDDP vs Baselines'):
    """Bar chart comparing save rate, violation rate, safety score."""
    agents = list(results_dict.keys())
    n = len(agents)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    metrics = [
        ('save_rate', 'Save Rate (%)', 100),
        ('violation_rate', 'Avg Violations', 1),
        ('safety_score', 'Safety Score (%)', 100),
    ]

    for ax, (key, ylabel, scale) in zip(axes, metrics):
        vals = [results_dict[a].get(key, 0) * scale for a in agents]
        bars = ax.bar(range(n), vals, color=colors[:n])
        ax.set_xticks(range(n))
        ax.set_xticklabels(agents, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

        for bar, val in zip(bars, vals):
            fmt = f'{val:.1f}' if key != 'violation_rate' else f'{val:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    fmt, ha='center', va='bottom', fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_training_curves(history_path, save_path=None):
    """Plot training curves from history JSON."""
    with open(history_path) as f:
        h = json.load(f)

    epochs = h.get('epoch', [])
    if not epochs:
        print("No training data found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Reward
    axes[0, 0].plot(epochs, h.get('avg_reward', []), 'b-', alpha=0.7)
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(True, alpha=0.3)

    # Save rate
    axes[0, 1].plot(epochs, [x * 100 for x in h.get('save_rate', [])], 'g-', alpha=0.7)
    axes[0, 1].set_title('Save Rate (%)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylim(-5, 105)
    axes[0, 1].grid(True, alpha=0.3)

    # Violations
    axes[1, 0].plot(epochs, h.get('avg_violations', []), 'r-', alpha=0.7)
    axes[1, 0].set_title('Avg Violations')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, alpha=0.3)

    # Deontic fraction
    axes[1, 1].plot(epochs, [x * 100 for x in h.get('deontic_frac', [])], 'm-', alpha=0.7)
    axes[1, 1].set_title('Deontic Mode (%)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylim(-5, 105)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('CDDP Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_stakes_heatmap(env, stakes_fn, save_path=None, tau=0.5):
    """Plot stakes values and mode boundaries for each cell in the grid."""
    from cddp.env.hostage_env import InstrumentedHostageEnv

    grid_size = env.grid_size
    stakes_map = np.zeros((grid_size, grid_size))
    mode_map = np.full((grid_size, grid_size), '', dtype=object)

    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) in env.walls:
                stakes_map[r, c] = np.nan
                mode_map[r, c] = 'wall'
                continue

            # Temporarily move agent
            old_pos = list(env.agent_pos)
            env.env.agent_pos = [r, c]

            s = stakes_fn.compute(env)
            stakes_map[r, c] = s
            mode_map[r, c] = 'D' if s < tau else 'U'

            env.env.agent_pos = old_pos

    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = plt.cm.RdYlGn_r
    cmap.set_bad('gray', 0.3)

    im = ax.imshow(stakes_map, cmap=cmap, vmin=0, vmax=1, origin='upper')
    plt.colorbar(im, ax=ax, label='Stakes S(s)')

    # Annotate cells
    for r in range(grid_size):
        for c in range(grid_size):
            if mode_map[r, c] == 'wall':
                ax.text(c, r, '█', ha='center', va='center', color='gray', fontsize=12)
            elif (r, c) == env.hostage_pos:
                ax.text(c, r, 'H', ha='center', va='center', color='white',
                        fontsize=14, fontweight='bold')
            elif (r, c) in set(env.civilians):
                ax.text(c, r, 'C', ha='center', va='center', color='yellow',
                        fontsize=12, fontweight='bold')
            elif tuple(env.agent_pos) == (r, c):
                ax.text(c, r, 'A', ha='center', va='center', color='blue',
                        fontsize=14, fontweight='bold')
            else:
                ax.text(c, r, f'{stakes_map[r,c]:.2f}\n{mode_map[r,c]}',
                        ha='center', va='center', fontsize=8)

    ax.set_title(f'Stakes Heatmap (τ={tau})', fontsize=14, fontweight='bold')
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_theorem_validation(step_data, save_path=None):
    """Plot cumulative deontic violations over steps (Theorem 1)."""
    deontic_steps = [s for s in step_data if s['mode'] == 'deontic']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mode distribution
    n_deontic = sum(1 for s in step_data if s['mode'] == 'deontic')
    n_utilitarian = sum(1 for s in step_data if s['mode'] == 'utilitarian')
    axes[0].bar(['Deontic', 'Utilitarian'], [n_deontic, n_utilitarian],
                color=['#2196F3', '#FF5722'])
    axes[0].set_title('Mode Distribution')
    axes[0].set_ylabel('Step Count')

    # Stakes distribution
    stakes = [s['stakes'] for s in step_data]
    axes[1].hist(stakes, bins=50, color='#9C27B0', alpha=0.7, edgecolor='black')
    axes[1].axvline(0.5, color='red', linestyle='--', label='τ=0.5')
    axes[1].set_title('Stakes Distribution')
    axes[1].set_xlabel('Stakes S(s)')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    plt.suptitle('CDDP Theorem Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
