"""
Phase 4 v2 Paper-Ready Plots — Trajectory Pareto Redesign
==========================================================
Changes vs plot_phase4_3.py:
  - plot_pareto() replaced with plot_pareto_trajectory():
    * 1 subplot per grid (not 3 snapshots)
    * each model = a CURVE connected across all T values
    * same COLORS / LINESTYLES as performance plots
    * ghost per-seed curves alpha=0.15, lw=1
    * bootstrap CI band alpha=0.15
    * dilemma zone background shading (same as performance plot)
    * timer labels at T_min/T_mid/T_max (fontsize=7, low opacity)
    * NO markers on mean lines
  - output to results/phase4_paper_ready_v2/

Main paper:
  grid{N}_phase4_performance.png  (4-panel with dilemma shading)
Appendix:
  grid{N}_phase4_switching_quality.png
  grid{N}_phase4_pareto.png         (trajectory frontier)
  phase4_summary.md                (metrics summary report)
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Try v2 results first, fall back to v3 for comparison
V2_SWEEP_DIR = os.path.join(project_root, "results")
OUT_ROOT     = os.path.join(project_root, "results", "phase4_paper_ready_v2")

MAIN_MODELS_G1 = [
    'PPO-Baseline-Finetuned',
    'Deontic-Oracle',
    'CDDP-Oracle',
    'CDDP-Prospective',
]
MAIN_MODELS_G2 = MAIN_MODELS_G1[:]   # same set for G2

COLORS = {
    'PPO-Baseline':                     'gray',
    'PPO-Baseline-Finetuned':           'black',
    'Deontic-Oracle':                   'steelblue',
    'CDDP-Oracle':                      '#2CA02C',   # green
    'CDDP-Prospective':                 '#9467BD',   # purple
    'CDDP-Prospective (Calibrated)':    '#D6604D',
    'CDDP-Prospective (Sticky)':        '#8856A7',
}

LINESTYLES = {
    'PPO-Baseline':                     '-',
    'PPO-Baseline-Finetuned':           '--',
    'Deontic-Oracle':                   '-.',
    'CDDP-Oracle':                      '-',
    'CDDP-Prospective':                 '-',
    'CDDP-Prospective (Calibrated)':    '--',
    'CDDP-Prospective (Sticky)':        ':',
}

N_BOOT = 1000
CI     = 0.95


# ── Helpers ──────────────────────────────────────────────────────────────────

def setup_ax(ax, title, ylabel, timers):
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Timer ($T$)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(timers)


def bootstrap_ci(vals, n_boot=N_BOOT, ci=CI):
    if len(vals) < 2:
        m = float(np.mean(vals)) if vals else float('nan')
        return m, m
    rng = np.random.default_rng(42)
    boot = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(boot, 100 * alpha)), float(np.percentile(boot, 100 * (1 - alpha)))


def get_mean_ci(data, model, timers, key, scale=1.0, default=0.0):
    means, lo_list, hi_list = [], [], []
    for T in timers:
        rows = data[model][str(T)].get('seeds_data', [])
        vals = [r.get(key, default) * scale for r in rows
                if isinstance(r.get(key, default), (int, float))
                and not np.isnan(r.get(key, default))]
        if vals:
            m = float(np.mean(vals))
            lo, hi = bootstrap_ci(vals)
        else:
            m = data[model][str(T)].get(f'mean_{key}', default) * scale
            lo, hi = m, m
        means.append(m)
        lo_list.append(lo)
        hi_list.append(hi)
    return means, lo_list, hi_list


def get_per_seed_series(data, model, timers, key, scale=1.0):
    n_seeds = len(data[model][str(timers[0])].get('seeds_data', []))
    series = []
    for si in range(n_seeds):
        row = []
        for T in timers:
            rows = data[model][str(T)].get('seeds_data', [])
            v = rows[si].get(key, float('nan')) * scale if si < len(rows) else float('nan')
            row.append(v)
        series.append(row)
    return series


def compute_dilemma_xcoords(data, model, timers):
    dilemma = []
    for T in timers:
        rows = data[model][str(T)].get('seeds_data', [])
        vals = [r.get('l_safe_s0', float('nan')) for r in rows]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            if float(np.median(vals)) > T:
                dilemma.append(T)
        else:
            mean_ls = data[model][str(T)].get('mean_l_safe_s0', float('nan'))
            if not np.isnan(mean_ls) and mean_ls > T:
                dilemma.append(T)
    return dilemma


def shade_dilemma_band(ax, dilemma_timers, timers):
    if not dilemma_timers:
        return
    sorted_t = sorted(dilemma_timers)
    dt = max(1, timers[1] - timers[0]) if len(timers) > 1 else 1
    ax.axvspan(sorted_t[0] - dt / 2, sorted_t[-1] + dt / 2,
               color='#FFCCCC', alpha=0.35, zorder=0, label='_nolegend_')


def load_results(grid_id):
    if grid_id == 1:
        path = os.path.join(V2_SWEEP_DIR, "phase4_v2_grid1_sweeps_debug.json")
    else:
        path = os.path.join(V2_SWEEP_DIR, "phase4_v2_grid2_sweeps.json")
    if os.path.exists(path):
        print(f"  [INFO] Loading {path}")
        with open(path) as f:
            return json.load(f)
    print(f"  [WARN] No sweep JSON found for grid {grid_id} at {path}")
    return None


# ── Figure 1: Performance ─────────────────────────────────────────────────────

def plot_performance(grid_id, data, out_dir, main_models):
    models   = [m for m in main_models if m in data]
    timers   = sorted(int(k) for k in next(iter(data.values())).keys())
    dilemma_ref = 'CDDP-Oracle' if 'CDDP-Oracle' in data else (models[0] if models else None)
    dilemma  = compute_dilemma_xcoords(data, dilemma_ref, timers) if dilemma_ref else []

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for ax in axes:
        shade_dilemma_band(ax, dilemma, timers)

    for model in models:
        c  = COLORS.get(model, 'black')
        ls = LINESTYLES.get(model, '-')

        for key, ax, scale in [
            ('feasible_success_rate',    axes[0], 100),
            ('violation_rate',           axes[1], 1),
            ('excess_violation_rate',    axes[2], 1),
            ('episodes_with_excess_rate', axes[3], 100),
        ]:
            for seed_series in get_per_seed_series(data, model, timers, key, scale):
                ax.plot(timers, seed_series, color=c, linestyle=ls, alpha=0.15, linewidth=1)
            means, lo, hi = get_mean_ci(data, model, timers, key, scale)
            ax.plot(timers, means, label=model, color=c, linestyle=ls,
                    linewidth=2.5, alpha=1.0, marker='')
            ax.fill_between(timers, lo, hi, color=c, alpha=0.15)

    setup_ax(axes[0], 'Success Rate vs Timer (Feasible Only)', 'Success %', timers)
    axes[0].set_ylim([-5, 105])
    setup_ax(axes[1], 'Total Violations vs Timer',     'Avg Violations / Ep',        timers)
    setup_ax(axes[2], 'Excess Violations vs Timer',    'Avg Excess Violations / Ep', timers)
    setup_ax(axes[3], 'P(excess > 0) vs Timer',        '% Episodes w/ Excess > 0',  timers)
    axes[3].set_ylim([-5, 105])

    if dilemma:
        patch = mpatches.Patch(color='#FFCCCC', alpha=0.7, label='Dilemma zone\n($L_{safe}>T$)')
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(patch); labels.append(patch.get_label())
        fig.legend(handles, labels, loc='lower center', ncol=len(models) + 1,
                   bbox_to_anchor=(0.5, -0.06), fontsize=11)
    else:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(models),
                   bbox_to_anchor=(0.5, -0.05), fontsize=11)

    plt.tight_layout()
    out = os.path.join(out_dir, f'grid{grid_id}_phase4_performance.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 2: Switching Quality ───────────────────────────────────────────────

def plot_switching_quality(grid_id, data, out_dir, main_models):
    cddp_models = [m for m in main_models if 'CDDP' in m and m in data]
    timers = sorted(int(k) for k in next(iter(data.values())).keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model in cddp_models:
        c  = COLORS.get(model, 'black')
        ls = LINESTYLES.get(model, '-')

        for seed_series in get_per_seed_series(data, model, timers, 'oscillation_rate', 100):
            axes[0].plot(timers, seed_series, color=c, linestyle=ls, alpha=0.15, linewidth=1)

        for key, ax, scale in [
            ('oscillation_rate', axes[0], 100),
            ('l_safe_s0',        axes[1], 1),
        ]:
            means, lo, hi = get_mean_ci(data, model, timers, key, scale)
            ax.plot(timers, means, label=model, color=c, linestyle=ls,
                    linewidth=2.5, alpha=1.0, marker='')
            ax.fill_between(timers, lo, hi, color=c, alpha=0.15)

    axes[1].plot(timers, timers, color='black', linestyle='--', linewidth=1.0,
                 alpha=0.5, label='$T$ (threshold)', zorder=1)

    setup_ax(axes[0], 'Mode Oscillation Rate',   '% Episodes (D→U→D)', timers)
    axes[0].set_ylim([-2, 50])
    setup_ax(axes[1], r'Mean $L_{safe}(s_0)$ vs Timer',
             'Steps (safe BFS distance at $s_0$)', timers)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(cddp_models),
               bbox_to_anchor=(0.5, -0.06), fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, f'grid{grid_id}_phase4_switching_quality.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 3: Pareto Trajectory Frontier ─────────────────────────────────────

def plot_pareto_trajectory(grid_id, data, out_dir, main_models):
    """Trajectory frontier: each model is a curve across all T values.

    x-axis: P(excess > 0) %
    y-axis: success %

    Points are connected in T order (tight→easy, i.e., left→right along x-axis
    typically).  Aesthetic exactly matches performance plots:
      - Same COLORS / LINESTYLES, no markers on mean line
      - Ghost per-seed curves alpha=0.15, lw=1
      - Bootstrap CI band alpha=0.15 (computed along each axis independently)
      - Dilemma zone: points where mean L_safe > T shown on shaded half of plot
      - Timer labels for T_min, T_mid, T_max at small fontsize=7
    """
    models = [m for m in main_models if m in data]
    timers = sorted(int(k) for k in next(iter(data.values())).keys())

    # Determine dilemma timers for background shading (using CDDP-Oracle as ref)
    dilemma_ref = 'CDDP-Oracle' if 'CDDP-Oracle' in data else (models[0] if models else None)
    dilemma_ts  = set(compute_dilemma_xcoords(data, dilemma_ref, timers)) if dilemma_ref else set()

    fig, ax = plt.subplots(figsize=(9, 7))

    # Build dilemma zone shading in (x, y) space by collecting all model mean points
    # in dilemma zone and finding the x-range
    dilemma_x_vals = []
    annotations_dict = {}

    for model in models:
        c  = COLORS.get(model, 'black')
        ls = LINESTYLES.get(model, '-')

        # Collect mean x (P(excess>0)%) and y (success%) across timers
        x_means, y_means = [], []
        x_los, x_his     = [], []
        y_los, y_his     = [], []

        for T in timers:
            rows = data[model][str(T)].get('seeds_data', [])
            xs = [r.get('episodes_with_excess_rate', float('nan')) * 100 for r in rows
                  if not np.isnan(r.get('episodes_with_excess_rate', float('nan')))]
            ys = [r.get('feasible_success_rate', float('nan')) * 100 for r in rows
                  if not np.isnan(r.get('feasible_success_rate', float('nan')))]

            if xs and ys:
                mx, my = np.mean(xs), np.mean(ys)
                xlo, xhi = bootstrap_ci(xs)
                ylo, yhi = bootstrap_ci(ys)
            else:
                mx = data[model][str(T)].get('mean_excess_incidence',
                     data[model][str(T)].get('mean_episodes_with_excess_rate', float('nan'))) * 100
                my = data[model][str(T)].get('mean_feasible_success_rate', float('nan')) * 100
                xlo, xhi, ylo, yhi = mx, mx, my, my

            x_means.append(mx); y_means.append(my)
            x_los.append(xlo);  x_his.append(xhi)
            y_los.append(ylo);  y_his.append(yhi)

            if T in dilemma_ts and not np.isnan(mx):
                dilemma_x_vals.append(mx)

        # Ghost per-seed trajectories (one curve per seed across all T)
        n_seeds = len(data[model][str(timers[0])].get('seeds_data', []))
        for si in range(n_seeds):
            xs_seed, ys_seed = [], []
            for T in timers:
                rows = data[model][str(T)].get('seeds_data', [])
                if si < len(rows):
                    xs_seed.append(rows[si].get('episodes_with_excess_rate', float('nan')) * 100)
                    ys_seed.append(rows[si].get('feasible_success_rate', float('nan')) * 100)
            if xs_seed:
                ax.plot(xs_seed, ys_seed, color=c, linestyle=ls, alpha=0.12, linewidth=0.8)

        # Mean trajectory line (no markers)
        ax.plot(x_means, y_means, color=c, linestyle=ls, linewidth=2.5,
                alpha=1.0, marker='', label=model)

        # Bootstrap CI bands as shaded ellipse-like regions
        # Since we are in 2D, we draw x CI and y CI as cross-hair errors on mean curve
        ax.fill_betweenx(y_means, x_los, x_his, color=c, alpha=0.10)
        ax.fill_between(x_means,  y_los,  y_his, color=c, alpha=0.10)

        # Collect points for T_min, T_mid, T_max annotations
        label_idxs = {0: timers[0], len(timers) // 2: timers[len(timers) // 2],
                      len(timers) - 1: timers[-1]}
        for idx, T in label_idxs.items():
            xi, yi = x_means[idx], y_means[idx]
            if not (np.isnan(xi) or np.isnan(yi)):
                key = (T, round(xi), round(yi))
                if key not in annotations_dict:
                    annotations_dict[key] = []
                annotations_dict[key].append({'x': xi, 'y': yi, 'c': c, 'model': model})

    # Render annotations side-by-side if they share the same T and proximate location
    for (T, rx, ry), items in annotations_dict.items():
        # Sort so the color order is predictable/consistent
        items.sort(key=lambda i: MAIN_MODELS_G1.index(i['model']) if i['model'] in MAIN_MODELS_G1 else MAIN_MODELS_G2.index(i['model']))
        
        for i, item in enumerate(items):
            # Base offset spreads out radially away from the curve
            xytext = (25 + i * 35, 25 + i * 15)
            
            # Custom overrides to aggressively prevent overlap
            if item['model'] == 'PPO-Baseline-Finetuned':
                if T == 8:
                    xytext = (-35, 45)
                elif T == 12:
                    xytext = (30, -35)
            elif item['model'] == 'CDDP-Oracle' and T == 3:
                xytext = (15, -45)
                
            ax.annotate(f'T{T}', (item['x'], item['y']), fontsize=10, fontweight='bold', color=item['c'], alpha=0.9,
                        xytext=xytext, textcoords='offset points',
                        arrowprops=dict(arrowstyle="-|>", color=item['c'], lw=0.6, alpha=0.7, shrinkA=0, shrinkB=1))

    # Dilemma zone shading: background column for dilemma x-range
    if dilemma_x_vals:
        x_lo_d = max(0, min(dilemma_x_vals) - 3)
        x_hi_d = min(105, max(dilemma_x_vals) + 3)
        ax.axvspan(x_lo_d, x_hi_d, color='#FFCCCC', alpha=0.25, zorder=0, label='_nolegend_')

    ax.set_xlabel('P(excess > 0)  %', fontsize=13)
    ax.set_ylabel('Success %',         fontsize=13)
    ax.set_title(f'Grid {grid_id} — Pareto Trajectory Frontier\n'
                 f'(each curve = model across all $T$ values; arrows: tight→easy)',
                 fontsize=14, fontweight='bold', pad=25)
    ax.set_xlim([0, 60])
    ax.set_ylim([60, 100])
    ax.grid(True, alpha=0.3)

    # Legend — same placement as performance plots
    handles, labels = ax.get_legend_handles_labels()
    if dilemma_x_vals:
        patch = mpatches.Patch(color='#FFCCCC', alpha=0.5, label='Dilemma zone\n($L_{safe}>T$)')
        handles.append(patch); labels.append(patch.get_label())
    ax.legend(handles, labels, loc='lower right', fontsize=10)

    plt.tight_layout()
    out = os.path.join(out_dir, f'grid{grid_id}_phase4_pareto.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Summary Report ────────────────────────────────────────────────────────────

def write_summary(grid_results):
    os.makedirs(OUT_ROOT, exist_ok=True)
    out = os.path.join(OUT_ROOT, 'summary.md')
    lines = ['# Phase 4 v2 — Paper-Ready Evaluation Summary\n',
             '## Configuration\n',
             '- **Seeds:** 15  |  **Episodes/seed:** 200\n',
             '- **Episode seeds:** fixed per (seed, grid), shared across all T and agents\n',
             '- **Oracle:** PlannerOracleAgent (BFS deontic + Dijkstra utilitarian)\n',
             '- **Baseline:** PPO-Baseline-Finetuned (G1 + G2)\n\n']

    for grid_id, data in grid_results.items():
        if data is None:
            continue
        timers = sorted(int(k) for k in next(iter(data.values())).keys())
        lines.append(f'\n## Grid {grid_id} (T={timers[0]}–{timers[-1]})\n')

        # L_safe drift check
        if 'CDDP-Oracle' in data:
            l_safe_means = []
            for T in timers:
                rows = data['CDDP-Oracle'][str(T)].get('seeds_data', [])
                vals = [r.get('l_safe_s0', float('nan')) for r in rows
                        if not np.isnan(r.get('l_safe_s0', float('nan')))]
                if vals:
                    l_safe_means.append(np.mean(vals))
            if l_safe_means:
                drift = max(l_safe_means) - min(l_safe_means)
                status = '✅' if drift < 0.5 else '⚠️'
                lines.append(f'**L_safe drift across T:** {drift:.3f} steps {status}\n\n')

        # Per-model table at tightest T
        tight_T = timers[0]
        lines.append(f'### Tightest timer (T={tight_T})\n')
        lines.append('| Model | Success% | Viols | Excess | P(excess>0)% |\n')
        lines.append('|-------|----------|-------|--------|-------------|\n')
        for model in MAIN_MODELS_G1:
            if model not in data or str(tight_T) not in data[model]:
                continue
            e = data[model][str(tight_T)]
            suc  = e.get('mean_feasible_success_rate', float('nan')) * 100
            viols = e.get('mean_total_violations', e.get('mean_violation_rate', float('nan')))
            exc  = e.get('mean_excess_violations', e.get('mean_excess_violation_rate', float('nan')))
            pex  = e.get('mean_excess_incidence', e.get('mean_episodes_with_excess_rate', float('nan'))) * 100
            lines.append(f'| {model} | {suc:.1f} | {viols:.3f} | {exc:.3f} | {pex:.1f} |\n')

        # Oracle vs Prospective regret at tight T
        if 'CDDP-Oracle' in data and 'CDDP-Prospective' in data:
            o_viols = data['CDDP-Oracle'][str(tight_T)].get('mean_violation_rate', float('nan'))
            p_viols = data['CDDP-Prospective'][str(tight_T)].get('mean_violation_rate', float('nan'))
            if not np.isnan(o_viols) and not np.isnan(p_viols):
                lines.append(f'\n**Regret to Oracle at T={tight_T}:** {p_viols - o_viols:+.3f} violations\n')

    with open(out, 'w') as f:
        f.write(''.join(lines))
    print(f"\nSummary written to: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Generating Phase 4 v2 Paper-Ready Plots ===")

    grid_results = {}
    for grid_id, main_models in [(1, MAIN_MODELS_G1), (2, MAIN_MODELS_G2)]:
        data = load_results(grid_id)
        grid_results[grid_id] = data
        if data is None:
            continue

        out_dir = os.path.join(OUT_ROOT, f"grid{grid_id}")
        os.makedirs(out_dir, exist_ok=True)

        timers = sorted(int(k) for k in next(iter(data.values())).keys())
        present = [m for m in main_models if m in data]
        print(f"\nGrid {grid_id} | timers={timers} | models={present}")

        plot_performance(grid_id, data, out_dir, main_models)
        plot_switching_quality(grid_id, data, out_dir, main_models)
        plot_pareto_trajectory(grid_id, data, out_dir, main_models)

    write_summary(grid_results)
    print(f"\nDone. Outputs in: {OUT_ROOT}")


if __name__ == "__main__":
    main()
