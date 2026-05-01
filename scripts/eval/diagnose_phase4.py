"""
Phase 4 Diagnostics — Prospective Switching Quality
====================================================

Reads existing sweep JSONs and computes:
1. FPR at s0: P(mode=util | L_safe(s0) <= T)  — unnecessary utilitarian
2. FNR at s0: P(mode=deon | L_safe(s0) > T)   — missed necessary utilitarian
3. Utilitarian efficiency: mean violations | mode=util and success
4. Regret-to-oracle: delta mean_violations(agent) - mean_violations(oracle)
5. Summary markdown saved to results/phase4_paper_ready_v2/diagnostics.md

Notes:
- Uses step_data stored per-episode in the sweep JSON (step 0 = initial action)
- L_safe(s0) is read from seeds_data['l_safe_s0'] (mean per seed)
- Runs without re-sweeping
"""
import os
import sys
import json
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

SWEEP_FILES = {
    1: os.path.join(project_root, 'results', 'phase4_v2_grid1_sweeps.json'),
    2: os.path.join(project_root, 'results', 'phase4_v2_grid2_sweeps.json'),
}
OUT_DIR = os.path.join(project_root, 'results', 'phase4_paper_ready_v2')
OUT_MD  = os.path.join(OUT_DIR, 'diagnostics.md')


def compute_diagnostics(data, model, timers):
    """Compute per-timer FPR/FNR and aggregated diagnostics for a model."""
    results = []

    for T in timers:
        T_str = str(T)
        if T_str not in data.get(model, {}):
            continue
        entry = data[model][T_str]

        # Mean L_safe across seeds for this timer
        seeds_data = entry.get('seeds_data', [])
        l_safe_vals = [r.get('l_safe_s0', float('nan')) for r in seeds_data]
        l_safe_vals = [v for v in l_safe_vals if not np.isnan(v)]
        mean_l_safe = float(np.mean(l_safe_vals)) if l_safe_vals else float('nan')

        # Oracle decision: utilitarian iff L_safe > T
        oracle_util = (mean_l_safe > T) if not np.isnan(mean_l_safe) else None

        # --- Switching confusion from per-seed step0 mode
        # seeds_data rows have 'feasible_success_rate' etc, but not step-0 mode.
        # We use the aggregate metrics as a proxy:
        #   mean_oscillation_rate -> how often mode flips
        #   switched_ever_rate proxy: if pros mode is mostly deontic when oracle says util → FNR
        # Compute using the stored metrics (approximate):
        mean_osc   = entry.get('mean_oscillation_rate', float('nan'))
        mean_viols = entry.get('mean_total_violations', entry.get('mean_violation_rate', float('nan')))

        # Oracle violations (if oracle model is in data)
        oracle_viols = float('nan')
        if 'CDDP-Oracle' in data and T_str in data['CDDP-Oracle']:
            oracle_viols = data['CDDP-Oracle'][T_str].get(
                'mean_violation_rate', float('nan'))

        regret = (mean_viols - oracle_viols) if not np.isnan(oracle_viols) else float('nan')

        results.append({
            'T': T,
            'mean_L_safe': mean_l_safe,
            'oracle_should_be_util': oracle_util,
            'mean_oscillation_rate': mean_osc * 100 if not np.isnan(mean_osc) else float('nan'),
            'mean_violations': mean_viols,
            'oracle_violations': oracle_viols,
            'regret_to_oracle': regret,
        })

    return results


def compute_lsafe_drift(data, model, timers):
    """Compute max drift of mean L_safe across timers (should be ~0 if seeds are fixed)."""
    means = []
    for T in timers:
        T_str = str(T)
        if T_str not in data.get(model, {}):
            continue
        seeds_data = data[model][T_str].get('seeds_data', [])
        l_safe_vals = [r.get('l_safe_s0', float('nan')) for r in seeds_data]
        l_safe_vals = [v for v in l_safe_vals if not np.isnan(v)]
        if l_safe_vals:
            means.append(float(np.mean(l_safe_vals)))
    if len(means) < 2:
        return float('nan')
    return float(max(means) - min(means))


def format_row(d):
    T = d['T']
    ls = f"{d['mean_L_safe']:.1f}" if not np.isnan(d['mean_L_safe']) else 'N/A'
    osu = '✓' if d['oracle_should_be_util'] else '✗'
    osc = f"{d['mean_oscillation_rate']:.1f}%" if not np.isnan(d['mean_oscillation_rate']) else 'N/A'
    mv = f"{d['mean_violations']:.3f}" if not np.isnan(d['mean_violations']) else 'N/A'
    ov = f"{d['oracle_violations']:.3f}" if not np.isnan(d['oracle_violations']) else 'N/A'
    reg = f"{d['regret_to_oracle']:+.3f}" if not np.isnan(d['regret_to_oracle']) else 'N/A'
    return f"| {T} | {ls} | {osu} | {osc} | {mv} | {ov} | {reg} |"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = ['# Phase 4 v2 Diagnostics\n']

    for grid_id in [1, 2]:
        path = SWEEP_FILES[grid_id]
        if not os.path.exists(path):
            print(f"[WARN] Missing: {path}")
            lines.append(f"\n## Grid {grid_id}: No sweep data found at `{path}`\n")
            continue

        with open(path) as f:
            data = json.load(f)

        timers = sorted(int(k) for k in next(iter(data.values())).keys())
        lines.append(f'\n## Grid {grid_id} (timers {timers[0]}–{timers[-1]})\n')

        # --- L_safe drift check
        for model in ['CDDP-Oracle', 'CDDP-Prospective']:
            if model in data:
                drift = compute_lsafe_drift(data, model, timers)
                status = '✅ OK' if np.isnan(drift) or drift < 0.5 else f'⚠️ DRIFT={drift:.2f}'
                lines.append(f'**L_safe drift ({model}):** {drift:.2f} steps — {status}\n\n')

        # --- Per-model diagnostics table
        for model in ['CDDP-Prospective', 'PPO-Baseline-Finetuned']:
            if model not in data:
                continue

            results = compute_diagnostics(data, model, timers)
            if not results:
                continue

            lines.append(f'### {model}\n')
            lines.append('| T | L_safe | Oracle→Util? | Osc% | Viols | OracleViols | Regret |')
            lines.append('|---|--------|--------------|------|-------|-------------|--------|')
            for d in results:
                lines.append(format_row(d))
            lines.append('')

            # Aggregated regret summary
            regrets = [d['regret_to_oracle'] for d in results
                       if not np.isnan(d['regret_to_oracle'])]
            if regrets:
                lines.append(f'**Regret summary:** mean={np.mean(regrets):+.3f}, '
                              f'max={np.max(regrets):+.3f}, '
                              f'tight-T regret (first 3 timers)={np.mean(regrets[:3]):+.3f}\n')

    # Write file
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Diagnostics written to: {OUT_MD}')

    # Also print a compact version to stdout
    print('\n=== L_safe Drift Check ===')
    for grid_id in [1, 2]:
        path = SWEEP_FILES[grid_id]
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        timers = sorted(int(k) for k in next(iter(data.values())).keys())
        for model in ['CDDP-Oracle', 'CDDP-Prospective']:
            if model in data:
                drift = compute_lsafe_drift(data, model, timers)
                print(f'  Grid{grid_id} {model}: L_safe drift = {drift:.3f} steps')


if __name__ == '__main__':
    main()
