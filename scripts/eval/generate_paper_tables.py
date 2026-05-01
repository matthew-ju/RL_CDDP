import json
import os
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
V2_SWEEP_DIR = os.path.join(project_root, "results")

def load_results(grid_id):
    if grid_id == 1:
        path = os.path.join(V2_SWEEP_DIR, "phase4_v2_grid1_sweeps_debug.json")
    else:
        path = os.path.join(V2_SWEEP_DIR, "phase4_v2_grid2_sweeps.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"Warning: Missing {path}")
    return {}

data_g1 = load_results(1)
data_g2 = load_results(2)

MODELS = ['PPO-Baseline-Finetuned', 'Deontic-Oracle', 'CDDP-Oracle', 'CDDP-Prospective']
TIMERS_G1 = [3,4,5,6,7,8,9,10,11,12]
TIMERS_G2 = list(range(7, 21))

all_rows = []

def get_mean_std(entry, key, scale=1.0, fmt="{:.3f}"):
    m = entry.get(f'mean_{key}')
    s = entry.get(f'std_{key}')
    if m is None or np.isnan(m): return "NaN"
    m_scaled = m * scale
    if s is None or np.isnan(s) or s < 1e-5:
        return fmt.format(m_scaled)
    return f"{fmt.format(m_scaled)} ± {fmt.format(s * scale)}"

def extract_row(data, T, model):
    if model not in data or str(T) not in data[model]:
        return {}
    entry = data[model][str(T)]
    
    suc = entry.get('mean_feasible_success_rate', np.nan) * 100
    viols = entry.get('mean_total_violations', entry.get('mean_violation_rate', np.nan))
    exc = entry.get('mean_excess_violations', entry.get('mean_excess_violation_rate', np.nan))
    pex = entry.get('mean_excess_incidence', entry.get('mean_episodes_with_excess_rate', np.nan)) * 100
    osc = entry.get('mean_oscillation_rate', np.nan) * 100
    lsafe = entry.get('mean_l_safe_s0', np.nan)
    
    return {
        'Success%': get_mean_std(entry, 'feasible_success_rate', 100, "{:.1f}"),
        'Total_Viols': get_mean_std(entry, 'total_violations', 1, "{:.3f}") if 'mean_total_violations' in entry else get_mean_std(entry, 'violation_rate', 1, "{:.3f}"),
        'Excess_Viols': get_mean_std(entry, 'excess_violations', 1, "{:.3f}") if 'mean_excess_violations' in entry else get_mean_std(entry, 'excess_violation_rate', 1, "{:.3f}"),
        'P(excess>0)%': get_mean_std(entry, 'excess_incidence', 100, "{:.1f}") if 'mean_excess_incidence' in entry else get_mean_std(entry, 'episodes_with_excess_rate', 100, "{:.1f}"),
        'Oscillation%': get_mean_std(entry, 'oscillation_rate', 100, "{:.1f}"),
        'L_safe_s0': get_mean_std(entry, 'l_safe_s0', 1, "{:.2f}"),
        '_suc': suc,
        '_exc': exc
    }

# Table 1: Grid 1 summary
print("\n### Table 1: Per-model, per-T summary for Grid 1")
print("| T | Model | Success% | Total_Viols | Excess_Viols | P(excess>0)% | Oscillation% | L_safe_s0 |")
print("|---|-------|----------|-------------|--------------|--------------|--------------|-----------|")
for T in TIMERS_G1:
    for model in MODELS:
        row = extract_row(data_g1, T, model)
        if not row: continue
        all_rows.append({'table_id': 1, 'T': T, 'Model': model, 
                         'Success%': row['Success%'], 'Total_Viols': row['Total_Viols'], 
                         'Excess_Viols': row['Excess_Viols'], 'P(excess>0)%': row['P(excess>0)%'],
                         'Oscillation%': row['Oscillation%'], 'L_safe_s0': row['L_safe_s0']})
        print(f"| {T} | {model} | {row['Success%']} | {row['Total_Viols']} | {row['Excess_Viols']} | {row['P(excess>0)%']} | {row['Oscillation%']} | {row['L_safe_s0']} |")

# Table 2: Grid 2 summary
print("\n### Table 2: Per-model, per-T summary for Grid 2")
print("| T | Model | Success% | Total_Viols | Excess_Viols | P(excess>0)% | Oscillation% | L_safe_s0 |")
print("|---|-------|----------|-------------|--------------|--------------|--------------|-----------|")
for T in TIMERS_G2:
    for model in MODELS:
        row = extract_row(data_g2, T, model)
        if not row: continue
        all_rows.append({'table_id': 2, 'T': T, 'Model': model, 
                         'Success%': row['Success%'], 'Total_Viols': row['Total_Viols'], 
                         'Excess_Viols': row['Excess_Viols'], 'P(excess>0)%': row['P(excess>0)%'],
                         'Oscillation%': row['Oscillation%'], 'L_safe_s0': row['L_safe_s0']})
        print(f"| {T} | {model} | {row['Success%']} | {row['Total_Viols']} | {row['Excess_Viols']} | {row['P(excess>0)%']} | {row['Oscillation%']} | {row['L_safe_s0']} |")

# Table 3: Key comparison table
print("\n### Table 3: Key comparison table")
print("| Model | G1_T3_Suc | G1_T3_Exc | G1_T12_Suc | G1_T12_Exc | G1_PeakExc_T | G1_PeakExc_Val | G2_T7_Suc | G2_T7_Exc | G2_T20_Suc | G2_T20_Exc | G2_PeakExc_T | G2_PeakExc_Val |")
print("|-------|-----------|-----------|------------|------------|--------------|----------------|-----------|-----------|------------|------------|--------------|----------------|")
for model in MODELS:
    r_g1_3 = extract_row(data_g1, 3, model)
    r_g1_12 = extract_row(data_g1, 12, model)
    r_g2_7 = extract_row(data_g2, 7, model)
    r_g2_20 = extract_row(data_g2, 20, model)
    
    g1_exc_vals = [(T, extract_row(data_g1, T, model).get('_exc', -1)) for T in TIMERS_G1]
    g1_peak_T, g1_peak_val = max(g1_exc_vals, key=lambda x: x[1]) if g1_exc_vals else (None, "NaN")
    
    g2_exc_vals = [(T, extract_row(data_g2, T, model).get('_exc', -1)) for T in TIMERS_G2]
    g2_peak_T, g2_peak_val = max(g2_exc_vals, key=lambda x: x[1]) if g2_exc_vals else (None, "NaN")
    
    def fmt(v): return f"{v:.3f}" if isinstance(v, float) else str(v)
    
    row_data = {
        'table_id': 3, 'Model': model,
        'G1_TightT_Success(T=3)': r_g1_3.get('Success%', 'NaN'), 'G1_TightT_Excess(T=3)': r_g1_3.get('Excess_Viols', 'NaN'),
        'G1_EasyT_Success(T=12)': r_g1_12.get('Success%', 'NaN'), 'G1_EasyT_Excess(T=12)': r_g1_12.get('Excess_Viols', 'NaN'),
        'G1_PeakExcess_T': g1_peak_T, 'G1_PeakExcess_Value': fmt(g1_peak_val),
        'G2_TightT_Success(T=7)': r_g2_7.get('Success%', 'NaN'), 'G2_TightT_Excess(T=7)': r_g2_7.get('Excess_Viols', 'NaN'),
        'G2_EasyT_Success(T=20)': r_g2_20.get('Success%', 'NaN'), 'G2_EasyT_Excess(T=20)': r_g2_20.get('Excess_Viols', 'NaN'),
        'G2_PeakExcess_T': g2_peak_T, 'G2_PeakExcess_Value': fmt(g2_peak_val)
    }
    all_rows.append(row_data)
    print(f"| {model} | {row_data['G1_TightT_Success(T=3)']} | {row_data['G1_TightT_Excess(T=3)']} | {row_data['G1_EasyT_Success(T=12)']} | {row_data['G1_EasyT_Excess(T=12)']} | {row_data['G1_PeakExcess_T']} | {row_data['G1_PeakExcess_Value']} | {row_data['G2_TightT_Success(T=7)']} | {row_data['G2_TightT_Excess(T=7)']} | {row_data['G2_EasyT_Success(T=20)']} | {row_data['G2_EasyT_Excess(T=20)']} | {row_data['G2_PeakExcess_T']} | {row_data['G2_PeakExcess_Value']} |")

# Table 4: Pareto dominance summary
print("\n### Table 4: Pareto dominance summary (Strictly better Success AND Excess)")
print("| Model A (Dominates) | Model B (Dominated) | Grid 1 Dominations | Grid 2 Dominations | Total |")
print("|---------------------|---------------------|--------------------|--------------------|-------|")
for ma in MODELS:
    for mb in MODELS:
        if ma == mb: continue
        g1_dom = 0
        for T in TIMERS_G1:
            ra = extract_row(data_g1, T, ma)
            rb = extract_row(data_g1, T, mb)
            if not ra or not rb: continue
            if ra['_suc'] > rb['_suc'] and ra['_exc'] < rb['_exc']: g1_dom += 1
        g2_dom = 0
        for T in TIMERS_G2:
            ra = extract_row(data_g2, T, ma)
            rb = extract_row(data_g2, T, mb)
            if not ra or not rb: continue
            if ra['_suc'] > rb['_suc'] and ra['_exc'] < rb['_exc']: g2_dom += 1
            
        all_rows.append({'table_id': 4, 'Model_A': ma, 'Model_B': mb, 'G1_Doms': g1_dom, 'G2_Doms': g2_dom, 'Total': g1_dom + g2_dom})
        print(f"| {ma} | {mb} | {g1_dom} | {g2_dom} | {g1_dom + g2_dom} |")

# Table 5: CDDP-Prospective improvement over PPO
print("\n### Table 5: CDDP-Prospective improvement over PPO")
print("| Grid | T | Delta_Success | Delta_Excess | Rel_Excess_Reduction% | Flag_Worse |")
print("|------|---|---------------|--------------|-----------------------|------------|")
for grid_id, data_g, timers in [(1, data_g1, TIMERS_G1), (2, data_g2, TIMERS_G2)]:
    for T in timers:
        rc = extract_row(data_g, T, 'CDDP-Prospective')
        rp = extract_row(data_g, T, 'PPO-Baseline-Finetuned')
        if not rc or not rp: continue
        
        d_suc = rc['_suc'] - rp['_suc']
        d_exc = rc['_exc'] - rp['_exc']
        
        rel_exc = ((rp['_exc'] - rc['_exc']) / rp['_exc'] * 100) if rp['_exc'] > 0 else 0
        worse = "FLAG" if (d_suc < -0.1 or d_exc > 0.001) else ""
        
        all_rows.append({'table_id': 5, 'Grid': grid_id, 'T': T, 'Delta_Success': d_suc, 'Delta_Excess': d_exc, 'Rel_Excess_Reduction%': rel_exc, 'Flag_Worse': worse})
        print(f"| {grid_id} | {T} | {d_suc:+.1f} | {d_exc:+.3f} | {rel_exc:+.1f}% | {worse} |")

df = pd.DataFrame(all_rows)
out_csv = os.path.join(V2_SWEEP_DIR, 'paper_tables.csv')
df.to_csv(out_csv, index=False)
print(f"\nSaved all tables to: {out_csv}")
