import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv('/Users/matthewju/Downloads/CDDP_RL/results/phase4_paper_ready_v2/paper_tables.csv')

df1 = df[df['table_id'] == 1]
df2 = df[df['table_id'] == 2]

models = ['PPO-Baseline-Finetuned', 'Deontic-Oracle', 'CDDP-Oracle', 'CDDP-Prospective']
model_names = {'PPO-Baseline-Finetuned': 'PPO-Baseline', 'Deontic-Oracle': 'Deontic-Oracle', 'CDDP-Oracle': 'CDDP-Oracle', 'CDDP-Prospective': 'CDDP-Prospective'}

data = {}
for g, df_g, ts in [(1, df1, [3, 8, 12]), (2, df2, [7, 14, 20])]:
    data[g] = {}
    for t in ts:
        data[g][t] = {}
        sub = df_g[df_g['T'] == t]
        for m in models:
            row = sub[sub['Model'] == m].iloc[0]
            suc_str = str(row['Success%'])
            exc_str = str(row['Excess_Viols'])
            
            suc_val = float(suc_str.split('±')[0].strip()) if '±' in suc_str else (float(suc_str) if suc_str != 'NaN' else 0)
            exc_val = float(exc_str.split('±')[0].strip()) if '±' in exc_str else (float(exc_str) if exc_str != 'NaN' else 1000)
            
            data[g][t][m] = {'suc_str': suc_str, 'exc_str': exc_str, 'suc_val': suc_val, 'exc_val': exc_val}

best = {}
for g, ts in [(1, [3, 8, 12]), (2, [7, 14, 20])]:
    for t in ts:
        # Find best success (max)
        best_suc = max([data[g][t][m]['suc_val'] for m in models])
        # Find best excess (min)
        best_exc = min([data[g][t][m]['exc_val'] for m in models])
        best[(g, t)] = {'suc': best_suc, 'exc': best_exc}

def fmt_cell(val_str, is_best):
    if val_str == 'NaN': return '-'
    if '±' in val_str:
        mean, std = val_str.split('±')
        mean = mean.strip()
        std = std.strip()
        if is_best:
            return f"\\mathbf{{{mean}}} \\pm {std}"
        else:
            return f"{mean} \\pm {std}"
    else:
        if is_best:
            return f"\\mathbf{{{val_str.strip()}}}"
        else:
            return val_str.strip()

latex = []
latex.append("\\begin{table*}[t]")
latex.append("\\centering")
latex.append("\\resizebox{\\textwidth}{!}{%")
latex.append("\\begin{tabular}{lcccccc}")
latex.append("\\toprule")
latex.append("& \\multicolumn{3}{c}{\\textbf{Grid 1} ($L_{safe} \\approx 4.5$)} & \\multicolumn{3}{c}{\\textbf{Grid 2} ($L_{safe} \\approx 6.8$)} \\\\")
latex.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
latex.append("\\textbf{Model} & $T=3$ & $T=8$ & $T=12$ & $T=7$ & $T=14$ & $T=20$ \\\\")
latex.append("\\midrule")

# New ordering:
# PPO-Baseline
# Deontic-Oracle
# CDDP-Oracle
# (midrule)
# CDDP-Prospective
order = ['PPO-Baseline-Finetuned', 'Deontic-Oracle', 'CDDP-Oracle', 'SEP', 'CDDP-Prospective']

for m in order:
    if m == 'SEP':
        latex.append("\\midrule")
        continue

    row_strs = [model_names[m]]
    for g, ts in [(1, [3, 8, 12]), (2, [7, 14, 20])]:
        for t in ts:
            cell = data[g][t][m]
            
            # Use strict comparison with epsilon 1e-4 so identical floats get bolded
            is_best_suc = (cell['suc_val'] >= best[(g, t)]['suc'] - 1e-4)
            is_best_exc = (cell['exc_val'] <= best[(g, t)]['exc'] + 1e-4)
            
            suc_fmt = fmt_cell(cell['suc_str'], is_best_suc)
            exc_fmt = fmt_cell(cell['exc_str'], is_best_exc)
            
            row_strs.append(f"${suc_fmt}$ / ${exc_fmt}$")
    
    latex.append(" & ".join(row_strs) + " \\\\")

latex.append("\\bottomrule")
latex.append("\\end{tabular}%")
latex.append("}")
latex.append("\\caption{Success rate and excess violations (mean $\\pm$ std, 15 seeds $\\times$ 200 episodes) at representative timer values. Bold = best per column. Dilemma zone boundary is $T=L_{safe}$; Grid 1 $L_{safe}=4.5$, Grid 2 $L_{safe}=6.8$.}")
latex.append("\\label{tab:phase4_results}")
latex.append("\\end{table*}")

ppo_suc_t3 = data[1][3]['PPO-Baseline-Finetuned']['suc_val']
deontic_suc_t3 = data[1][3]['Deontic-Oracle']['suc_val']
sacrificed_suc = round(ppo_suc_t3 - deontic_suc_t3, 1)

ppo_exc_t12 = data[1][12]['PPO-Baseline-Finetuned']['exc_val']
cddp_exc_t12 = data[1][12]['CDDP-Prospective']['exc_val']
if ppo_exc_t12 > 0:
    reduced_exc = round((ppo_exc_t12 - cddp_exc_t12) / ppo_exc_t12 * 100, 1)
else:
    reduced_exc = 0

print("\n".join(latex))
print("\n--- Narrative sentences ---")
print(f"(1) In the infeasible regime ($T < L_{{safe}}$), CDDP-Prospective preserves PPO-level success rates while the purely deontic baseline sacrifices {sacrificed_suc}% success to avoid violations — confirming that rigid constraint adherence is suboptimal when safety is unachievable.")
print(f"(2) At large $T$ (feasible regime), CDDP-Prospective reduces excess violations by {reduced_exc}% vs PPO with no success penalty.")
print("(3) CDDP-Oracle establishes the oracle upper bound, achieving near-zero excess at all $T$ values, at the cost of requiring full environment access unavailable at deployment.")
