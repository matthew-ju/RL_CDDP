import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def parse_mean(val_str):
    if pd.isna(val_str) or val_str == '':
        return np.nan
    if isinstance(val_str, (int, float)):
        return float(val_str)
    return float(str(val_str).split('±')[0].strip())

def main():
    csv_path = os.path.join(project_root, 'final_paper_results', 'phase4_paper_ready_v2', 'paper_tables.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter for Grid 1 rows
    df = df[df['table_id'].astype(str).isin(['1', '1.0'])]
    
    # Parse the string columns into mean floats
    df['Success_Mean'] = df['Success%'].apply(parse_mean)
    df['Excess_Viols_Mean'] = df['Excess_Viols'].apply(parse_mean)
    df['T'] = df['T'].astype(float)
    
    models = ['CDDP-Oracle', 'CDDP-Prospective']
    colors = {'CDDP-Oracle': '#769958', 'CDDP-Prospective': '#2D6FBC'}
    labels = {'CDDP-Oracle': 'CDDP-Oracle (Correct Calibration)', 'CDDP-Prospective': 'CDDP-Prospective (Miscalibrated)'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for model in models:
        model_df = df[df['Model'] == model].sort_values('T')
        
        # Left Panel: Success Rate Drop
        ax1.plot(model_df['T'], model_df['Success_Mean'], label=labels[model], 
                color=colors[model], marker='o', linewidth=2.5, markersize=7)
                
        # Right Panel: Excess Violations Spike
        ax2.plot(model_df['T'], model_df['Excess_Viols_Mean'], label=labels[model], 
                color=colors[model], marker='s', linewidth=2.5, markersize=7)

    # Formatting Left Panel
    ax1.set_xlabel('Bomb Timer Initial ($T$)', fontsize=11)
    ax1.set_ylabel('Success Rate (%)', fontsize=11)
    ax1.set_title('(a) RL Mission Success (Tight Regime Consequence)', fontsize=12)
    ax1.grid(color='#EEEEEE', linestyle='-', linewidth=1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(2.5, 12.5)
    ax1.set_ylim(85, 101)  # Focus on the drop region
    ax1.legend(framealpha=0.9, edgecolor='#CCCCCC', fontsize=10)

    # Formatting Right Panel
    ax2.set_xlabel('Bomb Timer Initial ($T$)', fontsize=11)
    ax2.set_ylabel('Mean Excess Violations', fontsize=11)
    ax2.set_title('(b) RL Safety Failures (Easy Regime Consequence)', fontsize=12)
    ax2.grid(color='#EEEEEE', linestyle='-', linewidth=1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(2.5, 12.5)
    ax2.set_ylim(-0.05, 0.8)
    ax2.legend(framealpha=0.9, edgecolor='#CCCCCC', fontsize=10)

    plt.suptitle('Behavioral Consequences: True RL Deployment (Grid 1)', fontsize=14, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(os.path.join(project_root, 'results'), exist_ok=True)
    out_path = os.path.join(project_root, 'results', 'true_behavioral_consequences.png')
    plt.savefig(out_path, dpi=300)
    print(f"True behavioral consequences plot saved to {out_path}")

if __name__ == '__main__':
    main()
