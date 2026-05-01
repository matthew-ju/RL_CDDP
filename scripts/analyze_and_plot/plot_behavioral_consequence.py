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
    
    # Filter for the main data rows (table_id 1 and 2)
    df = df[df['table_id'].isin(['1', '2', 1, 2])]
    
    # Parse the string columns into mean floats
    df['Success_Mean'] = df['Success%'].apply(parse_mean)
    df['Excess_Viols_Mean'] = df['Excess_Viols'].apply(parse_mean)
    df['T'] = df['T'].astype(float)
    
    # We only care about the two CDDP variants
    models = ['CDDP-Oracle', 'CDDP-Prospective']
    colors = {'CDDP-Oracle': '#769958', 'CDDP-Prospective': '#2D6FBC'}
    labels = {'CDDP-Oracle': 'CDDP-Oracle (Correct Calibration)', 'CDDP-Prospective': 'CDDP-Prospective (Miscalibrated)'}
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    for row, metric in enumerate(['Success_Mean', 'Excess_Viols_Mean']):
        for col, grid in enumerate([1, 2]):
            ax = axs[row, col]
            grid_df = df[df['table_id'].astype(str) == str(grid)]
            
            for model in models:
                model_df = grid_df[grid_df['Model'] == model].sort_values('T')
                ax.plot(model_df['T'], model_df[metric], label=labels[model], 
                        color=colors[model], marker='o', linewidth=2, markersize=6)
            
            # Formatting
            ax.set_xlabel('Bomb Timer Initial ($T$)')
            ax.grid(color='#CCCCCC', linestyle='--', alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if row == 0:
                ax.set_ylabel('Success Rate (%)')
                ax.set_title(f'Grid {grid}: Mission Success (Tight Regimes)')
                # Focus on the tight regimes where success drops
                if grid == 1:
                    ax.set_xlim(2.5, 7.5)
                else:
                    ax.set_xlim(6.5, 12.5)
            else:
                ax.set_ylabel('Mean Excess Violations')
                ax.set_title(f'Grid {grid}: Excess Violations (Easy Regimes)')
                # Focus on easy regimes where premature softening happens
                if grid == 1:
                    ax.set_xlim(6.5, 12.5)
                else:
                    ax.set_xlim(11.5, 20.5)
                    
            if row == 0 and col == 0:
                ax.legend(framealpha=0.9, edgecolor='#CCCCCC')

    plt.suptitle('Empirical Behavioral Consequences of Stakes Miscalibration', fontsize=14, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    os.makedirs(os.path.join(project_root, 'results'), exist_ok=True)
    out_path = os.path.join(project_root, 'results', 'behavioral_consequences_empirical.png')
    plt.savefig(out_path, dpi=300)
    print(f"Plot successfully generated and saved to {out_path}")

if __name__ == '__main__':
    main()
