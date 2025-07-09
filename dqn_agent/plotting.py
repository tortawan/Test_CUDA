# dqn_agent/plotting.py
import pandas as pd
import matplotlib.pyplot as plt
from .config import logger, TOTAL_EXPERIMENTS

def plot_all_results(df_results: pd.DataFrame, plot_path: str):
    """
    Plots and saves the training progress for all experiments, showing
    individual runs, the mean, and standard deviation.
    """
    if df_results.empty:
        logger.warning("Attempted to plot with an empty DataFrame. Skipping plot generation.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 8))
    
    # Plot individual experiment runs with transparency
    for exp_id in df_results['experiment_id'].unique():
        exp_df = df_results[df_results['experiment_id'] == exp_id]
        plt.plot(exp_df['episode'], exp_df['avg_eval_score'], 'o-', label=None, alpha=0.2, color='gray', markersize=4)

    # Calculate and plot the mean and standard deviation across all experiments
    agg_df = df_results.groupby('episode')['avg_eval_score'].agg(['mean', 'std']).reset_index()
    
    plt.plot(agg_df['episode'], agg_df['mean'], 'o-', color='blue', label='Mean Score Across All Experiments', markersize=5)
    
    plt.fill_between(
        agg_df['episode'],
        agg_df['mean'] - agg_df['std'],
        agg_df['mean'] + agg_df['std'],
        color='blue',
        alpha=0.2,
        label='Standard Deviation'
    )

    plt.title(f'Comparison of {TOTAL_EXPERIMENTS} Identical DQN Experiments on LunarLander-v3', fontsize=18, fontweight='bold')
    plt.xlabel('Episode Number', fontsize=14)
    plt.ylabel('Average Evaluation Score', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add a horizontal line for the target score
    target_score = df_results.get('target_score', [250])[0] # A bit of a hack to get a value
    plt.axhline(y=target_score, color='r', linestyle='--', label=f'Target Score ({target_score})')
    
    plt.tight_layout()
    
    try:
        plt.savefig(plot_path)
        logger.info(f"Comparative plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {plot_path}: {e}")
    finally:
        plt.close()