#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6_analyze_results.py

- FEATURE (Restored): Re-introduced R² vs. Data Percentage scaling plots for
  each target, along with a summative bar chart comparing performance at 100% data.
- FEATURE (New): Generates Bayesian optimization convergence plots, showing
  how the model's performance improves with each hyperparameter trial.
- FEATURE (New): Creates dedicated markdown reports explaining the Bayesian
  optimization process and providing deeper analysis of feature importances.
- UPDATE (Expanded): Predicted-vs-actual scatter plots and residual analysis
  are now generated for ALL targets, not just a subset.
"""

import os
import logging
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import json

# Suppress benign warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------
# Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Input/Output Paths
METRICS_DIR = "metrics"
REPORTING_DIR = "reporting"
RESIDUALS_DIR = os.path.join(METRICS_DIR, "residuals")

os.makedirs(REPORTING_DIR, exist_ok=True)

# --- Define explicit order for plots and tables ---
TARGET_ORDER = [
    "mrna_half_life", "mrna_max_half_life_delta", "mrna_min_half_life_delta",
    "log2_tpm_abundance", "protein_half_life", "log2_ppm_abundance"
]

# Aesthetics
sns.set_theme(style="whitegrid", palette="viridis")
VARIANT_PALETTE = {
    "baseline": "#d1d1d1",
    "embeddings_only": "#440154",
}

# ------------------------------
# Data Loading
# ------------------------------

def load_data(filename, directory=METRICS_DIR):
    """Generic data loader with error handling."""
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        logging.error(f"Required file not found: {path}. Please run the training script first.")
        return None
    return pd.read_csv(path)

# ------------------------------
# Explanation & Report Generation
# ------------------------------

def generate_bayesian_optimization_report():
    """Generates a markdown file explaining Bayesian optimization plots."""
    logging.info("Generating Bayesian Optimization explanation report...")
    content = """
# Understanding the Bayesian Optimization Convergence Plots

The plots titled "Bayesian Optimization Convergence" illustrate how the hyperparameter tuning process found progressively better models.

## How to Read the Graph

-   **X-axis (Optimization Call):** This represents each "step" or "trial" in the optimization process. For each step, the optimizer chooses a new set of hyperparameters to try.
-   **Y-axis (Best R² Found):** This shows the best cross-validated R² score discovered *up to that point*.

An ideal convergence plot shows the R² value rapidly increasing in the early calls and then plateauing. This indicates that the optimizer has successfully explored the hyperparameter space and settled on a high-performing region.

## What are "Training Steps" in Bayesian Optimization?

Unlike neural networks that train over thousands of *epochs* or *steps* with gradient descent, the "training" in this context is different. Each **"call"** or **"step"** on the x-axis represents:

1.  **Selecting Hyperparameters:** The Bayesian optimizer picks a promising set of hyperparameters (e.g., `n_estimators`, `learning_rate`).
2.  **Full Model Training & Cross-Validation:** An XGBoost model is trained from scratch with these hyperparameters on the training data, and its performance is evaluated using 3-fold group cross-validation.
3.  **Recording the Score:** The average R² from the cross-validation is recorded.
4.  **Updating the Belief:** The optimizer uses this new result to update its internal probability model of which hyperparameter sets are most likely to yield the best results.

This cycle repeats for the total number of calls (`N_OPTIMIZATION_CALLS` = 25 in the script). The plot tracks the best score found throughout this entire search process.
"""
    with open(os.path.join(REPORTING_DIR, "bayesian_optimization_explained.md"), "w") as f:
        f.write(content)
    logging.info(f"Saved Bayesian optimization report to {REPORTING_DIR}/")


def generate_feature_analysis_report():
    """Generates a markdown file explaining feature importance."""
    logging.info("Generating Feature Importance explanation report...")
    content = """
# Understanding the Feature Importance Plot

The "Aggregated Feature Importance" plot shows the relative contribution of the three different molecular embeddings (DNA, RNA, and Protein) to the XGBoost model's predictions.

## How to Read the Graph

-   **X-axis (Prediction Target):** The biological value the model is trying to predict (e.g., protein half-life).
-   **Y-axis (Feature Importance):** A score (specifically, "gain" or "total gain") that represents how much a feature contributes to improving the model's accuracy. A higher bar means that feature was more influential.
-   **Colors (Feature Group):** The bars are segmented by the source of the embedding.

## Important Considerations

1.  **Relationship to Embedding Size:** Feature importance scores from tree-based models like XGBoost can be influenced by the number of features. The embeddings have different dimensions (DNA: 768, RNA: 768, Protein: 320). Because the DNA and RNA embeddings have more features, they have more "opportunities" to be selected for splits in the decision trees, which can naturally inflate their aggregate importance scores relative to the smaller protein embedding. **This plot does not normalize for the size of the embedding vector.**

2.  **CodonBERT's Download Source:** The fact that the CodonBERT model (for RNA embeddings) was downloaded from a different source (GitHub) than the others (Hugging Face) has **no impact** on this analysis. Once the embeddings are generated, they are simply numerical arrays. The training script is agnostic to their origin.

3.  **Interpretation:** A high importance for a specific embedding type suggests that the sequence information from that molecule is a strong predictor for the target variable. For example, if the 'protein' bar is high for predicting 'protein_half_life', it implies the amino acid sequence is a key determinant.
"""
    with open(os.path.join(REPORTING_DIR, "feature_analysis_report.md"), "w") as f:
        f.write(content)
    logging.info(f"Saved feature analysis report to {REPORTING_DIR}/")


# ------------------------------
# Plotting Suite
# ------------------------------

def plot_performance_scaling(results_df: pd.DataFrame):
    """(Restored) Generates R^2 vs. Data Percentage plots for each target."""
    logging.info("Generating performance scaling plots...")
    results_df['data_pct'] = results_df['data_fraction'].str.extract(r'(\d+)pct_data').astype(int)

    for target in TARGET_ORDER:
        plt.figure(figsize=(10, 6))
        target_df = results_df[results_df['target'] == target]
        
        if target_df.empty:
            continue

        sns.lineplot(data=target_df, x='data_pct', y='holdout_r2', hue='variant', marker='o', palette=VARIANT_PALETTE)
        
        plt.title(f'Model Performance vs. Training Data Size\nTarget: {target}', fontsize=16)
        plt.xlabel('Percentage of Training Data Used', fontsize=12)
        plt.ylabel('Holdout R²', fontsize=12)
        plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter())
        plt.ylim(bottom=max(-1, target_df['holdout_r2'].min() - 0.1), top=min(1, target_df['holdout_r2'].max() + 0.1))
        plt.legend(title='Model Variant')
        plt.tight_layout()
        
        save_path = os.path.join(REPORTING_DIR, f"performance_scaling_{target}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    logging.info(f"Saved performance scaling plots to {REPORTING_DIR}/")

def plot_summary_performance(results_df: pd.DataFrame):
    """(New) Generates a summative bar chart of performance at 100% data."""
    logging.info("Generating summary performance comparison plot...")
    df_100 = results_df[results_df['data_fraction'] == "100pct_data"].copy()

    if df_100.empty:
        logging.warning("No data for 100% fraction found. Skipping summary plot.")
        return

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=df_100, x='target', y='holdout_r2', hue='variant', order=TARGET_ORDER, palette=VARIANT_PALETTE)
    
    plt.title('Model Comparison: Holdout R² at 100% Training Data', fontsize=16, pad=20)
    plt.xlabel('Prediction Target', fontsize=12)
    plt.ylabel('Holdout R²', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.legend(title='Model Variant', loc='upper right')

    # --- NEW: Set Y-axis limits to focus on the XGBoost model's performance ---
    max_r2 = df_100[df_100['variant'] == 'embeddings_only']['holdout_r2'].max()
    plt.ylim(bottom=-0.1, top=max_r2 + 0.1)

    # --- NEW: Add numerical labels to each bar ---
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9) if p.get_height() > 0 else (0, -12), 
                   textcoords = 'offset points',
                   fontsize=9)

    plt.tight_layout()

    save_path = os.path.join(REPORTING_DIR, "performance_summary_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Saved summary performance plot to {save_path}")


def plot_predicted_vs_actual(results_df: pd.DataFrame):
    """(Expanded) Generates predicted vs. actual scatter plots for all targets."""
    logging.info("Generating predicted vs. actual scatter plots for ALL targets...")
    df_100 = results_df[results_df['data_fraction'] == "100pct_data"].copy()

    for target in TARGET_ORDER:
        target_subset_df = df_100[df_100['target'] == target]
        if target_subset_df.empty:
            continue
        
        best_variant_row = target_subset_df.sort_values('holdout_r2', ascending=False).iloc[0]
        variant = best_variant_row['variant']
        r2 = best_variant_row['holdout_r2']
        mse = best_variant_row['holdout_mse']

        residuals_file = os.path.join(RESIDUALS_DIR, "100pct_data", variant, f"{target}__holdout_predictions.csv")
        if not os.path.exists(residuals_file):
            logging.warning(f"Residuals file not found for {target} ({variant}). Skipping plot.")
            continue
        
        pred_df = pd.read_csv(residuals_file)
        
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=pred_df, x='y_true', y='y_pred', alpha=0.5, color=VARIANT_PALETTE.get(variant, "#333"))
        
        min_val = min(pred_df['y_true'].min(), pred_df['y_pred'].min())
        max_val = max(pred_df['y_true'].max(), pred_df['y_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        
        plt.title(f"Predicted vs. Actual: {target}\n(Best Variant: {variant})", fontsize=16)
        plt.xlabel("Actual Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}\nMSE = {mse:.3f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        plt.tight_layout()
        
        save_path = os.path.join(REPORTING_DIR, f"predicted_vs_actual_{target}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    logging.info(f"Saved predicted vs. actual plots to {REPORTING_DIR}/")


def plot_feature_importances(importances_df: pd.DataFrame):
    """Generates a bar plot of aggregated feature importances."""
    logging.info("Generating feature importance plot...")
    if importances_df is None: return

    df_100 = importances_df[importances_df['data_fraction'] == '100pct_data'].copy()
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_100, x='target', y='importance', hue='feature_group',
                order=TARGET_ORDER, hue_order=['dna', 'rna', 'protein'])
    plt.title("Aggregated Feature Importance by Embedding Type (100% Data)", fontsize=16)
    plt.xlabel("Prediction Target", fontsize=12)
    plt.ylabel("Feature Importance (Gain)", fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    save_path = os.path.join(REPORTING_DIR, "feature_importances.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Saved feature importance plot to {save_path}")


def plot_optimization_convergence(history_df: pd.DataFrame):
    """(New) Plots the convergence of the Bayesian optimization."""
    logging.info("Generating Bayesian optimization convergence plots...")
    if history_df is None: return
    
    # We only care about the 100% data fraction for this analysis
    df_100 = history_df[history_df['data_fraction'] == '100pct_data'].copy()
    df_100['r2'] = -df_100['objective_value']
    
    # Calculate the best R^2 found so far at each step
    df_100['best_r2_so_far'] = df_100.groupby('target')['r2'].cummax()

    for target in TARGET_ORDER:
        target_df = df_100[df_100['target'] == target]
        if target_df.empty:
            continue

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=target_df, x='call_number', y='best_r2_so_far', marker='.')
        
        plt.title(f'Bayesian Optimization Convergence\nTarget: {target}', fontsize=16)
        plt.xlabel('Optimization Call', fontsize=12)
        plt.ylabel('Best R² Found (Cross-Validation)', fontsize=12)
        plt.grid(True, which='both', linestyle='--')
        plt.tight_layout()
        
        save_path = os.path.join(REPORTING_DIR, f"optimization_convergence_{target}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    logging.info(f"Saved optimization convergence plots to {REPORTING_DIR}/")


def analyze_residuals(results_df: pd.DataFrame):
    """(Expanded) Identifies and reports genes with the largest prediction errors for all targets."""
    logging.info("Analyzing prediction residuals for ALL targets...")
    report_content = "## Residual Analysis: Top Prediction Errors\n\n"
    df_100 = results_df[results_df['data_fraction'] == "100pct_data"].copy()

    for target in TARGET_ORDER:
        target_subset_df = df_100[df_100['target'] == target]
        if target_subset_df.empty:
            continue
            
        best_variant_row = target_subset_df.sort_values('holdout_r2', ascending=False).iloc[0]
        variant = best_variant_row['variant']

        residuals_file = os.path.join(RESIDUALS_DIR, "100pct_data", variant, f"{target}__holdout_predictions.csv")
        if not os.path.exists(residuals_file):
            continue
        
        pred_df = pd.read_csv(residuals_file)
        pred_df['error'] = pred_df['y_pred'] - pred_df['y_true']
        
        over_predicted = pred_df.nlargest(5, 'error')
        under_predicted = pred_df.nsmallest(5, 'error')

        report_content += f"### Target: {target} (Best Variant: {variant})\n\n"
        report_content += "**Top 5 Over-predicted Genes:**\n"
        report_content += over_predicted[['gene_identifier', 'y_true', 'y_pred', 'error']].to_markdown(index=False, floatfmt=".3f") + "\n\n"
        report_content += "**Top 5 Under-predicted Genes:**\n"
        report_content += under_predicted[['gene_identifier', 'y_true', 'y_pred', 'error']].to_markdown(index=False, floatfmt=".3f") + "\n\n"

    with open(os.path.join(REPORTING_DIR, "residual_analysis.md"), "w") as f:
        f.write(report_content)
    logging.info(f"Saved residual analysis to {REPORTING_DIR}/residual_analysis.md")


def generate_performance_tables(results_df: pd.DataFrame):
    """Generates markdown tables for R² and MSE for all models at 100% data."""
    logging.info("Generating performance metric tables...")
    df_100 = results_df[results_df['data_fraction'] == "100pct_data"].copy()
    
    if df_100.empty:
        logging.warning("No data for 100% fraction found. Skipping performance tables.")
        return

    r2_pivot = df_100.pivot_table(index='target', columns='variant', values='holdout_r2').reindex(TARGET_ORDER)
    r2_markdown = r2_pivot.to_markdown(floatfmt=".4f")
    
    mse_pivot = df_100.pivot_table(index='target', columns='variant', values='holdout_mse').reindex(TARGET_ORDER)
    mse_markdown = mse_pivot.to_markdown(floatfmt=".4f")
    
    report_content = "## Model Performance Metrics (100% Data)\n\n"
    report_content += "### Coefficient of Determination (R²)\n\n"
    report_content += r2_markdown + "\n\n"
    report_content += "### Mean Squared Error (MSE)\n\n"
    report_content += mse_markdown + "\n"

    with open(os.path.join(REPORTING_DIR, "performance_metrics_summary.md"), "w") as f:
        f.write(report_content)
    logging.info(f"Saved performance metric tables to {REPORTING_DIR}/")


# ------------------------------
# Main Execution
# ------------------------------

def main():
    """Main function to run all analysis and generate plots."""
    logging.info("--- Starting Full Analysis and Reporting Pipeline ---")
    
    results_df = load_data("bayesian_optimization_summary.csv")
    importances_df = load_data("feature_importances.csv")
    history_df = load_data("optimization_history.csv")

    if results_df is None:
        logging.error("Could not load core results. Exiting.")
        return

    # --- Generate Reporting Suite ---
    generate_performance_tables(results_df)
    generate_bayesian_optimization_report()
    generate_feature_analysis_report()
    
    plot_performance_scaling(results_df)
    plot_summary_performance(results_df)
    plot_predicted_vs_actual(results_df)
    
    if importances_df is not None:
        plot_feature_importances(importances_df)
        
    if history_df is not None:
        plot_optimization_convergence(history_df)
        
    analyze_residuals(results_df)

    logging.info("\n--- Reporting suite generated successfully in 'reporting/' directory. ---")

if __name__ == "__main__":
    main()


