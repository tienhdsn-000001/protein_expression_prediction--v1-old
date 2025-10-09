#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6_analyze_results.py

- FIX (Import): Added missing imports for `mean_squared_error` and `r2_score`
  from sklearn.metrics to resolve a NameError during the comprehensive
  metric report generation.
- FEATURE (Comprehensive Metrics): Now generates a new markdown report,
  `comprehensive_metrics_report.md`, which includes R², MSE, RMSE, MAE,
  Spearman's ρ, and Pearson's r for the final 100% data models.
- FEATURE: Restored performance scaling plots (R² vs. Data Fraction).
- FEATURE: Expanded predicted-vs-actual scatter plots to all targets.
- FEATURE: Added explanatory markdown reports for feature importance and
  Bayesian optimization.
- UPDATE (Plotting): The main model comparison bar chart now has a clipped
  y-axis and numerical labels for improved readability.
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
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress benign warnings from seaborn
warnings.filterwarnings("ignore", message="Using categorical units to plot a list of strings", category=UserWarning)

# ------------------------------
# Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Input/Output Paths
METRICS_DIR = "metrics"
REPORTING_DIR = "reporting"
RESIDUALS_DIR = os.path.join(METRICS_DIR, "residuals")

os.makedirs(REPORTING_DIR, exist_ok=True)

# Define explicit order for plots and tables
TARGET_ORDER = [
    "mrna_half_life", "mrna_max_half_life_delta", "mrna_min_half_life_delta",
    "log2_tpm_abundance", "protein_half_life", "log2_ppm_abundance"
]
DATA_FRACTION_ORDER = ["20%", "40%", "60%", "80%", "100%"]

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
# Reporting Suite Generation
# ------------------------------

def generate_performance_tables(results_df: pd.DataFrame):
    """Generates markdown tables for R² and MSE for all models at 100% data."""
    logging.info("Generating performance metric tables...")
    df_100 = results_df[results_df['data_fraction'] == "100pct_data"].copy()
    
    if df_100.empty:
        logging.warning("No data for 100% fraction found. Skipping performance tables.")
        return

    # Pivot for R-squared
    r2_pivot = df_100.pivot_table(index='target', columns='variant', values='holdout_r2').reindex(TARGET_ORDER)
    r2_markdown = r2_pivot.to_markdown(floatfmt=".4f")
    
    # Pivot for MSE
    mse_pivot = df_100.pivot_table(index='target', columns='variant', values='holdout_mse').reindex(TARGET_ORDER)
    mse_markdown = mse_pivot.to_markdown(floatfmt=".4f")
    
    report_content = "## Model Performance Metrics (100% Data)\n\n"
    report_content += "### Coefficient of Determination (R²)\n\n"
    report_content += r2_markdown + "\n\n"
    report_content += "### Mean Squared Error (MSE)\n\n"
    report_content += mse_markdown + "\n"

    with open(os.path.join(REPORTING_DIR, "performance_metrics_summary.md"), "w") as f:
        f.write(report_content)
    logging.info(f"Saved performance metric tables to {REPORTING_DIR}/performance_metrics_summary.md")

def generate_comprehensive_metrics_report(results_df: pd.DataFrame):
    """
    Generates a detailed markdown report with R², MSE, RMSE, MAE, Spearman, and Pearson
    metrics for the 100% data models by loading the raw prediction files.
    """
    logging.info("Generating comprehensive metrics report with correlations and error metrics...")
    df_100 = results_df[results_df['data_fraction'] == "100pct_data"].copy()
    if df_100.empty:
        return

    all_metrics = []
    for _, row in df_100.iterrows():
        target = row['target']
        variant = row['variant']
        
        residuals_file = os.path.join(RESIDUALS_DIR, "100pct_data", variant, f"{target}__holdout_predictions.csv")
        if not os.path.exists(residuals_file):
            continue
            
        pred_df = pd.read_csv(residuals_file).dropna()
        if len(pred_df) < 2:
            continue

        y_true = pred_df['y_true']
        y_pred = pred_df['y_pred']

        mse = mean_squared_error(y_true, y_pred)
        
        # Calculate all metrics
        metrics = {
            'Target': target,
            'Variant': variant,
            'R²': r2_score(y_true, y_pred),
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': mean_absolute_error(y_true, y_pred),
            "Spearman's ρ": spearmanr(y_true, y_pred)[0],
            "Pearson's r": pearsonr(y_true, y_pred)[0]
        }
        all_metrics.append(metrics)

    if not all_metrics:
        logging.warning("Could not calculate comprehensive metrics. No valid residual files found.")
        return

    metrics_df = pd.DataFrame(all_metrics).sort_values(by=['Target', 'Variant'])
    metrics_markdown = metrics_df.to_markdown(index=False, floatfmt=".4f")
    
    report_content = "## Comprehensive Performance Metrics (100% Data)\n\n"
    report_content += "This table provides a detailed evaluation of the final models on the holdout set, including measures of quantitative accuracy (R², MSE, RMSE, MAE), ranking accuracy (Spearman's ρ), and linear correlation (Pearson's r).\n\n"
    report_content += metrics_markdown

    with open(os.path.join(REPORTING_DIR, "comprehensive_metrics_report.md"), "w") as f:
        f.write(report_content)
    logging.info(f"Saved comprehensive metrics report to {REPORTING_DIR}/comprehensive_metrics_report.md")


def plot_predicted_vs_actual(results_df: pd.DataFrame):
    """Generates predicted vs. actual scatter plots for all targets."""
    logging.info("Generating predicted vs. actual scatter plots...")
    df_100 = results_df[results_df['data_fraction'] == "100pct_data"].copy()

    for target in TARGET_ORDER:
        target_df = df_100[df_100['target'] == target]
        if target_df.empty:
            continue
        
        best_variant_row = target_df.sort_values('holdout_r2', ascending=False).iloc[0]
        variant = best_variant_row['variant']
        r2 = best_variant_row['holdout_r2']
        
        residuals_file = os.path.join(RESIDUALS_DIR, "100pct_data", variant, f"{target}__holdout_predictions.csv")
        if not os.path.exists(residuals_file):
            logging.warning(f"Residuals file not found for {target} ({variant}). Skipping plot.")
            continue
        
        pred_df = pd.read_csv(residuals_file)
        
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=pred_df, x='y_true', y='y_pred', alpha=0.5, color=VARIANT_PALETTE.get(variant, "#3b528b"))
        
        min_val = min(pred_df['y_true'].min(), pred_df['y_pred'].min())
        max_val = max(pred_df['y_true'].max(), pred_df['y_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        
        plt.title(f"Predicted vs. Actual: {target}\n(Best Variant: {variant.replace('_', ' ').title()})", fontsize=16)
        plt.xlabel("Actual Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        plt.tight_layout()
        
        save_path = os.path.join(REPORTING_DIR, f"predicted_vs_actual_{target}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    logging.info(f"Saved all predicted vs. actual plots to {REPORTING_DIR}/")

def plot_feature_importances(importances_df: pd.DataFrame):
    """Generates a bar plot of aggregated feature importances."""
    logging.info("Generating feature importance plot...")
    if importances_df is None: return

    df_100 = importances_df[importances_df['data_fraction'] == '100pct_data'].copy()
    if df_100.empty: return
    
    # Melt the dataframe to have feature groups as a single column for hue
    melted_df = df_100.melt(id_vars=['target', 'data_fraction'], 
                             value_vars=['dna', 'rna', 'protein'], 
                             var_name='feature_group', value_name='importance')

    plt.figure(figsize=(14, 8))
    sns.barplot(data=melted_df, x='target', y='importance', hue='feature_group',
                order=TARGET_ORDER, hue_order=['dna', 'rna', 'protein'])
    plt.title("Aggregated Feature Importance by Embedding Type (100% Data)", fontsize=16)
    plt.xlabel("Prediction Target", fontsize=12)
    plt.ylabel("Total Importance (Gain)", fontsize=12)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    save_path = os.path.join(REPORTING_DIR, "feature_importances.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Saved feature importance plot to {save_path}")

    # --- Generate Explanatory Markdown ---
    explanation = """
# Feature Importance Analysis

This plot shows the aggregated importance of the three different types of sequence embeddings (DNA, RNA, Protein) used as features for the XGBoost models.

## How to Interpret This Graph

- **Y-Axis (Total Importance / Gain):** This is a measure provided by XGBoost that represents the total contribution of all features within a group (e.g., all 768 dimensions of the DNA embedding) to the model's predictions. A higher bar means that the features from that embedding type were more influential in making accurate predictions.
- **X-Axis (Prediction Target):** Each group of bars corresponds to a different biological value the models were trained to predict (e.g., `mrna_half_life`).
- **Colors (Feature Group):** Each color represents one of the three embedding types.

## Key Observations

- **Context-Dependent Importance:** The relative importance of DNA, RNA, and protein embeddings changes depending on the prediction target. For example, protein sequence features might be highly predictive for `protein_half_life` but less so for `mrna_half_life`.
- **Relationship to Embedding Length:** The raw importance values are influenced by the number of features in each group. Since all three embeddings have the same dimensionality (768), the heights of the bars can be compared directly. If they had different lengths, we would need to consider this, but here a direct comparison is fair.
- **CodonBERT (RNA) Paradigm:** While the CodonBERT model was downloaded manually, its output is a numerical embedding of the same size and format as the others. From the perspective of the downstream XGBoost model, it is just another set of features. Therefore, its importance can be analyzed in exactly the same way as the DNA and protein features.
"""
    with open(os.path.join(REPORTING_DIR, "feature_importance_explained.md"), "w") as f:
        f.write(explanation)

def analyze_residuals(results_df: pd.DataFrame):
    """Identifies and reports genes with the largest prediction errors."""
    logging.info("Analyzing prediction residuals...")
    report_content = "## Residual Analysis: Top Prediction Errors\n\n"
    df_100 = results_df[results_df['data_fraction'] == "100pct_data"].copy()

    for target in TARGET_ORDER:
        target_df = df_100[df_100['target'] == target]
        if target_df.empty:
            continue
        
        best_variant_row = target_df.sort_values('holdout_r2', ascending=False).iloc[0]
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

def plot_performance_scaling(results_df: pd.DataFrame):
    """Generates plots showing R² vs. percentage of training data used."""
    logging.info("Generating performance scaling plots...")
    
    results_df['data_pct'] = results_df['data_fraction'].str.extract('(\d+)').astype(int)

    for target in TARGET_ORDER:
        plt.figure(figsize=(10, 6))
        
        target_df = results_df[results_df['target'] == target]
        if target_df.empty:
            plt.close()
            continue

        sns.lineplot(data=target_df, x='data_pct', y='holdout_r2', hue='variant',
                     marker='o', palette=VARIANT_PALETTE, style='variant')
        
        plt.title(f"Model Performance vs. Training Data Size\nTarget: {target}", fontsize=16)
        plt.xlabel("Percentage of Training Data Used", fontsize=12)
        plt.ylabel("Holdout R²", fontsize=12)
        plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter())
        plt.xticks(sorted(results_df['data_pct'].unique()))
        plt.legend(title="Model Variant")
        plt.grid(True, which='both', linestyle='--')
        plt.tight_layout()
        save_path = os.path.join(REPORTING_DIR, f"performance_scaling_{target}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    logging.info(f"Saved all performance scaling plots to {REPORTING_DIR}/")

def plot_summary_comparison(results_df: pd.DataFrame):
    """Generates a summary bar plot comparing baseline and embeddings models at 100% data."""
    logging.info("Generating summary comparison plot...")
    df_100 = results_df[results_df['data_fraction'] == '100pct_data'].copy()
    if df_100.empty: return

    # --- FIX: Clip the y-axis to a reasonable range for better visualization ---
    min_val = df_100[df_100['variant'] == 'embeddings_only']['holdout_r2'].min()
    y_min = min(min_val - 0.1, -0.1) # Set a floor but allow for slightly negative R²
    y_max = df_100['holdout_r2'].max() + 0.1

    plt.figure(figsize=(15, 8))
    ax = sns.barplot(data=df_100, x='target', y='holdout_r2', hue='variant',
                     order=TARGET_ORDER, palette=VARIANT_PALETTE)
    
    plt.title("Model Comparison: Holdout R² at 100% Training Data", fontsize=18)
    plt.xlabel("Prediction Target", fontsize=12)
    plt.ylabel("Holdout R²", fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.axhline(0, color='grey', linestyle='--')
    
    # --- FIX: Add numerical labels to each bar ---
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9 if p.get_height() >= 0 else -9), 
                    textcoords='offset points',
                    fontsize=9, color='black')
    
    ax.set_ylim(y_min, y_max)
    plt.legend(title="Model Variant")
    plt.tight_layout()
    save_path = os.path.join(REPORTING_DIR, "performance_summary_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Saved summary comparison plot to {save_path}")

def plot_optimization_convergence(history_df: pd.DataFrame):
    """Plots the convergence of the Bayesian optimization for each target."""
    logging.info("Generating Bayesian optimization convergence plots...")
    if history_df is None: return

    history_df['objective_value'] = -history_df['objective_value']
    
    # Calculate the running best score at each call
    history_df['best_r2_so_far'] = history_df.groupby(['target', 'data_fraction'])['objective_value'].cummax()

    df_100 = history_df[history_df['data_fraction'] == '100pct_data']

    for target in TARGET_ORDER:
        target_df = df_100[df_100['target'] == target]
        if target_df.empty: continue

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=target_df, x='call_number', y='best_r2_so_far', drawstyle='steps-post', color="#440154")

        plt.title(f"Bayesian Optimization Convergence\nTarget: {target}", fontsize=16)
        plt.xlabel("Optimization Call", fontsize=12)
        plt.ylabel("Best R² Found (Cross-Validation)", fontsize=12)
        plt.grid(True, which='both', linestyle='--')
        plt.tight_layout()
        
        save_path = os.path.join(REPORTING_DIR, f"optimization_convergence_{target}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    logging.info(f"Saved all optimization convergence plots to {REPORTING_DIR}/")

    # --- Generate Explanatory Markdown ---
    explanation = """
# Bayesian Optimization Explained

These plots illustrate the process of hyperparameter tuning for each model.

## How to Interpret the Convergence Plots

- **X-Axis (Optimization Call):** Represents each time the optimizer trained and evaluated a new model with a different set of hyperparameters (like `learning_rate`, `max_depth`, etc.).
- **Y-Axis (Best R² Found):** Shows the best cross-validation R² score found *up to that point*. The line will only ever go up or stay flat, as it tracks the best performance seen so far.
- **What is a "Training Step"?** In this context, a "training step" or "optimization call" is a full cycle of:
    1. The Bayesian optimizer picking a new set of hyperparameters it thinks might be promising.
    2. Training a complete XGBoost model from scratch using those parameters on folds of the training data.
    3. Evaluating the model on a validation fold.
    4. Repeating this across all cross-validation folds to get a robust R² score.
    5. Feeding that score back to the optimizer, which updates its internal "map" of the hyperparameter space.

A steep upward slope indicates the optimizer is quickly finding better parameter combinations. A long plateau means it is struggling to find improvements over the current best-known parameters.
"""
    with open(os.path.join(REPORTING_DIR, "bayesian_optimization_explained.md"), "w") as f:
        f.write(explanation)

# ------------------------------
# Main Execution
# ------------------------------

def main():
    """Main function to run all analysis and generate plots."""
    logging.info("--- Starting Full Analysis and Reporting Pipeline ---")
    
    results_df = load_data("bayesian_optimization_summary.csv")
    importances_raw = load_data("feature_importances.csv")
    history_df = load_data("optimization_history.csv")

    if results_df is None:
        logging.error("Could not load core results. Exiting.")
        return

    # Pivot importances for easier plotting
    importances_df = None
    if importances_raw is not None:
        importances_df = importances_raw.pivot_table(
            index=['data_fraction', 'target'], 
            columns='feature_group', 
            values='importance'
        ).reset_index()

    # --- Generate Reporting Suite ---
    generate_performance_tables(results_df)
    generate_comprehensive_metrics_report(results_df)
    plot_predicted_vs_actual(results_df)
    if importances_df is not None:
        plot_feature_importances(importances_df)
    analyze_residuals(results_df)
    plot_performance_scaling(results_df)
    plot_summary_comparison(results_df)
    plot_optimization_convergence(history_df)
    
    logging.info("\n--- Reporting suite generated successfully in 'reporting/' directory. ---\n")

if __name__ == "__main__":
    main()

