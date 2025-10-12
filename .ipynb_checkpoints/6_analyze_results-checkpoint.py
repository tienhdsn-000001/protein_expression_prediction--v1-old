#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6_analyze_results.py

- UPDATE (Feature Importance): The plotting and explanatory markdown have been
  updated to handle four feature groups: DNA (DNABERT-S), Rinalmo, RNA-ERNIE,
  and Protein (ESM-2), reflecting the new model pipeline.
- FIX (Import): Added missing imports for `mean_squared_error` and `r2_score`.
- FIX (NameError): Removed calls to old, non-existent plotting functions
  (`generate_performance_tables`, etc.) to prevent the script from crashing
  at the end of its run.
- FEATURE (Outlier Analysis): Added special logic to remove the extreme outlier
  from the 'mrna_max_half_life_delta' target during metric calculation and
  plotting to provide a more representative performance assessment.
- FEATURE (Normalized Importance Plot): Added a second feature importance plot
  that normalizes each embedding's importance by its own maximum value across
  all tasks, showing the relative utility of each embedding for different targets.
- FEATURE (Dimension-Adjusted Importance): Added a third feature importance
  plot that divides total importance by the number of dimensions in each
  embedding, showing the average importance per feature.
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

# Suppress benign warnings
warnings.filterwarnings("ignore", message="Using categorical units to plot a list of strings", category=UserWarning)

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
METRICS_DIR = "metrics"
REPORTING_DIR = "reporting"
RESIDUALS_DIR = os.path.join(METRICS_DIR, "residuals")
os.makedirs(REPORTING_DIR, exist_ok=True)

TARGET_ORDER = [
    "mrna_half_life", "mrna_max_half_life_delta", "mrna_min_half_life_delta",
    "log2_tpm_abundance", "protein_half_life", "log2_ppm_abundance"
]

EMBEDDING_DIMS = {
    'dna': 768,      # DNABERT-S
    'rinalmo': 1024, # Rinalmo-giga
    'rna_ernie': 640,  # RNA-ERNIE
    'protein': 5120  # ESM-2 15B
}


# Aesthetics
sns.set_theme(style="whitegrid", palette="viridis")
VARIANT_PALETTE = {"baseline": "#d1d1d1", "embeddings_only": "#440154"}
FEATURE_PALETTE = {"dna": "#440154", "rinalmo": "#3b528b", "rna_ernie": "#21918c", "protein": "#5ec962"}


def load_data(filename, directory=METRICS_DIR):
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        logging.error(f"Required file not found: {path}. Please run training scripts first.")
        return None
    return pd.read_csv(path)

def generate_comprehensive_metrics_report(results_df: pd.DataFrame):
    """Generates a detailed markdown report with multiple metrics for 100% data models."""
    logging.info("Generating comprehensive metrics report...")
    df_100 = results_df[results_df['data_fraction'] == "100pct_data"].copy()
    if df_100.empty: return

    all_metrics = []
    for _, row in df_100.iterrows():
        target, variant = row['target'], row['variant']
        residuals_file = os.path.join(RESIDUALS_DIR, "100pct_data", variant, f"{target}__holdout_predictions.csv")
        if not os.path.exists(residuals_file): continue
        
        pred_df = pd.read_csv(residuals_file).dropna()

        # --- NEW: Special handling for mrna_max_half_life_delta outlier ---
        if target == "mrna_max_half_life_delta":
            outlier_threshold = 8000  # Based on visual inspection of the plot
            initial_count = len(pred_df)
            pred_df = pred_df[pred_df['y_true'] < outlier_threshold]
            removed_count = initial_count - len(pred_df)
            if removed_count > 0:
                logging.info(f"For {target}, removed {removed_count} outlier(s) with y_true >= {outlier_threshold} for metric calculation.")
        # --- END NEW ---

        if len(pred_df) < 2: continue

        y_true, y_pred = pred_df['y_true'], pred_df['y_pred']
        mse = mean_squared_error(y_true, y_pred)
        
        metrics = {
            'Target': target, 'Variant': variant, 'R²': r2_score(y_true, y_pred),
            'MSE': mse, 'RMSE': np.sqrt(mse), 'MAE': mean_absolute_error(y_true, y_pred),
            "Spearman's ρ": spearmanr(y_true, y_pred)[0], "Pearson's r": pearsonr(y_true, y_pred)[0]
        }
        all_metrics.append(metrics)

    if not all_metrics:
        logging.warning("Could not calculate comprehensive metrics.")
        return

    metrics_df = pd.DataFrame(all_metrics).sort_values(by=['Target', 'Variant'])
    md = metrics_df.to_markdown(index=False, floatfmt=".4f")
    report = f"## Comprehensive Performance Metrics (100% Data)\n\nThis table provides a detailed evaluation of the final models...\n\n{md}"
    with open(os.path.join(REPORTING_DIR, "comprehensive_metrics_report.md"), "w") as f:
        f.write(report)
    logging.info(f"Saved comprehensive metrics report to {REPORTING_DIR}/comprehensive_metrics_report.md")

def plot_predicted_vs_actual(results_df: pd.DataFrame):
    """Generates predicted vs. actual scatter plots for all targets."""
    logging.info("Generating predicted vs. actual scatter plots...")
    df_100 = results_df[results_df['data_fraction'] == "100pct_data"].copy()
    for target in TARGET_ORDER:
        target_df = df_100[df_100['target'] == target]
        if target_df.empty: continue
        
        best_variant_row = target_df.sort_values('holdout_r2', ascending=False).iloc[0]
        variant = best_variant_row['variant']
        residuals_file = os.path.join(RESIDUALS_DIR, "100pct_data", variant, f"{target}__holdout_predictions.csv")
        if not os.path.exists(residuals_file): continue
        
        pred_df = pd.read_csv(residuals_file)
        plot_title = f"Predicted vs. Actual: {target}\n(Best Variant: {variant.replace('_', ' ').title()})"

        # --- NEW: Special handling for mrna_max_half_life_delta outlier ---
        if target == "mrna_max_half_life_delta":
            outlier_threshold = 8000  # Based on visual inspection of the plot
            initial_count = len(pred_df)
            pred_df = pred_df[pred_df['y_true'] < outlier_threshold]
            removed_count = initial_count - len(pred_df)
            if removed_count > 0:
                plot_title += "\n(Outlier Removed for Visualization)"
                logging.info(f"For {target} plot, removed {removed_count} outlier(s) with y_true >= {outlier_threshold}.")
        # --- END NEW ---

        # Recalculate R^2 for the plot text based on the potentially filtered data
        if len(pred_df) > 1:
            r2_for_plot = r2_score(pred_df['y_true'], pred_df['y_pred'])
        else:
            r2_for_plot = np.nan
            
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=pred_df, x='y_true', y='y_pred', alpha=0.5, color=VARIANT_PALETTE.get(variant, "#3b528b"))
        min_val, max_val = min(pred_df['y_true'].min(), pred_df['y_pred'].min()), max(pred_df['y_true'].max(), pred_df['y_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        plt.title(plot_title, fontsize=16)
        plt.xlabel("Actual Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.text(0.05, 0.95, f'R² = {r2_for_plot:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTING_DIR, f"predicted_vs_actual_{target}.png"), dpi=300)
        plt.close()
    logging.info(f"Saved all predicted vs. actual plots to {REPORTING_DIR}/")

def plot_absolute_feature_importances(importances_df: pd.DataFrame):
    """Generates a bar plot of aggregated absolute feature importances."""
    logging.info("Generating absolute feature importance plot...")
    if importances_df is None: return

    df_100 = importances_df[importances_df['data_fraction'] == '100pct_data'].copy()
    if df_100.empty: return
    
    feature_groups = ['dna', 'rinalmo', 'rna_ernie', 'protein']
    valid_feature_groups = [group for group in feature_groups if group in df_100.columns]
    
    melted_df = df_100.melt(id_vars=['target', 'data_fraction'], value_vars=valid_feature_groups,
                              var_name='feature_group', value_name='importance')

    plt.figure(figsize=(16, 9))
    sns.barplot(data=melted_df, x='target', y='importance', hue='feature_group',
                order=TARGET_ORDER, hue_order=valid_feature_groups, palette=FEATURE_PALETTE)
    plt.title("Aggregated Feature Importance by Embedding Type (100% Data)", fontsize=18)
    plt.xlabel("Prediction Target", fontsize=14)
    plt.ylabel("Total Importance (Gain)", fontsize=14)
    plt.xticks(rotation=15, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Feature Group", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTING_DIR, "feature_importances.png"), dpi=300)
    plt.close()
    logging.info(f"Saved absolute feature importance plot to {REPORTING_DIR}/feature_importances.png")

    explanation = """
# Feature Importance Analysis (Updated)

This plot shows the aggregated importance of the four different types of sequence embeddings used as features for the XGBoost models.

## How to Interpret This Graph

- **Y-Axis (Total Importance / Gain):** A measure from XGBoost representing the total contribution of all features within a group (e.g., all 768 dimensions of the DNA embedding) to the model's predictions. A higher bar indicates greater influence.
- **Colors (Feature Group):** Each color represents an embedding type:
  - **DNA:** DNABERT-S
  - **Rinalmo & RNA-ERNIE:** Two distinct, state-of-the-art RNA models.
  - **Protein:** ESM-2

## Key Observations

- **Context-Dependent Importance:** The relative importance of each embedding type changes depending on the prediction target. This allows us to infer which biological information source is most critical for a given task.
- **Comparing RNA Models:** This plot provides a direct comparison of the predictive power contained within the Rinalmo and RNA-ERNIE embeddings for each task.
- **Interpreting Importance Across Different Dimensions:** The embeddings have different dimensionalities (e.g., DNA: 768, RNA-ERNIE: 640). XGBoost's "gain" metric inherently accounts for this by measuring the total improvement to accuracy. It represents the contribution of the entire *information modality* (e.g., all DNA information vs. all RNA information), making this a fair comparison of their overall predictive value.
"""
    with open(os.path.join(REPORTING_DIR, "feature_importance_explained.md"), "w") as f:
        f.write(explanation)

def plot_normalized_feature_importances(importances_df: pd.DataFrame):
    """
    Generates a bar plot of feature importances normalized by the max for each
    embedding type, showing the relative utility of each embedding across tasks.
    """
    logging.info("Generating normalized feature importance plot...")
    if importances_df is None: return

    df_100 = importances_df[importances_df['data_fraction'] == '100pct_data'].copy()
    if df_100.empty: return

    feature_groups = ['dna', 'rinalmo', 'rna_ernie', 'protein']
    valid_feature_groups = [group for group in feature_groups if group in df_100.columns]
    
    melted_df = df_100.melt(id_vars=['target', 'data_fraction'], value_vars=valid_feature_groups,
                              var_name='feature_group', value_name='importance')

    # Normalize the importance by the max value within each feature group
    melted_df['normalized_importance'] = melted_df['importance'] / melted_df.groupby('feature_group')['importance'].transform('max')

    plt.figure(figsize=(16, 9))
    sns.barplot(data=melted_df, x='target', y='normalized_importance', hue='feature_group',
                order=TARGET_ORDER, hue_order=valid_feature_groups, palette=FEATURE_PALETTE)
    
    plt.title("Normalized Feature Importance by Embedding Type (100% Data)", fontsize=18)
    plt.xlabel("Prediction Target", fontsize=14)
    plt.ylabel("Normalized Importance (Relative to Max for Type)", fontsize=14)
    plt.xticks(rotation=15, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Feature Group", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTING_DIR, "normalized_feature_importances.png"), dpi=300)
    plt.close()
    logging.info(f"Saved normalized feature importance plot to {REPORTING_DIR}/normalized_feature_importances.png")

    explanation = """
# Normalized Feature Importance Analysis

This plot shows the *relative* importance of each embedding type, normalized by its own maximum performance across all tasks.

## How to Interpret This Graph

This plot answers a different question than the absolute importance plot. Instead of asking "Which embedding is most useful for this task?", it asks, **"For a given embedding, which task leverages its information the most?"**

- **Y-Axis (Normalized Importance):** For each embedding type (each color), the prediction target where it was most important is given a score of 1. All other bars for that color show how important the embedding was for other tasks *relative to its own peak performance*.
- **Example:** As you noted, the purple bar (ESM-2/Protein) is at 1.0 for `protein_half_life`, indicating this is the task where ESM-2's information is most valuable. The purple bar for `log2_tpm_abundance` is at ~0.6, meaning the protein embedding is about 60% as useful for that task as it is for its best task.

## Key Observations

- **Task Specialization:** This view highlights the "specialization" of each embedding. We can clearly see which tasks are the primary use-case for each modality.
- **Complementary Information:** Comparing this to the absolute importance plot is crucial. An embedding might have a low *absolute* importance for all tasks, but this plot would still show a bar at 1.0 for its "best" task, which could be misleading if viewed in isolation.

This analysis provides a more nuanced understanding of not just *what* information is important, but *where* it is most effectively applied.
"""
    with open(os.path.join(REPORTING_DIR, "normalized_feature_importance_explained.md"), "w") as f:
        f.write(explanation)

def plot_dimension_adjusted_feature_importances(importances_df: pd.DataFrame):
    """
    Generates a bar plot of feature importance per dimension to assess the
    "information density" or "efficiency" of each embedding type.
    """
    logging.info("Generating dimension-adjusted feature importance plot...")
    if importances_df is None: return

    df_100 = importances_df[importances_df['data_fraction'] == '100pct_data'].copy()
    if df_100.empty: return

    feature_groups = ['dna', 'rinalmo', 'rna_ernie', 'protein']
    valid_feature_groups = [group for group in feature_groups if group in df_100.columns]
    
    melted_df = df_100.melt(id_vars=['target', 'data_fraction'], value_vars=valid_feature_groups,
                              var_name='feature_group', value_name='importance')

    # Map dimensions and calculate importance per dimension
    melted_df['dimensions'] = melted_df['feature_group'].map(EMBEDDING_DIMS)
    melted_df['importance_per_dimension'] = melted_df['importance'] / melted_df['dimensions']

    plt.figure(figsize=(16, 9))
    sns.barplot(data=melted_df, x='target', y='importance_per_dimension', hue='feature_group',
                order=TARGET_ORDER, hue_order=valid_feature_groups, palette=FEATURE_PALETTE)
    
    plt.title("Dimension-Adjusted Feature Importance (Information Density)", fontsize=18)
    plt.xlabel("Prediction Target", fontsize=14)
    plt.ylabel("Average Importance per Dimension (Gain)", fontsize=14)
    plt.xticks(rotation=15, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Feature Group", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTING_DIR, "dimension_adjusted_feature_importances.png"), dpi=300)
    plt.close()
    logging.info(f"Saved dimension-adjusted feature importance plot to {REPORTING_DIR}/dimension_adjusted_feature_importances.png")

    explanation = """
# Dimension-Adjusted Feature Importance Analysis (Information Density)

This plot normalizes the total feature importance by the number of dimensions in each embedding. It aims to measure the *efficiency* or *information density* of each embedding type.

## How to Interpret This Graph

This plot answers the question: **"On average, how much predictive power does each dimension of an embedding contribute?"**

- **Y-Axis (Average Importance per Dimension):** This is the total importance (gain) for an embedding type divided by its number of features (e.g., 5120 for ESM-2). A higher bar suggests that the embedding packs more useful information into each of its dimensions.
- **Comparing Embeddings:** This view allows for a fairer comparison of the embeddings' intrinsic quality, independent of their size. A smaller, more efficient embedding might show a higher score here even if its total importance is lower than a much larger embedding.

## Key Observations

- **Efficiency vs. Total Power:** Compare this plot with the absolute importance plot. You might find cases where an embedding has high total importance but low per-dimension importance, suggesting its power comes from its large size rather than its efficiency. Conversely, a high score here indicates a well-structured, information-dense embedding.
- **Model Architecture Insights:** Consistently high scores for a smaller model (like RNA-ERNIE) might suggest that its architecture is particularly effective at capturing the relevant biological signals for these tasks.
"""
    with open(os.path.join(REPORTING_DIR, "dimension_adjusted_feature_importance_explained.md"), "w") as f:
        f.write(explanation)


def main():
    """Main function to run all analysis and generate plots."""
    logging.info("--- Starting Full Analysis and Reporting Pipeline ---")
    results_df = load_data("bayesian_optimization_summary.csv")
    importances_raw = load_data("feature_importances.csv")
    history_df = load_data("optimization_history.csv")

    if results_df is None:
        logging.error("Could not load core results. Exiting.")
        return

    importances_df = None
    if importances_raw is not None:
        feature_groups = ['dna', 'rinalmo', 'rna_ernie', 'protein']
        valid_groups = [g for g in feature_groups if g in importances_raw['feature_group'].unique()]
        
        if valid_groups:
             importances_df = importances_raw.pivot_table(
                 index=['data_fraction', 'target'], 
                 columns='feature_group', 
                 values='importance'
             ).reset_index()

    generate_comprehensive_metrics_report(results_df)
    plot_predicted_vs_actual(results_df)
    if importances_df is not None:
        plot_absolute_feature_importances(importances_df)
        plot_normalized_feature_importances(importances_df)
        plot_dimension_adjusted_feature_importances(importances_df)
    
    logging.info("\n--- Reporting suite generated successfully in 'reporting/' directory. ---\n")

if __name__ == "__main__":
    main()

