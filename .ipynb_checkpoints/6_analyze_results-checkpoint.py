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
- FEATURE (Scaling Grid): Added a 2x3 faceted grid plot to show
  R² performance scaling vs. data size for all 6 targets.
- UPDATE (Data Loading): The script now dynamically builds the performance summary
  by scanning raw residual files instead of relying on a pre-made summary CSV,
  ensuring all plots are generated from the primary data source.
- FEATURE (Embeddings-Only Grid): Added a dedicated 2x3 scaling grid for only
  the embeddings_only models to provide a more granular view of their performance.
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
import glob
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

# NEW: Added target name map for the new scaling plot
TARGET_NAME_MAP = {
    'log2_tpm_abundance': 'log2(TPM) Abundance',
    'log2_ppm_abundance': 'log2(PPM) Abundance',
    'protein_half_life': 'Protein Half-life',
    'mrna_half_life': 'mRNA Half-life',
    'mrna_min_half_life_delta': 'mRNA Min Half-life Delta',
    'mrna_max_half_life_delta': 'mRNA Max Half-life Delta'
}

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
    """Loads data from the metrics directory."""
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

def plot_performance_scaling_grid(results_df: pd.DataFrame):
    """
    Generates and saves a 2x3 grid of plots showing how Holdout R² scales
    with data size, faceted by all 6 target variables.
    """
    logging.info("Generating performance scaling grid plot...")
    if results_df is None or results_df.empty:
        logging.warning("No results data to generate scaling plot.")
        return

    # --- Prepare data for plotting ---
    df_to_plot = results_df.copy()
    
    # 1. Convert data_fraction string (e.g., "20pct_data") to numeric
    df_to_plot['Percentage of Training Data Used'] = df_to_plot['data_fraction'].str.replace('pct_data', '').astype(int)
    
    # 2. Rename 'variant' for a prettier legend title
    df_to_plot = df_to_plot.rename(columns={'variant': 'Model Variant'})
    
    # 3. Map target names to prettier names for facet titles
    df_to_plot['Target Name'] = df_to_plot['target'].map(TARGET_NAME_MAP)
    
    # 4. Set the order of the facets
    target_name_order = [TARGET_NAME_MAP[t] for t in TARGET_ORDER if t in TARGET_NAME_MAP]

    # --- Create the faceted plot ---
    g = sns.relplot(
        data=df_to_plot,
        x='Percentage of Training Data Used',
        y='holdout_r2',
        hue='Model Variant',
        style='Model Variant',
        col='Target Name',
        kind='line',
        marker='o',
        col_wrap=3,
        height=4,
        aspect=1.2,
        legend='full',
        col_order=target_name_order,
        palette=VARIANT_PALETTE, # Use the existing palette
        facet_kws={'sharey': False, 'sharex': True} # Let Y-axis scale for each plot
    )
    
    g.fig.suptitle('Model Performance vs. Training Data Size', fontsize=20, y=1.03)
    g.set_axis_labels('Percentage of Training Data Used (%)', 'Holdout R²')
    g.set_titles("{col_name}")
    
    # Improve gridlines like the example image
    for ax in g.axes.flat:
        ax.grid(True, which='both', linestyle='--', linewidth=0.7)
        ax.set_facecolor('#f0f0f0') # Light gray background for contrast

    g.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the figure
    save_path = os.path.join(REPORTING_DIR, 'performance_scaling_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Generated performance scaling grid plot at: {save_path}")

# --- START: NEW ADDITIVE FUNCTION ---
def plot_embedding_only_scaling_grid(results_df: pd.DataFrame):
    """
    Generates a 2x3 grid plot showing R² scaling for ONLY embeddings_only models
    to provide a more granular view of their performance.
    """
    logging.info("Generating performance scaling grid plot for embeddings_only models...")
    if results_df is None or results_df.empty:
        logging.warning("No results data to generate embeddings_only scaling plot.")
        return

    # --- Prepare data for plotting ---
    # 1. Filter for only the embeddings_only variant
    df_to_plot = results_df[results_df['variant'] == 'embeddings_only'].copy()
    
    if df_to_plot.empty:
        logging.warning("No 'embeddings_only' variant data found to generate scaling plot.")
        return

    # 2. Convert data_fraction string to numeric
    df_to_plot['Percentage of Training Data Used'] = df_to_plot['data_fraction'].str.replace('pct_data', '').astype(int)
    
    # 3. Map target names for facet titles
    df_to_plot['Target Name'] = df_to_plot['target'].map(TARGET_NAME_MAP)
    
    # 4. Set the order of the facets
    target_name_order = [TARGET_NAME_MAP[t] for t in TARGET_ORDER if t in TARGET_NAME_MAP]

    # --- Create the faceted plot ---
    # Since we only have one variant, we remove hue and style for a cleaner look
    g = sns.relplot(
        data=df_to_plot,
        x='Percentage of Training Data Used',
        y='holdout_r2',
        col='Target Name',
        kind='line',
        marker='o',
        col_wrap=3,
        height=4,
        aspect=1.2,
        col_order=target_name_order,
        color=VARIANT_PALETTE["embeddings_only"], # Use the specific color
        facet_kws={'sharey': False, 'sharex': True}
    )
    
    g.fig.suptitle('Embeddings-Only Model Performance vs. Training Data Size', fontsize=20, y=1.03)
    g.set_axis_labels('Percentage of Training Data Used (%)', 'Holdout R²')
    g.set_titles("{col_name}")
    
    for ax in g.axes.flat:
        ax.grid(True, which='both', linestyle='--', linewidth=0.7)
        ax.set_facecolor('#f0f0f0')

    g.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(REPORTING_DIR, 'performance_scaling_grid_embeddings_only.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Generated embeddings-only performance scaling grid plot at: {save_path}")
# --- END: NEW ADDITIVE FUNCTION ---

def plot_performance_scaling(results_df: pd.DataFrame):
    """
    Generates individual plots showing how Holdout R² scales with data size
    for each target variable separately.
    """
    logging.info("Generating individual performance scaling plots...")
    if results_df is None or results_df.empty:
        logging.warning("No results data to generate individual scaling plots.")
        return

    # Prepare data for plotting
    df_to_plot = results_df.copy()
    df_to_plot['Percentage of Training Data Used'] = df_to_plot['data_fraction'].str.replace('pct_data', '').astype(int)
    df_to_plot = df_to_plot.rename(columns={'variant': 'Model Variant'})

    for target in TARGET_ORDER:
        plt.figure(figsize=(10, 6))
        subset_df = df_to_plot[df_to_plot['target'] == target]
        
        if subset_df.empty:
            logging.warning(f"No data for target '{target}' in individual scaling plot.")
            continue

        sns.lineplot(data=subset_df, x='Percentage of Training Data Used', y='holdout_r2', 
                     hue='Model Variant', style='Model Variant', marker='o', palette=VARIANT_PALETTE)
        
        target_name = TARGET_NAME_MAP.get(target, target.replace('_', ' ').title())
        plt.title(f'Performance Scaling for {target_name}', fontsize=16)
        plt.xlabel('Training Data Used (%)', fontsize=12)
        plt.ylabel('Holdout R²', fontsize=12)
        plt.legend(title='Model Variant')
        plt.grid(True, which='both', linestyle='--', linewidth=0.7)
        
        # Adjust y-axis to start near the minimum R^2 value but not below zero
        if not subset_df['holdout_r2'].empty:
            min_r2 = subset_df['holdout_r2'].min()
            plt.ylim(bottom=max(0, min_r2 - 0.1) if pd.notna(min_r2) else 0)

        plt.tight_layout()
        save_path = os.path.join(REPORTING_DIR, f'performance_scaling_{target}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
    logging.info(f"Saved all individual performance scaling plots to {REPORTING_DIR}/")

def plot_optimization_history(history_df: pd.DataFrame):
    """Plots the convergence of Bayesian optimization for each target."""
    logging.info("Generating optimization convergence plots...")
    if history_df is None or 'target_col' not in history_df.columns:
        logging.warning("Optimization history data is missing or malformed. Skipping plots.")
        return

    for target in TARGET_ORDER:
        if target not in history_df['target_col'].unique():
            continue
            
        plt.figure(figsize=(10, 6))
        subset_df = history_df[history_df['target_col'] == target].copy()
        
        if 'iteration' not in subset_df.columns:
            subset_df['iteration'] = range(1, len(subset_df) + 1)
        subset_df = subset_df.sort_values('iteration')
        
        # Calculate the best score found at each iteration
        best_scores = np.maximum.accumulate(subset_df['test_score'])
        
        plt.plot(subset_df['iteration'], best_scores, marker='o', linestyle='-', color="#440154")
        
        target_name = TARGET_NAME_MAP.get(target, target.replace('_', ' ').title())
        plt.title(f'Bayesian Optimization Convergence for {target_name}', fontsize=16)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Holdout R² Score Found', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7)
        plt.tight_layout()
        save_path = os.path.join(REPORTING_DIR, f"optimization_convergence_{target}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    logging.info(f"Saved all optimization convergence plots to {REPORTING_DIR}/")

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
... (explanation markdown) ...
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
... (explanation markdown) ...
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
... (explanation markdown) ...
"""
    with open(os.path.join(REPORTING_DIR, "dimension_adjusted_feature_importance_explained.md"), "w") as f:
        f.write(explanation)


def main():
    """Main function to run all analysis and generate plots."""
    logging.info("--- Starting Full Analysis and Reporting Pipeline ---")
    
    # --- MODIFIED: Build results_df by scanning all residual files ---
    logging.info("Scanning for residual files to build performance summary...")
    all_results = []
    # Use glob to find all holdout prediction files recursively within RESIDUALS_DIR
    residual_files = glob.glob(os.path.join(RESIDUALS_DIR, "**", "*__holdout_predictions.csv"), recursive=True)

    if not residual_files:
        logging.error(f"No holdout prediction files found in {RESIDUALS_DIR}. Cannot generate reports. Exiting.")
        return
        
    for f_path in residual_files:
        try:
            # e.g., metrics/residuals/20pct_data/baseline/target__holdout_predictions.csv
            parts = f_path.replace("\\", "/").split('/')
            data_fraction = parts[-3] # '20pct_data'
            variant = parts[-2]       # 'baseline' or 'embeddings_only'
            filename = parts[-1]
            target = filename.split('__')[0]

            pred_df = pd.read_csv(f_path).dropna(subset=['y_true', 'y_pred'])

            # Special outlier handling for mrna_max_half_life_delta
            if target == "mrna_max_half_life_delta":
                outlier_threshold = 8000
                pred_df = pred_df[pred_df['y_true'] < outlier_threshold]

            if len(pred_df) < 2:
                logging.warning(f"Not enough data points to calculate R² for {f_path}")
                continue

            r2 = r2_score(pred_df['y_true'], pred_df['y_pred'])
            
            all_results.append({
                'data_fraction': data_fraction,
                'variant': variant,
                'target': target,
                'holdout_r2': r2
            })
        except Exception as e:
            logging.warning(f"Could not process residual file {f_path}: {e}")
    
    if not all_results:
        logging.error("Failed to calculate any results from residual files. Exiting.")
        return
        
    results_df = pd.DataFrame(all_results)
    logging.info(f"Successfully compiled {len(results_df)} results from {len(residual_files)} residual files.")
    # --- END MODIFICATION ---

    # Load other necessary files
    importances_raw = load_data("feature_importances.csv")
    history_df = load_data("optimization_history.csv")

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

    # --- Call all reporting functions ---
    generate_comprehensive_metrics_report(results_df)
    plot_predicted_vs_actual(results_df)
    
    # Call the re-introduced scaling and convergence plot functions
    plot_performance_scaling(results_df)
    plot_optimization_history(history_df)
    
    # Call the performance scaling grid plot function
    plot_performance_scaling_grid(results_df)

    # --- NEW: Call the added function for embeddings_only scaling ---
    plot_embedding_only_scaling_grid(results_df)
    # --- END NEW ---
    
    if importances_df is not None:
        plot_absolute_feature_importances(importances_df)
        plot_normalized_feature_importances(importances_df)
        plot_dimension_adjusted_feature_importances(importances_df)
    
    logging.info("\n--- Reporting suite generated successfully in 'reporting/' directory. ---\n")

if __name__ == "__main__":
    main()

