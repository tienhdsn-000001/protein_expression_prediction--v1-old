#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6_analyze_results.py

- FINALIZATION: Suppressed a benign UserWarning from the underlying plotting
  library to ensure a clean final output after debugging confirmed that data
  types were being handled correctly by our script.
"""

import os
import logging
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Suppress the specific UserWarning from seaborn that persists despite correct data types.
warnings.filterwarnings("ignore", message="Using categorical units to plot a list of strings", category=UserWarning)

# ------------------------------
# Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Input paths
METRICS_DIR = "metrics"

# Output path
PLOT_DIR = "analysis_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Experiment Configuration
DATA_FRACTIONS = ["20pct_data", "40pct_data", "60pct_data", "80pct_data", "100pct_data"]
VARIANTS = ["embeddings_only", "augmented"]
HALF_LIFE_TARGETS = ["max_half_life_delta", "min_half_life_delta"]
ABUNDANCE_TARGETS = ["log2_ppm_abundance", "log2_tpm_abundance"]

# Aesthetics
sns.set_theme(style="whitegrid", palette="viridis")
VARIANT_PALETTE = {
    "embeddings_only": "#440154",
    "augmented": "#21918c",
}

# ------------------------------
# Data Loading
# ------------------------------

def load_all_results() -> pd.DataFrame:
    """Load Bayesian optimization summary and ensure correct data types."""
    summary_path = os.path.join(METRICS_DIR, "bayesian_optimization_summary.csv")
    if not os.path.exists(summary_path):
        logging.error(f"Summary file not found at {summary_path}. Please run training first.")
        return pd.DataFrame()

    df = pd.read_csv(summary_path)
    df['data_frac_val'] = df['data_fraction'].str.replace('pct_data', '', regex=False).astype(int)
    return df

# ------------------------------
# Plotting Functions
# ------------------------------

def plot_r2_by_target_and_variant(results_df: pd.DataFrame):
    """Bar plot of R² scores for the 100% data fraction, split by category."""
    df_100 = results_df[results_df['data_fraction'] == "100pct_data"].copy()
    df_100 = df_100[df_100['holdout_r2'].notna()]

    # Plot for Half-Life Delta Models
    logging.info("Generating R² summary plot for half-life delta models (100% data)...")
    df_half_life = df_100[df_100['target'].isin(HALF_LIFE_TARGETS)]
    if not df_half_life.empty:
        plt.figure(figsize=(12, 7))
        sns.barplot(
            data=df_half_life, x='target', y='holdout_r2', hue='variant',
            palette=VARIANT_PALETTE, order=HALF_LIFE_TARGETS
        )
        plt.title("Half-Life Delta Model Performance (100% Training Data)", fontsize=16)
        plt.xlabel("Prediction Target", fontsize=12)
        plt.ylabel("Holdout R-squared", fontsize=12)
        plt.xticks(rotation=10)
        plt.legend(title="Model Variant")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_path = os.path.join(PLOT_DIR, "summary_r2_half_life_delta.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Saved plot to {save_path}")

    # Plot for Abundance Models
    logging.info("Generating R² summary plot for abundance models (100% data)...")
    df_abundance = df_100[df_100['target'].isin(ABUNDANCE_TARGETS)]
    if not df_abundance.empty:
        plt.figure(figsize=(12, 7))
        sns.barplot(
            data=df_abundance, x='target', y='holdout_r2', hue='variant',
            palette=VARIANT_PALETTE, order=ABUNDANCE_TARGETS
        )
        plt.title("Abundance Model Performance (100% Training Data)", fontsize=16)
        plt.xlabel("Prediction Target", fontsize=12)
        plt.ylabel("Holdout R-squared", fontsize=12)
        plt.xticks(rotation=10)
        plt.legend(title="Model Variant")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_path = os.path.join(PLOT_DIR, "summary_r2_abundance.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Saved plot to {save_path}")


def plot_r2_vs_data_fraction(results_df: pd.DataFrame):
    """Line plot showing how R² changes with data fraction across all models."""
    logging.info("Generating R² vs. data fraction plot...")
    g = sns.relplot(
        data=results_df, x='data_frac_val', y='holdout_r2',
        hue='variant', style='variant', col='target',
        kind='line', col_wrap=2, height=4.5, aspect=1.2,
        palette=VARIANT_PALETTE, markers=True, dashes=False
    )
    g.fig.suptitle("Holdout R-squared vs. Training Data Fraction", fontsize=18, y=1.03)
    g.set_axis_labels("Training Data Fraction (%)", "Holdout R-squared")
    g.set_titles("Target: {col_name}")
    g.despine(left=True)
    save_path = os.path.join(PLOT_DIR, "summary_r2_vs_data_fraction.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Saved plot to {save_path}")


def plot_individual_model_performance_scaling(results_df: pd.DataFrame):
    """Generates a separate bar chart for each model to show performance scaling."""
    logging.info("Generating individual performance scaling plots for each model...")
    individual_plots_dir = os.path.join(PLOT_DIR, "individual_model_scaling")
    os.makedirs(individual_plots_dir, exist_ok=True)
    unique_models = results_df[['target', 'variant']].drop_duplicates()

    for _, row in unique_models.iterrows():
        target = row['target']
        variant = row['variant']
        model_df = results_df[(results_df['target'] == target) & (results_df['variant'] == variant)].copy()
        if model_df.empty:
            continue

        model_df['data_frac_val'] = model_df['data_frac_val'].astype(int)

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(
            data=model_df, x='data_frac_val', y='holdout_r2',
            palette='viridis', order=[20, 40, 60, 80, 100],
            hue='data_frac_val', legend=False
        )
        ax.set_title(f"Performance vs. Data Fraction\nTarget: {target} | Variant: {variant}", fontsize=14)
        ax.set_xlabel("Training Data Fraction (%)", fontsize=12)
        ax.set_ylabel("Holdout R-squared", fontsize=12)
        min_y = min(0, model_df['holdout_r2'].min())
        max_y = model_df['holdout_r2'].max()
        ax.set_ylim(bottom=min_y - abs(min_y * 0.1), top=max_y + abs(max_y * 0.1))
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9),
                        textcoords='offset points')
        plt.tight_layout()
        safe_variant_name = variant.replace(" ", "_")
        save_path = os.path.join(individual_plots_dir, f"scaling_{target}_{safe_variant_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    logging.info(f"Saved individual scaling plots to {individual_plots_dir}")

# ------------------------------
# Main Execution
# ------------------------------

def main():
    """Main function to run all analysis and generate plots."""
    logging.info("--- Starting Analysis Pipeline ---")
    results_df = load_all_results()

    if results_df.empty:
        logging.error("Could not load results. Exiting.")
        return

    plot_r2_by_target_and_variant(results_df)
    plot_r2_vs_data_fraction(results_df)
    plot_individual_model_performance_scaling(results_df)
    logging.info("--- Analysis script completed successfully. ---")

if __name__ == "__main__":
    main()

