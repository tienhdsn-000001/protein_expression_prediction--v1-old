
don't include/
tmp_preds/

*.pkl
*.npy" > .gitignore
git init
git init -b main
git add .
git commit -m "recurated data and delta half life models"
  git config --global user.email "tien.hdns@gmail.com"
  git config --global user.name "hudst-000001"
git commit -m "recurated data and delta half life models"
git remote add origin [https://github.com/tienhdsn-000001/protein_expression_prediction.git](https://github.com/tienhdsn-000001/protein_expression_prediction.git)
git remote add origin https://github.com/tienhdsn-000001/protein_expression_prediction.git
git push -u origin main
git pull origin main --allow-unrelated-histories
git push -u origin main
# Tell Git to use your version of every conflicting file
git checkout --ours .
# Stage these resolved files
git add .
# Finalize the merge with a commit
git commit -m "Resolved merge conflicts by accepting local versions"
git commit -m "recurated data and delta half life models"
git push -u origin main
rm -rf .git
git init -b main
touch .gitignore
echo "# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/

# Jupyter Notebook
.ipynb_checkpoints

# Project Specific
embeddings/
data/models/
models/
don't include/
tmp_preds/
*.pkl
*.npy" > .gitignore
git add .
rm -rf .git
git init -b main
touch .gitignore
echo "# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/

# Jupyter Notebook
.ipynb_checkpoints

# Project Specific
embeddings/
data/
protein_expression_prediction-main/
predictions/
models/
don't include/
tmp_preds/
*.pkl
*.npy
" > .gitignore
git add.
git add .
git rm -r --cached .
git add .
git add .gitignore
git commit -m "Add .gitignore file"
git add .
git commit -m "new data curation and delta half life models"
echo "# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/

# Jupyter & Environment Caches
.ipynb_checkpoints
.cache/
.config/
.npm/
.local/
.docker/
.triton/
.cupy/

# Project Specific Outputs
embeddings/
models/
tmp_preds/
data/
don't_include/

# Data & Large Files
*.pkl
*.npy" > .gitignore
cat gitignore
cat .gitignore
git add .gitignore
git commit -m "Add .gitignore file"
git reset --soft HEAD~1
git rm -r --cached .
git add .gitignore
git commit -m "Correct and update .gitignore"
git add .
git commit -m "new data curation and delta half life  model"
git remote add origin [https://github.com/tienhdsn-000001/protein_expression_prediction.git](https://github.com/tienhdsn-000001/protein_expression_prediction.git)
git remote add origin https://github.com/tienhdsn-000001/protein_expression_prediction.git
git push -u origin main
git push --help
git checkout --ours .
git add .
git pull origin main --allow-unrelated-histories
rm -rf .npm/
rm -rf models/
git pull origin main --allow-unrelated-histories
git checkout --ours .
git add .
git commit -m "new data curation and delta half life models"
git push -u origin main
rm -rf .npm/
rm -rf models/
rm -rf metrics/
rm -rf analysis_plots/
bash post_train.sh
bash 4_run_evaluation.sh
bash run_evaluation.sh
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sed -n 's/^blob //p' | sort --numeric-sort --key=2 | cut -d' ' -f1,2 | numfmt --field=2 --to=iec-i --suffix=B --padding=7 | tail -n 20
bash post_train.sh
rm -rf .git
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sed -n 's/^blob //p' | sort --numeric-sort --key=2 | cut -d' ' -f1,2 | numfmt --field=2 --to=iec-i --suffix=B --padding=7 | tail -n 20
bash post_train.sh
bash post_train.sh
bash post_train.sh
bash post_train.sh
bash post_train.sh
bash post_train.sh
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
    if not df_half_life.empty:;         plt.figure(figsize=(12, 7))
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
        plt.tight_layout()         save_path = os.path.join(PLOT_DIR, "summary_r2_half_life_delta.png")
        plt.savefig(save_path, dpi=300)
        plt.close()         logging.info(f"Saved plot to {save_path}")
    # Plot for Abundance Models
    logging.info("Generating R² summary plot for abundance models (100% data)...")
    df_abundance = df_100[df_100['target'].isin(ABUNDANCE_TARGETS)]
    if not df_abundance.empty:;         plt.figure(figsize=(12, 7))
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
        plt.tight_layout()         save_path = os.path.join(PLOT_DIR, "summary_r2_abundance.png")
        plt.savefig(save_path, dpi=300)
        plt.close()         logging.info(f"Saved plot to {save_path}")
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
    plt.close()     logging.info(f"Saved plot to {save_path}")
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
        if model_df.empty:;             continue         model_df['data_frac_val'] = model_df['data_frac_val'].astype(int)
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
        for p in ax.patches:;             ax.annotate(f"{p.get_height():.3f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9),
                        textcoords='offset points')
        plt.tight_layout()         safe_variant_name = variant.replace(" ", "_")
        save_path = os.path.join(individual_plots_dir, f"scaling_{target}_{safe_variant_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()     logging.info(f"Saved individual scaling plots to {individual_plots_dir}")
# ------------------------------
# Main Execution
# ------------------------------
def main():
    """Main function to run all analysis and generate plots."""
    logging.info("--- Starting Analysis Pipeline ---")
    results_df = load_all_results()
    if results_df.empty:;         logging.error("Could not load results. Exiting.")
        return
    plot_r2_by_target_and_variant(results_df)
    plot_r2_vs_data_fraction(results_df)
    plot_individual_model_performance_scaling(results_df)
    logging.info("--- Analysis script completed successfully. ---")
if __name__ == "__main__":;     main()
bash post_train.sh
rm -rf .git
git init -b main
touch .gitignore
# Populate it with a comprehensive list of rules
echo "# Python & Environment Caches
__pycache__/
*.pyc
.cache/
.config/
.npm/
.local/
.docker/
.triton/
.cupy/

# Project Specific Outputs & Data
embeddings/
models/


tmp_preds/
data/
don't_include/

*.pkl
*.npy
*.npz
" > .gitignore
git add .gitignore
git commit -m "Add .gitignore file"
git add .
git status
git commit -m "new data curation and delta half lives models"
git remote add origin https://github.com/tienhdsn-000001/protein_expression_prediction
git push -u origin main -f
pythyon 4_train_model.py
python 4_train_model.py
bash run_evaluation.sh
bash run_evaluation.sh
bash run_evaluation.sh
bash run_evaluation.sh
bash run_evaluation.sh
bash run_evaluation.sh
bash run_evaluation.sh
bash run_evaluation.sh
bash run_evaluation.sh
bash run_evaluation.sh
python 3_generate_embeddings.py
python 3_generate_embeddings.py
python 3_generate_embeddings.py
python 2_download_models.py
python 3_generate_embeddings.py
bash 1_setup_environment
bash 1_setup_environment.sh
python 3_generate_embeddings.py
python 3_generate_embeddings.py
bash 1_setup_environment.sh
python 3_generate_embeddings.py
bash 1a_setup_dnabert2_env.sh
bash 1b_setup_main_env.sh
bash 1a_setup_dnabert2_env.sh
bash 1b_setup_main_env.sh
bash run_pipeline.sh
pip install einops
bash shortcut.sh
bash 1a_setup_dnabert2_env.sh
bash run_pipeline.sh
bash run_evaluation.sh
bash run_evaluation.sh
bash run_evaluation.sh
bash run_evaluation.sh
bash run_evaluation.sh
bash post_train.sh
nano .gitignore
git init
git add .
git commit -m "Fixed data leakage with operon-level sorting. PPM is only predictable value"
git remote add origin https://github.com/tienhdsn-000001/protein_expression_prediction.git
git push -u origin master
git push -u origin main
echo .git
git push -u origin main
git push -u origin main --allow-unrelated-histories
git pull origin main --allow-unrelated-histories
git push -u origin main
git init
nano .gitignore
git add .
rm -rf .git
nano .git
print .git
prod .git
nano .git
rm -rf .git
git init
git add .
git commit -m "Data leakage fix. PPM prediction only success"
git remote add origin https://github.com/tienhdsn-000001/protein_expression_prediction.git
git push -u origin main
git push -u origin
git push -u origin master
bash run_evaluation.sh
bash run_evaluation.sh
bash post_train.sh
bash post_train.sh
bash post_train.sh
bash post_train.sh
bash post_train.sh
bash run_evaluation.sh
bash full.sh
bash shortcut.sh
bash shortcut.sh
bash run_evaluation.sh
bash post_train.sh
bash run_evaluation.sh
bash run_evaluation.sh
bash post_train.sh
nano .gitignore
git init -b main
git add .
git commit -m "Further optimization and reporting metrics"
git remote add origin https://github.com/tienhdsn-000001/protein_expression_prediction
git push -u origin main -f
rm -rf .git
nano .gitignore
nano .gitignore
git init -b main
git add .
git commit -m "Further optimization and reporting"
git remote add origin https://github.com/tienhdsn-000001/protein_expression_prediction
git push -u origin master -f
python 6_analyze_results.py
python 6_analyze_results.py
git add .
git commit -m "Pearson, Spearman, MAE, MSE, RMSE"
git push -u origin main -f
git push -u origin master
pip install multimolecule
bash full.sh
bash full.sh
bash full.sh
bash full.sh
hf auth login
python 2_download_model.py
python 2_download_models.py
python 2_download_models.py
python 3_generate_embeddings.py
pip install multimolecule
python 3_generate_embeddings.py
python 3_generate_embeddings.py
python 3_generate_embeddings.py
python 3_generate_embeddings.py
python 3_generate_embeddings.py
python 3_generate_embeddings.py
python 3_generate_embeddings.py
python 3_generate_embeddings.py
python 3_generate_embeddings.py --regenerate protein
python 3_generate_embeddings.py --regenerate protein
python 3_generate_embeddings.py --regenerate protein
python get_device()
python 3_generate_embeddings.py --regenerate protein
python 3_generate_embeddings.py --regenerate protein
bg
fg && next_command
bg
fg && run_evaluation.sh
bg 1 > /dev/null 2>&1
fg
bg 1 > /dev/null 2>&1
fg && ./run_evaluation.sh
bash run_evaluation.sh
bash run_evaluation.sh
bash run_evaluation.sh
python 4_train_model.py --fractions 100pct_data
bash temp.sh
bash post_train.sh
bash temp.sh
bash temp.sh
bash temp.sh
bash temp.sh
python 4b_continue_optimization.py
python 4b_continue_optimization.py
bash temp.sh
bash post_train.sh
bash post_train.sh
python 6_analyze_results.py
python 6_analyze_results.py
python 6_analyze_results.py
python 6_analyze_results.py
python 6_analyze_results.py
python 6_analyze_results.py
python 6_analyze_results.py
