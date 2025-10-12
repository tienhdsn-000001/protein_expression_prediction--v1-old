#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4_train_model.py

- FIX (CRITICAL, OOM): Added `max_bin=128` to the XGBoost parameters to
  reduce GPU memory usage during training. This prevents out-of-memory
  errors on tasks with a larger number of training samples, such as the
  abundance predictions.
- FEATURE (Run Customization): Added a '--targets' command-line argument.
  This allows for specifying which specific prediction targets to run,
  providing granular control for resuming failed jobs.
- FIX (Resumable Logic): The logic for resuming runs has been updated to
  correctly filter existing results based on both the --fractions and
  --targets arguments, ensuring data integrity.
"""

import os
import csv
import json
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

import xgboost as xgb
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not found. GPU acceleration disabled for XGBoost.")

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ANNOTATION_FILE = "Ecoli_Annotation_v3.tsv"
EMBEDDINGS_DIR = "embeddings"
MODELS_DIR = "models"
METRICS_DIR = "metrics"
RESIDUALS_DIR = os.path.join(METRICS_DIR, "residuals")
DATA_FRACTIONS = {
    "20pct_data": 0.20,
    "40pct_data": 0.40,
    "60pct_data": 0.60,
    "80pct_data": 0.80,
    "100pct_data": 1.0
}
HOLDOUT_FRACTION = 0.1
TARGETS = {
    "mrna_half_life": "mrna_half_life",
    "mrna_max_half_life_delta": "mrna_max_half_life_delta",
    "mrna_min_half_life_delta": "mrna_min_half_life_delta",
    "log2_ppm_abundance": "log2_ppm_abundance",
    "log2_tpm_abundance": "log2_tpm_abundance",
    "protein_half_life": "protein_half_life",
}
N_OPTIMIZATION_CALLS = 25
SEARCH_SPACE = [
    Integer(100, 1000, name='n_estimators'), Real(0.01, 0.3, name='learning_rate'),
    Integer(3, 10, name='max_depth'), Real(0.5, 1.0, name='subsample'),
    Real(0.5, 1.0, name='colsample_bytree'),
]
# --- FIX: Added max_bin=128 to reduce GPU memory usage ---
BASE_XGB_PARAMS = {
    'objective': 'reg:squarederror', 'tree_method': 'hist',
    'device': 'cuda' if CUPY_AVAILABLE else 'cpu', 'n_jobs': -1, 'random_state': 42,
    'max_bin': 128
}

# Helpers
def load_ecoli(path: str) -> pd.DataFrame:
    """Robustly loads the E. coli annotation TSV."""
    try:
        return pd.read_csv(
            path, sep="\t", engine="python", encoding="utf-16",
            dtype=str, na_filter=False, quotechar='"', quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="error",
        )
    except (UnicodeError, pd.errors.ParserError):
        logging.info("UTF-16 parsing failed. Falling back to UTF-8.")
        return pd.read_csv(
            path, sep="\t", engine="python", encoding="utf-8",
            dtype=str, na_filter=False, quotechar='"', quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="error",
        )

class ToCuPyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        if not CUPY_AVAILABLE: return X
        return cp.asarray(X) if not isinstance(X, cp.ndarray) else X

def to_numpy_if_cupy(a):
    if CUPY_AVAILABLE and isinstance(a, cp.ndarray): return cp.asnumpy(a)
    return np.asarray(a)

def build_model(**xgb_params) -> Pipeline:
    # Ensure integer parameters are correctly typed
    if 'n_estimators' in xgb_params:
        xgb_params['n_estimators'] = int(xgb_params['n_estimators'])
    if 'max_depth' in xgb_params:
        xgb_params['max_depth'] = int(xgb_params['max_depth'])
    return Pipeline([
        ('scaler', StandardScaler()),
        ('tocupy', ToCuPyTransformer()),
        ('xgb', xgb.XGBRegressor(**{**BASE_XGB_PARAMS, **xgb_params}))
    ])

def build_baseline_model() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ])

def export_holdout_predictions(df_meta, holdout_indices, target_name, data_fraction, variant, model, X_mat, y_series):
    """Calculates metrics and saves predictions for the holdout set."""
    # Find indices that are both in the holdout set and have valid entries in the metadata and series
    valid_indices_mask = np.isin(holdout_indices, df_meta.index) & np.isin(holdout_indices, y_series.index)
    valid_indices = holdout_indices[valid_indices_mask]
    
    y_holdout = y_series.loc[valid_indices]
    non_na_mask = y_holdout.notna()
    final_indices = y_holdout[non_na_mask].index.values
    
    if final_indices.size == 0:
        logging.warning(f"No valid holdout rows for {target_name}. Skipping.")
        return {'holdout_r2': np.nan, 'holdout_mse': np.nan}
        
    # Get features and true values for the final set of valid indices
    X_hold, y_hold = X_mat[final_indices], y_series.loc[final_indices].values
    y_pred = model.predict(X_hold)
    y_hold_np, y_pred_np = to_numpy_if_cupy(y_hold), to_numpy_if_cupy(y_pred)
    
    gene_id_col = next((c for c in ['Gene_Name', 'Protein_ID'] if c in df_meta.columns), None)
    gene_ids = df_meta.loc[final_indices, gene_id_col].values if gene_id_col else final_indices
    
    out_df = pd.DataFrame({'gene_identifier': gene_ids, 'y_true': y_hold_np, 'y_pred': y_pred_np})
    out_df['abs_error'] = np.abs(out_df['y_true'] - out_df['y_pred'])
    frac_dir = os.path.join(RESIDUALS_DIR, data_fraction, variant)
    os.makedirs(frac_dir, exist_ok=True)
    out_df.to_csv(os.path.join(frac_dir, f"{target_name}__holdout_predictions.csv"), index=False)
    
    holdout_r2 = r2_score(y_hold_np, y_pred_np)
    holdout_mse = mean_squared_error(y_hold_np, y_pred_np)
    logging.info(f"[{data_fraction}|{variant}|{target_name}] Holdout R²: {holdout_r2:.4f}, MSE: {holdout_mse:.4f}")
    
    return {'holdout_r2': float(holdout_r2), 'holdout_mse': float(holdout_mse)}

def get_feature_importances(model: Pipeline, dims: Dict) -> Dict:
    """Extracts feature importances and aggregates them by embedding type."""
    if 'xgb' not in model.named_steps:
        return {}
    importances = model.named_steps['xgb'].feature_importances_
    
    return {
        'dna': float(importances[:dims['dna']].sum()),
        'rinalmo': float(importances[dims['dna']:dims['dna'] + dims['rinalmo']].sum()),
        'rna_ernie': float(importances[dims['dna'] + dims['rinalmo']:dims['dna'] + dims['rinalmo'] + dims['rna_ernie']].sum()),
        'protein': float(importances[dims['dna'] + dims['rinalmo'] + dims['rna_ernie']:].sum())
    }

def create_objective_function(X_train, y_train, train_groups, gkf, search_space):
    """
    Factory to create a correctly-scoped objective function for gp_minimize.
    """
    @use_named_args(search_space)
    def objective(**params):
        model = build_model(**params)
        # Calculate cross-validated R² score, which skopt will minimize (-R²)
        scores = [-r2_score(
            y_train[test_idx],
            model.fit(X_train[train_idx], y_train[train_idx]).predict(X_train[test_idx])
        ) for train_idx, test_idx in gkf.split(X_train, y_train, groups=train_groups)]
        return np.mean(scores)
    return objective

def main():
    # --- FEATURE: Add command-line arguments for selective training ---
    parser = argparse.ArgumentParser(description="Train models on the E. coli dataset.")
    parser.add_argument(
        '--fractions', 
        nargs='+', 
        choices=list(DATA_FRACTIONS.keys()) + ['all'], 
        default=['all'],
        help="Specify which data fractions to run. Default is 'all'."
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        choices=list(TARGETS.keys()) + ['all'],
        default=['all'],
        help="Specify which prediction targets to run. Default is 'all'."
    )
    args = parser.parse_args()
    
    fractions_to_run = DATA_FRACTIONS.keys() if 'all' in args.fractions else sorted(args.fractions, key=lambda k: DATA_FRACTIONS[k])
    targets_to_run = TARGETS.keys() if 'all' in args.targets else args.targets

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(RESIDUALS_DIR, exist_ok=True)

    logging.info("--- Loading and Preparing Data ---")
    df_master = load_ecoli(ANNOTATION_FILE)
    logging.info(f"Loaded {len(df_master)} rows from {ANNOTATION_FILE}.")

    logging.info("Loading embeddings...")
    x_dna = np.load(os.path.join(EMBEDDINGS_DIR, 'dnabert_embeddings.npy'))
    x_rinalmo = np.load(os.path.join(EMBEDDINGS_DIR, 'rinalmo_embeddings.npy'))
    x_rna_ernie = np.load(os.path.join(EMBEDDINGS_DIR, 'rna_ernie_embeddings.npy'))
    x_prot = np.load(os.path.join(EMBEDDINGS_DIR, 'esm2_embeddings.npy'))
    
    embedding_dims = {
        'dna': x_dna.shape[1], 'rinalmo': x_rinalmo.shape[1], 
        'rna_ernie': x_rna_ernie.shape[1], 'protein': x_prot.shape[1]
    }
    logging.info(f"Embedding dimensions: {embedding_dims}")

    X = np.concatenate([x_dna, x_rinalmo, x_rna_ernie, x_prot], axis=1)
    
    min_rows = min(len(df_master), X.shape[0])
    df_master, X = df_master.iloc[:min_rows], X[:min_rows]
    
    for col in TARGETS.values():
        if col in df_master.columns:
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
    
    valid_rows_mask = df_master[list(TARGETS.values())].notna().any(axis=1)
    df_clean, X_clean = df_master[valid_rows_mask].reset_index(drop=True), X[valid_rows_mask]
    logging.info(f"Retained {len(df_clean)} rows with at least one target.")

    logging.info("--- Performing robust operon-level data split ---")
    operon_id_buffer = 75
    long_enough_mask = df_clean['DNA_Sequence_Input'].str.len() >= operon_id_buffer
    df_operon, X_operon = df_clean[long_enough_mask].reset_index(drop=True), X_clean[long_enough_mask]
    logging.info(f"Filtered to {len(df_operon)} rows with DNA sequence long enough for grouping.")

    df_operon['operon_group_key'] = df_operon['DNA_Sequence_Input'].str[:operon_id_buffer]
    unique_operon_groups = df_operon['operon_group_key'].unique()

    train_operon_groups, holdout_operon_groups = train_test_split(
        unique_operon_groups, test_size=HOLDOUT_FRACTION, random_state=42
    )

    train_val_idx = df_operon[df_operon['operon_group_key'].isin(train_operon_groups)].index.values
    holdout_idx = df_operon[df_operon['operon_group_key'].isin(holdout_operon_groups)].index.values

    logging.info(f"Split data based on {len(unique_operon_groups)} unique operon groups.")
    logging.info(f"Training/Validation set: {len(train_val_idx)} genes from {len(train_operon_groups)} groups.")
    logging.info(f"Holdout set: {len(holdout_idx)} genes from {len(holdout_operon_groups)} groups.")

    with open("splits.json", "w") as f:
        json.dump({'holdout': holdout_idx.tolist()}, f)
    logging.info("Splits saved to splits.json.")
    
    y_cols = {key: df_operon[col] for key, col in TARGETS.items()}
    groups = df_operon['operon_group_key']

    # --- FIX: Robustly load previous results, filtering out fractions AND targets to be re-run ---
    summary_path = os.path.join(METRICS_DIR, "bayesian_optimization_summary.csv")
    importances_path = os.path.join(METRICS_DIR, "feature_importances.csv")
    history_path = os.path.join(METRICS_DIR, "optimization_history.csv")

    all_results = []
    all_importances = []
    optimization_history = []

    is_resuming = 'all' not in args.fractions or 'all' not in args.targets
    if is_resuming:
        if os.path.exists(summary_path):
            df_summary = pd.read_csv(summary_path)
            # Filter out any combination of fraction and target we are about to run
            df_summary = df_summary[~((df_summary['data_fraction'].isin(fractions_to_run)) & (df_summary['target'].isin(targets_to_run)))]
            all_results = df_summary.to_dict('records')
            logging.info(f"Resuming run. Loaded {len(all_results)} previous results.")
        if os.path.exists(importances_path):
            df_importances = pd.read_csv(importances_path)
            df_importances = df_importances[~((df_importances['data_fraction'].isin(fractions_to_run)) & (df_importances['target'].isin(targets_to_run)))]
            all_importances = df_importances.to_dict('records')
        if os.path.exists(history_path):
            df_history = pd.read_csv(history_path)
            df_history = df_history[~((df_history['data_fraction'].isin(fractions_to_run)) & (df_history['target'].isin(targets_to_run)))]
            optimization_history = df_history.to_dict('records')

    for frac_key in fractions_to_run:
        frac_val = DATA_FRACTIONS[frac_key]
        logging.info(f"\n--- Starting training for data fraction: {frac_key} ({frac_val*100:.1f}%) ---")
        
        if frac_val == 1.0:
            sub_train_idx = train_val_idx
        else:
            sub_train_idx, _ = train_test_split(train_val_idx, train_size=frac_val, random_state=42)

        for target_key in targets_to_run:
            y_series = y_cols[target_key]
            valid_idx_mask = y_series.iloc[sub_train_idx].notna()
            current_train_idx = sub_train_idx[valid_idx_mask]
            
            X_train, y_train = X_operon[current_train_idx], y_series.iloc[current_train_idx].values
            train_groups = groups.iloc[current_train_idx]

            # Baseline Model
            logging.info(f"[{frac_key}|baseline|{target_key}] Training Linear Regression...")
            baseline_model = build_baseline_model().fit(X_train, y_train)
            baseline_metrics = export_holdout_predictions(df_operon, holdout_idx, target_key, frac_key, "baseline", baseline_model, X_operon, y_series)
            all_results.append({'data_fraction': frac_key, 'variant': 'baseline', 'target': target_key, **baseline_metrics})

            # Embeddings-Only Model (XGBoost)
            logging.info(f"[{frac_key}|embeddings_only|{target_key}] Running Bayesian Optimization...")
            gkf = GroupKFold(n_splits=3)
            
            search_space_for_fraction = [
                Integer(100, int(max(101, 1000 * frac_val)), name='n_estimators') if dim.name == 'n_estimators' else dim
                for dim in SEARCH_SPACE
            ]
            
            objective_func = create_objective_function(X_train, y_train, train_groups, gkf, search_space_for_fraction)

            res = gp_minimize(
                objective_func,
                search_space_for_fraction,
                n_calls=N_OPTIMIZATION_CALLS,
                random_state=42,
                n_jobs=-1
            )
            
            best_params = {dim.name: val for dim, val in zip(search_space_for_fraction, res.x)}
            logging.info(f"Best parameters: {best_params}")

            # Store optimization history
            param_df = pd.DataFrame(res.x_iters, columns=[d.name for d in search_space_for_fraction])
            for i in range(len(res.func_vals)):
                history_entry = {
                    'data_fraction': frac_key, 'target': target_key, 'variant': 'embeddings_only',
                    'call_number': i + 1, 'objective_value': res.func_vals[i]
                }
                for param_name in param_df.columns:
                    history_entry[param_name] = param_df.iloc[i][param_name]
                optimization_history.append(history_entry)

            # Train final model and evaluate
            final_model = build_model(**best_params).fit(X_train, y_train)
            model_dir = os.path.join(MODELS_DIR, frac_key, target_key)
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(final_model, os.path.join(model_dir, "embeddings_only.pkl"))

            final_metrics = export_holdout_predictions(df_operon, holdout_idx, target_key, frac_key, "embeddings_only", final_model, X_operon, y_series)
            all_results.append({'data_fraction': frac_key, 'variant': 'embeddings_only', 'target': target_key, **final_metrics, **best_params})

            # Get and store feature importances
            importances = get_feature_importances(final_model, embedding_dims)
            for group, imp_val in importances.items():
                all_importances.append({'data_fraction': frac_key, 'target': target_key, 'feature_group': group, 'importance': imp_val})

    # Save all collected metrics
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    pd.DataFrame(all_importances).to_csv(importances_path, index=False)
    pd.DataFrame(optimization_history).to_csv(history_path, index=False)
    
    logging.info("\n--- Training pipeline complete. All models and metrics saved. ---")

if __name__ == "__main__":
    main()

