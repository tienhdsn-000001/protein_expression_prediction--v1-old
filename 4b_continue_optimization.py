#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4b_continue_optimization.py

- NEW SCRIPT: This script is designed to be run after `4_train_model.py`.
- PURPOSE: It loads the results of the initial Bayesian optimization for the
  100% data models and continues the search for a large number of additional
  steps. This allows for a much deeper exploration of the hyperparameter
  space to find the true performance plateau for the best models.
- INTELLIGENT STOPPING: The script includes a simple early stopping mechanism.
  If the best score found does not improve by a meaningful amount over a set
  number of recent trials, the optimization for that target will stop, saving
  computational time.
- OUTPUT: It overwrites the original model files and updates the metric CSVs
  with the improved results.
"""

import os
import csv
import json
import logging
import joblib
import numpy as np
import pandas as pd

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

# ------------------------------
# Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Core Paths ---
ANNOTATION_FILE = "Ecoli_Annotation_v3.tsv"
EMBEDDINGS_DIR = "embeddings"
MODELS_DIR = "models"
METRICS_DIR = "metrics"
RESIDUALS_DIR = os.path.join(METRICS_DIR, "residuals")

# --- Optimization Parameters ---
# Total number of calls will be the initial 25 + MAX_ADDITIONAL_CALLS
MAX_ADDITIONAL_CALLS = 125 
DATA_FRACTION_TO_IMPROVE = "100pct_data"

# --- Early Stopping Configuration ---
# Stop if the best score hasn't improved by at least this much...
PLATEAU_TOLERANCE = 0.0005
# ...over this many of the most recent calls.
PLATEAU_PATIENCE = 20

# Inherited from training script
TARGETS = {
    "mrna_half_life": "mrna_half_life",
    "mrna_max_half_life_delta": "mrna_max_half_life_delta",
    "mrna_min_half_life_delta": "mrna_min_half_life_delta",
    "log2_ppm_abundance": "log2_ppm_abundance",
    "log2_tpm_abundance": "log2_tpm_abundance",
    "protein_half_life": "protein_half_life",
}
SEARCH_SPACE = [
    Integer(100, 1000, name='n_estimators'), Real(0.01, 0.3, name='learning_rate'),
    Integer(3, 10, name='max_depth'), Real(0.5, 1.0, name='subsample'),
    Real(0.5, 1.0, name='colsample_bytree'),
]
BASE_XGB_PARAMS = {
    'objective': 'reg:squarederror', 'tree_method': 'hist',
    'device': 'cuda' if CUPY_AVAILABLE else 'cpu', 'n_jobs': -1, 'random_state': 42,
}

# ------------------------------
# Helper Functions (from 4_train_model.py)
# ------------------------------

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
    return Pipeline([
        ('scaler', StandardScaler()),
        ('tocupy', ToCuPyTransformer()),
        ('xgb', xgb.XGBRegressor(**{**BASE_XGB_PARAMS, **xgb_params}))
    ])

def export_holdout_predictions(df_meta, holdout_indices, target_name, data_fraction, variant, model, X_mat, y_series):
    """Calculates metrics and saves predictions for the holdout set."""
    valid_indices_mask = np.isin(holdout_indices, df_meta.index) & np.isin(holdout_indices, y_series.index)
    valid_indices = holdout_indices[valid_indices_mask]
    
    y_holdout = y_series.loc[valid_indices]
    non_na_mask = y_holdout.notna()
    final_indices = y_holdout[non_na_mask].index.values
    
    if final_indices.size == 0:
        return {'holdout_r2': np.nan, 'holdout_mse': np.nan}
        
    X_hold, y_hold = X_mat[final_indices], y_series.loc[final_indices].values
    y_pred = model.predict(X_hold)
    y_hold_np, y_pred_np = to_numpy_if_cupy(y_hold), to_numpy_if_cupy(y_pred)
    
    gene_id_col = next((c for c in ['Gene_Name', 'Protein_ID'] if c in df_meta.columns), None)
    gene_ids = df_meta.loc[final_indices, gene_id_col].values if gene_id_col else final_indices
    
    out_df = pd.DataFrame({'gene_identifier': gene_ids, 'y_true': y_hold_np, 'y_pred': y_pred_np})
    frac_dir = os.path.join(RESIDUALS_DIR, data_fraction, variant)
    os.makedirs(frac_dir, exist_ok=True)
    out_df.to_csv(os.path.join(frac_dir, f"{target_name}__holdout_predictions.csv"), index=False)
    
    holdout_r2 = r2_score(y_hold_np, y_pred_np)
    holdout_mse = mean_squared_error(y_hold_np, y_pred_np)
    logging.info(f"Holdout R²: {holdout_r2:.4f}, MSE: {holdout_mse:.4f}")
    
    return {'holdout_r2': float(holdout_r2), 'holdout_mse': float(holdout_mse)}

# ------------------------------
# Main Execution
# ------------------------------

def main():
    logging.info("--- Starting Continued Bayesian Optimization ---")
    
    # --- Load All Data (same as training script) ---
    df_master = load_ecoli(ANNOTATION_FILE)
    x_dna = np.load(os.path.join(EMBEDDINGS_DIR, 'dnabert_embeddings.npy'))
    x_rna = np.load(os.path.join(EMBEDDINGS_DIR, 'codonbert_embeddings.npy'))
    x_prot = np.load(os.path.join(EMBEDDINGS_DIR, 'esm2_embeddings.npy'))
    X = np.concatenate([x_dna, x_rna, x_prot], axis=1)
    
    min_rows = min(len(df_master), X.shape[0])
    df_master, X = df_master.iloc[:min_rows], X[:min_rows]
    
    for col in TARGETS.values():
        if col in df_master.columns:
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
    
    valid_rows_mask = df_master[list(TARGETS.values())].notna().any(axis=1)
    df_clean, X_clean = df_master[valid_rows_mask].reset_index(drop=True), X[valid_rows_mask]

    operon_id_buffer = 75
    long_enough_mask = df_clean['DNA_Sequence_Input'].str.len() >= operon_id_buffer
    df_operon, X_operon = df_clean[long_enough_mask].reset_index(drop=True), X_clean[long_enough_mask]
    df_operon['operon_group_key'] = df_operon['DNA_Sequence_Input'].str[:operon_id_buffer]
    
    # --- Load Splits and Previous Results ---
    with open("splits.json", "r") as f:
        holdout_idx = np.array(json.load(f)['holdout'])
    
    train_val_idx = df_operon.index.difference(holdout_idx).values
    
    summary_path = os.path.join(METRICS_DIR, "bayesian_optimization_summary.csv")
    history_path = os.path.join(METRICS_DIR, "optimization_history.csv")
    df_summary = pd.read_csv(summary_path)
    df_history = pd.read_csv(history_path)

    y_cols = {key: df_operon[col] for key, col in TARGETS.items()}
    groups = df_operon['operon_group_key']

    for target_key, y_series in y_cols.items():
        logging.info(f"\n--- Continuing optimization for target: {target_key} ---")
        
        valid_idx_mask = y_series.iloc[train_val_idx].notna()
        current_train_idx = train_val_idx[valid_idx_mask]
        
        X_train, y_train = X_operon[current_train_idx], y_series.iloc[current_train_idx].values
        train_groups = groups.iloc[current_train_idx]
        
        # --- Restore state from previous run ---
        prev_history = df_history[
            (df_history['data_fraction'] == DATA_FRACTION_TO_IMPROVE) &
            (df_history['target'] == target_key) &
            (df_history['variant'] == 'embeddings_only')
        ]
        
        # --- FIX: Reconstruct x0 from the full history of all tried parameters ---
        param_names = [dim.name for dim in SEARCH_SPACE]
        if not all(p in prev_history.columns for p in param_names):
             logging.error(f"FATAL: History CSV for {target_key} is missing parameter columns. Please re-run 4_train_model.py first.")
             continue
        
        x0 = prev_history[param_names].values.tolist()
        y0 = prev_history['objective_value'].tolist()
        
        logging.info(f"Starting from {len(y0)} previous calls. Best score so far: {min(y0):.4f}")

        # --- Define objective and run extended optimization ---
        gkf = GroupKFold(n_splits=3)
        @use_named_args(SEARCH_SPACE)
        def objective(**params):
            model = build_model(**params)
            scores = [-r2_score(y_train[test_idx], model.fit(X_train[train_idx], y_train[train_idx]).predict(X_train[test_idx])) for train_idx, test_idx in gkf.split(X_train, y_train, groups=train_groups)]
            return np.mean(scores)

        # Early stopping callback
        def early_stopping_callback(res):
            # The number of func_vals will be total calls, not just additional ones
            current_total_calls = len(res.func_vals)
            
            # Check if we have enough calls to evaluate the plateau
            if current_total_calls < len(y0) + PLATEAU_PATIENCE:
                return False 
            
            recent_scores = res.func_vals[-PLATEAU_PATIENCE:]
            best_recent = min(recent_scores)
            
            # Find the best score before this recent window
            best_past = min(res.func_vals[:-PLATEAU_PATIENCE])
            
            if best_past - best_recent < PLATEAU_TOLERANCE:
                logging.info(f"Stopping early for {target_key}. No improvement in the last {PLATEAU_PATIENCE} calls.")
                return True # Stop the optimization
            return False

        res = gp_minimize(
            objective, SEARCH_SPACE, 
            n_calls=MAX_ADDITIONAL_CALLS,
            x0=x0, y0=y0, # Start from previous state
            random_state=42, n_jobs=-1,
            callback=early_stopping_callback
        )
        
        # --- Update and Save Results ---
        best_params = {dim.name: val for dim, val in zip(SEARCH_SPACE, res.x)}
        logging.info(f"Finished at {len(res.func_vals)} total calls. New best score: {res.fun:.4f}")

        # --- FIX: Cast integer hyperparameters back to int, as skopt can return them as floats ---
        if 'n_estimators' in best_params:
            best_params['n_estimators'] = int(best_params['n_estimators'])
        if 'max_depth' in best_params:
            best_params['max_depth'] = int(best_params['max_depth'])

        logging.info(f"New best parameters (types corrected): {best_params}")

        final_model = build_model(**best_params)
        final_model.fit(X_train, y_train)
        
        model_dir = os.path.join(MODELS_DIR, DATA_FRACTION_TO_IMPROVE, target_key)
        joblib.dump(final_model, os.path.join(model_dir, "embeddings_only.pkl"))
        
        metrics = export_holdout_predictions(df_operon, holdout_idx, target_key, DATA_FRACTION_TO_IMPROVE, "embeddings_only", final_model, X_operon, y_series)
        
        # Update summary dataframe
        summary_idx = df_summary[
            (df_summary['data_fraction'] == DATA_FRACTION_TO_IMPROVE) &
            (df_summary['target'] == target_key) &
            (df_summary['variant'] == 'embeddings_only')
        ].index
        
        for key, value in {**metrics, **best_params}.items():
            df_summary.loc[summary_idx, key] = value

        # Update history dataframe
        df_history = df_history[~((df_history['data_fraction'] == DATA_FRACTION_TO_IMPROVE) & (df_history['target'] == target_key))]
        new_history_df = pd.DataFrame({
            'data_fraction': DATA_FRACTION_TO_IMPROVE,
            'target': target_key,
            'variant': 'embeddings_only',
            'call_number': range(1, len(res.func_vals) + 1),
            'objective_value': res.func_vals
        })
        # Add parameter columns to the new history
        param_df = pd.DataFrame(res.x_iters, columns=[d.name for d in SEARCH_SPACE])
        for col in param_df.columns:
            new_history_df[col] = param_df[col]

        df_history = pd.concat([df_history, new_history_df], ignore_index=True)

    # Save updated metrics back to disk
    df_summary.to_csv(summary_path, index=False)
    df_history.to_csv(history_path, index=False)
    
    logging.info("\n--- Continued optimization complete. Models and metrics updated. ---")


if __name__ == "__main__":
    main()


