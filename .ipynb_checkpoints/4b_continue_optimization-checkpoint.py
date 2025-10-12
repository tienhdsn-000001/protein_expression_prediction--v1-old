#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4b_continue_optimization.py

- FIX (CRITICAL, TypeError): Resolved the 'unexpected keyword argument' error by
  making this script self-contained. The `create_objective_function` is now defined
  locally, which correctly uses the local `build_model` function, resolving both
  the immediate TypeError and the underlying pickling instability with multiprocessing.
- FIX (CRITICAL, OOM): Added `max_bin=128` to the local XGBoost parameters to
  match the main training script, preventing potential out-of-memory errors
  on GPU during the extended optimization.
- UPDATE (Pipeline Alignment): The data loading section has been updated to use
  the new RNA embeddings (Rinalmo, RNA-ERNIE) and correctly calculate feature
  importances for the four new embedding groups, aligning it with the current
  pipeline.
- REMOVED: All dynamic `importlib` calls for core functions to ensure stability.
"""
import os
import csv
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import xgboost as xgb


try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not found. GPU acceleration disabled for XGBoost.")

# --- Configuration (Aligned with 4_train_model.py) ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ANNOTATION_FILE = "Ecoli_Annotation_v3.tsv"
EMBEDDINGS_DIR = "embeddings"
MODELS_DIR = "models"
METRICS_DIR = "metrics"
RESIDUALS_DIR = os.path.join(METRICS_DIR, "residuals")
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
# --- FIX: Added max_bin=128 to reduce GPU memory usage ---
BASE_XGB_PARAMS = {
    'objective': 'reg:squarederror', 'tree_method': 'hist',
    'device': 'cuda' if CUPY_AVAILABLE else 'cpu', 'n_jobs': -1, 'random_state': 42,
    'max_bin': 128
}
N_ADDITIONAL_CALLS = 75
EARLY_STOPPING_PATIENCE = 20

# --- Helper Functions (Copied from 4_train_model.py for stability) ---

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
    if 'n_estimators' in xgb_params:
        xgb_params['n_estimators'] = int(xgb_params['n_estimators'])
    if 'max_depth' in xgb_params:
        xgb_params['max_depth'] = int(xgb_params['max_depth'])
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
    """Factory to create a correctly-scoped objective function for gp_minimize."""
    @use_named_args(search_space)
    def objective(**params):
        # This now calls the local `build_model` function, solving the error
        model = build_model(**params)
        scores = [-r2_score(
            y_train[test_idx],
            model.fit(X_train[train_idx], y_train[train_idx]).predict(X_train[test_idx])
        ) for train_idx, test_idx in gkf.split(X_train, y_train, groups=train_groups)]
        return np.mean(scores)
    return objective

class EarlyStoppingCallback:
    """Callback to stop gp_minimize optimization early if there is no improvement."""
    def __init__(self, best_score_so_far, patience=20):
        self.patience = patience
        self.best_score_so_far = best_score_so_far
        self.counter = 0

    def __call__(self, result):
        latest_score = result.func_vals[-1]
        if latest_score < self.best_score_so_far - 1e-6:
            self.best_score_so_far = latest_score
            self.counter = 0
            logging.debug(f"Callback: New best score found: {latest_score:.4f}")
        else:
            self.counter += 1
        if self.counter >= self.patience:
            logging.info(f"Callback: Stopping early. No improvement in the last {self.patience} calls.")
            return True
        else:
            return False

def main():
    logging.info("--- Starting Continued Bayesian Optimization ---")

    df_master = load_ecoli(ANNOTATION_FILE)
    logging.info("Loading embeddings...")
    # --- UPDATE: Load new RNA embeddings ---
    x_dna = np.load(os.path.join(EMBEDDINGS_DIR, 'dnabert_embeddings.npy'))
    x_rinalmo = np.load(os.path.join(EMBEDDINGS_DIR, 'rinalmo_embeddings.npy'))
    x_rna_ernie = np.load(os.path.join(EMBEDDINGS_DIR, 'rna_ernie_embeddings.npy'))
    x_prot = np.load(os.path.join(EMBEDDINGS_DIR, 'esm2_embeddings.npy'))

    # --- UPDATE: Correct embedding dimensions and concatenation order ---
    embedding_dims = {
        'dna': x_dna.shape[1], 'rinalmo': x_rinalmo.shape[1],
        'rna_ernie': x_rna_ernie.shape[1], 'protein': x_prot.shape[1]
    }

    X = np.concatenate([x_dna, x_rinalmo, x_rna_ernie, x_prot], axis=1)
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
    
    with open("splits.json", 'r') as f:
        splits = json.load(f)
    holdout_idx = np.array(splits['holdout'])

    # Ensure holdout indices are valid before dropping
    valid_holdout_idx = np.intersect1d(holdout_idx, df_operon.index)
    train_val_idx = df_operon.index.drop(valid_holdout_idx).values
    
    y_cols = {key: df_operon[col] for key, col in TARGETS.items()}
    groups = df_operon['operon_group_key']

    summary_path = os.path.join(METRICS_DIR, "bayesian_optimization_summary.csv")
    history_path = os.path.join(METRICS_DIR, "optimization_history.csv")
    importances_path = os.path.join(METRICS_DIR, "feature_importances.csv")

    if not os.path.exists(summary_path) or not os.path.exists(history_path):
        logging.error("Metrics files not found. Please run 4_train_model.py first.")
        return

    df_summary = pd.read_csv(summary_path)
    df_history = pd.read_csv(history_path)
    df_importances = pd.read_csv(importances_path)

    for target_key in TARGETS:
        logging.info(f"\n--- Continuing optimization for target: {target_key} ---")

        target_history = df_history[
            (df_history['data_fraction'] == '100pct_data') &
            (df_history['target'] == target_key) &
            (df_history['variant'] == 'embeddings_only')
        ]

        valid_history = target_history.dropna(subset=['objective_value'])
        if valid_history.empty:
            logging.warning(f"No valid previous optimization history found for {target_key}. Skipping.")
            continue

        x0 = valid_history[[d.name for d in SEARCH_SPACE]].values.tolist()
        y0 = valid_history['objective_value'].tolist()

        if not y0:
            logging.warning(f"Objective values list for {target_key} is empty after cleaning. Skipping.")
            continue

        best_score_so_far = min(y0)
        logging.info(f"Starting from {len(y0)} previous calls. Best score so far: {best_score_so_far:.4f}")
        
        y_series = y_cols[target_key]
        valid_idx_mask = y_series.iloc[train_val_idx].notna()
        current_train_idx = train_val_idx[valid_idx_mask]
        
        X_train, y_train = X_operon[current_train_idx], y_series.iloc[current_train_idx].values
        train_groups = groups.iloc[current_train_idx]
        
        gkf = GroupKFold(n_splits=3)
        # --- FIX: Call the local create_objective_function ---
        objective_func = create_objective_function(X_train, y_train, train_groups, gkf, SEARCH_SPACE)
        
        early_stopper = EarlyStoppingCallback(best_score_so_far=best_score_so_far, patience=EARLY_STOPPING_PATIENCE)
        
        res = gp_minimize(
            objective_func,
            SEARCH_SPACE,
            n_calls=N_ADDITIONAL_CALLS,
            random_state=42,
            n_jobs=-1,
            x0=x0,
            y0=y0,
            callback=early_stopper
        )

        best_new_score = res.fun
        
        if best_new_score < best_score_so_far:
            logging.info(f"Improvement found for {target_key}! New best score: {best_new_score:.4f}")
            best_params = {dim.name: val for dim, val in zip(SEARCH_SPACE, res.x)}
        else:
            logging.info(f"No significant improvement found for {target_key}. Keeping original best parameters.")
            best_params_series = valid_history.loc[valid_history['objective_value'].idxmin()]
            best_params = {dim.name: best_params_series[dim.name] for dim in SEARCH_SPACE}

        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        
        logging.info(f"Final best parameters: {best_params}")

        final_model = build_model(**best_params).fit(X_train, y_train)
        model_dir = os.path.join(MODELS_DIR, '100pct_data', target_key)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(final_model, os.path.join(model_dir, "embeddings_only.pkl"))
        
        final_metrics = export_holdout_predictions(df_operon, holdout_idx, target_key, '100pct_data', "embeddings_only", final_model, X_operon, y_series)
        
        summary_mask = (df_summary['data_fraction'] == '100pct_data') & (df_summary['target'] == target_key) & (df_summary['variant'] == 'embeddings_only')
        for col, val in {**final_metrics, **best_params}.items():
            df_summary.loc[summary_mask, col] = val
        
        # Update history
        df_history = df_history[~((df_history['data_fraction'] == '100pct_data') & (df_history['target'] == target_key) & (df_history['variant'] == 'embeddings_only'))]
        new_history_df = pd.DataFrame(res.x_iters, columns=[d.name for d in SEARCH_SPACE])
        new_history_df['objective_value'] = res.func_vals
        new_history_df['data_fraction'] = '100pct_data'
        new_history_df['target'] = target_key
        new_history_df['variant'] = 'embeddings_only'
        new_history_df['call_number'] = range(1, len(res.func_vals) + 1)
        df_history = pd.concat([df_history, new_history_df], ignore_index=True)

        # Update importances
        importances = get_feature_importances(final_model, embedding_dims)
        df_importances = df_importances[~((df_importances['data_fraction'] == '100pct_data') & (df_importances['target'] == target_key))]
        for group, imp_val in importances.items():
            df_importances = pd.concat([df_importances, pd.DataFrame([{'data_fraction': '100pct_data', 'target': target_key, 'feature_group': group, 'importance': imp_val}])], ignore_index=True)

    df_summary.to_csv(summary_path, index=False)
    df_history.to_csv(history_path, index=False)
    df_importances.to_csv(importances_path, index=False)
    
    logging.info("\n--- Continued optimization complete. All metrics updated. ---")

if __name__ == "__main__":
    main()
