#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4_train_model.py

- UPDATE (Optimization History): The script now captures and saves the full
  history of the Bayesian optimization process, including the specific
  hyperparameters for each call, to `metrics/optimization_history.csv`.
- FIX (Stratification): Removed the `stratify` parameter from data fraction
  subsampling to resolve ValueErrors with single-member groups.
- CRITICAL UPDATE (Operon Split): Implemented a robust, sequence-based split
  to prevent data leakage between operons.
- UPDATE (Reporting): Trains a baseline Linear Regression model and saves
  R², MSE, and aggregated feature importances.
- FIX (Data Leakage): Uses GroupKFold in hyperparameter tuning.
"""

import os
import csv
import json
import logging
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
BASE_XGB_PARAMS = {
    'objective': 'reg:squarederror', 'tree_method': 'hist',
    'device': 'cuda' if CUPY_AVAILABLE else 'cpu', 'n_jobs': -1, 'random_state': 42,
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
    valid_indices_mask = np.isin(holdout_indices, df_meta.index) & np.isin(holdout_indices, y_series.index)
    valid_indices = holdout_indices[valid_indices_mask]
    
    y_holdout = y_series.loc[valid_indices]
    non_na_mask = y_holdout.notna()
    
    final_indices = y_holdout[non_na_mask].index.values
    
    if final_indices.size == 0:
        logging.warning(f"No valid non-NA holdout rows for {target_name}. Skipping.")
        return {'holdout_r2': np.nan, 'holdout_mse': np.nan}
        
    X_hold, y_hold = X_mat[final_indices], y_series.loc[final_indices].values
    y_pred = model.predict(X_hold)
    
    y_hold_np, y_pred_np = to_numpy_if_cupy(y_hold), to_numpy_if_cupy(y_pred)
    
    gene_id_col = next((c for c in ['Gene_Name', 'Protein_ID', 'gene_id', 'locus_tag'] if c in df_meta.columns), None)
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

def get_feature_importances(model: Pipeline, dna_dim, rna_dim, prot_dim) -> Dict:
    """Extracts feature importances and aggregates them by embedding type."""
    if 'xgb' not in model.named_steps:
        return {}
    importances = model.named_steps['xgb'].feature_importances_
    
    return {
        'dna': float(importances[:dna_dim].sum()),
        'rna': float(importances[dna_dim:dna_dim + rna_dim].sum()),
        'protein': float(importances[dna_dim + rna_dim:].sum())
    }

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(RESIDUALS_DIR, exist_ok=True)

    logging.info("--- Loading and Preparing Data ---")
    df_master = load_ecoli(ANNOTATION_FILE)
    logging.info(f"Loaded {len(df_master)} rows from {ANNOTATION_FILE}.")

    logging.info("Loading embeddings...")
    x_dna = np.load(os.path.join(EMBEDDINGS_DIR, 'dnabert_embeddings.npy'))
    x_rna = np.load(os.path.join(EMBEDDINGS_DIR, 'codonbert_embeddings.npy'))
    x_prot = np.load(os.path.join(EMBEDDINGS_DIR, 'esm2_embeddings.npy'))
    X = np.concatenate([x_dna, x_rna, x_prot], axis=1)
    
    dna_dim, rna_dim, prot_dim = x_dna.shape[1], x_rna.shape[1], x_prot.shape[1]

    if len(df_master) != X.shape[0]:
        min_rows = min(len(df_master), X.shape[0])
        logging.warning(f"Row count mismatch! Annotations: {len(df_master)}, Embeddings: {X.shape[0]}. Truncating to {min_rows}.")
        df_master, X = df_master.iloc[:min_rows], X[:min_rows]
    
    assert len(df_master) == X.shape[0], "FATAL: Mismatch between annotations and features."

    for col in TARGETS.values():
        if col in df_master.columns:
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
    
    valid_rows_mask = df_master[list(TARGETS.values())].notna().any(axis=1)
    df_clean, X_clean = df_master[valid_rows_mask].reset_index(drop=True), X[valid_rows_mask]
    logging.info(f"Retained {len(df_clean)} rows with at least one target.")

    logging.info("--- Performing robust operon-level data split ---")
    operon_id_buffer = 75
    if 'DNA_Sequence_Input' not in df_clean.columns:
        raise ValueError("FATAL: 'DNA_Sequence_Input' not found for operon-based split.")

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
    
    all_results, all_importances, all_optim_history = [], [], []

    for frac_name, frac_val in DATA_FRACTIONS.items():
        logging.info(f"\n--- Starting training for data fraction: {frac_name} ({frac_val*100}%) ---")
        
        if frac_val < 1.0:
            sub_train_idx, _ = train_test_split(train_val_idx, train_size=frac_val, random_state=42)
        else:
            sub_train_idx = train_val_idx

        logging.info(f"Using {len(sub_train_idx)} samples for this fraction.")

        for target_key, y_series in y_cols.items():
            valid_idx_mask = y_series.iloc[sub_train_idx].notna()
            current_train_idx = sub_train_idx[valid_idx_mask]
            
            if len(current_train_idx) < 50:
                logging.warning(f"Skipping {target_key} for {frac_name}: only {len(current_train_idx)} valid samples.")
                continue

            X_train, y_train = X_operon[current_train_idx], y_series.iloc[current_train_idx].values
            train_groups = groups.iloc[current_train_idx]
            
            # --- Baseline Model ---
            logging.info(f"[{frac_name}|baseline|{target_key}] Training Linear Regression...")
            baseline_model = build_baseline_model()
            baseline_model.fit(X_train, y_train)
            
            model_dir = os.path.join(MODELS_DIR, frac_name, target_key)
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(baseline_model, os.path.join(model_dir, "baseline.pkl"))

            metrics = export_holdout_predictions(df_operon, holdout_idx, target_key, frac_name, "baseline", baseline_model, X_operon, y_series)
            all_results.append({'data_fraction': frac_name, 'variant': 'baseline', 'target': target_key, **metrics})

            # --- XGBoost Model (Embeddings Only) ---
            logging.info(f"[{frac_name}|embeddings_only|{target_key}] Running Bayesian Optimization...")
            
            gkf = GroupKFold(n_splits=3)
            
            @use_named_args(SEARCH_SPACE)
            def objective(**params):
                model = build_model(**params)
                if CUPY_AVAILABLE:
                    scores = [-r2_score(to_numpy_if_cupy(y_train[test_idx]), to_numpy_if_cupy(model.fit(X_train[train_idx], y_train[train_idx]).predict(X_train[test_idx]))) for train_idx, test_idx in gkf.split(X_train, y_train, groups=train_groups)]
                else:
                    scores = [-r2_score(y_train[test_idx], model.fit(X_train[train_idx], y_train[train_idx]).predict(X_train[test_idx])) for train_idx, test_idx in gkf.split(X_train, y_train, groups=train_groups)]
                return np.mean(scores)

            res = gp_minimize(objective, SEARCH_SPACE, n_calls=N_OPTIMIZATION_CALLS, random_state=42, n_jobs=-1)
            best_params = {dim.name: val for dim, val in zip(SEARCH_SPACE, res.x)}
            
            # --- UPDATE: Save optimization history with full parameters for each call ---
            for i, score in enumerate(res.func_vals):
                params_at_call = {dim.name: res.x_iters[i][j] for j, dim in enumerate(SEARCH_SPACE)}
                history_record = {
                    'data_fraction': frac_name, 'target': target_key,
                    'variant': 'embeddings_only', 'call_number': i + 1,
                    'objective_value': score,
                    **params_at_call
                }
                all_optim_history.append(history_record)
            
            logging.info(f"Best parameters: {best_params}")
            final_model = build_model(**best_params)
            final_model.fit(X_train, y_train)
            joblib.dump(final_model, os.path.join(model_dir, "embeddings_only.pkl"))
            
            metrics = export_holdout_predictions(df_operon, holdout_idx, target_key, frac_name, "embeddings_only", final_model, X_operon, y_series)
            all_results.append({'data_fraction': frac_name, 'variant': 'embeddings_only', 'target': target_key, **metrics, **best_params})
            
            importances = get_feature_importances(final_model, dna_dim, rna_dim, prot_dim)
            if importances:
                for group, imp_val in importances.items():
                    all_importances.append({
                        'data_fraction': frac_name, 'target': target_key, 
                        'feature_group': group, 'importance': imp_val
                    })

    pd.DataFrame(all_results).to_csv(os.path.join(METRICS_DIR, "bayesian_optimization_summary.csv"), index=False)
    pd.DataFrame(all_importances).to_csv(os.path.join(METRICS_DIR, "feature_importances.csv"), index=False)
    pd.DataFrame(all_optim_history).to_csv(os.path.join(METRICS_DIR, "optimization_history.csv"), index=False)
    logging.info("\n--- Training pipeline complete. All models and metrics saved. ---")

if __name__ == "__main__":
    main()

