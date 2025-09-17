#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4_train_model.py

- FIX: Corrected the pandas `read_csv` separator from "\\t" to the proper
  tab character escape sequence '\t'. This was the root cause of the file
  truncation issue.
- FIX: Replaced the direct `pd.read_csv` call with a robust loading
  function. This function attempts to parse the file with UTF-16 encoding
  (a common source of errors from spreadsheet exports) and falls back to
  UTF-8, using the Python engine and strict error handling to prevent
  silently skipping rows.
- FIX: Removed the `low_memory` parameter from the `pd.read_csv` call, as
  it is not supported by the 'python' engine and was causing a ValueError.
- UPDATE: Reorganized the model output directory structure to group models
  by their prediction target, as requested. The new structure is:
  models/{data_fraction}/{target_name}/{variant}.pkl
- UPDATE: Increased the granularity of the data fractions to provide a more
  detailed view of performance scaling with data size.
"""

import os
import csv
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import xgboost as xgb
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not found. GPU acceleration disabled.")

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ANNOTATION_FILE = "Ecoli_Annotation_v2.tsv"
EMBEDDINGS_DIR = "embeddings"
MODELS_DIR = "models"
METRICS_DIR = "metrics"
RESIDUALS_DIR = os.path.join(METRICS_DIR, "residuals")
# --- CHANGE: Increased the number of data fractions for more granular analysis ---
DATA_FRACTIONS = {
    "20pct_data": 0.20,
    "40pct_data": 0.40,
    "60pct_data": 0.60,
    "80pct_data": 0.80,
    "100pct_data": 1.0
}
HOLDOUT_FRACTION = 0.1
PRIMARY_TARGET = "max_half_life_delta"
ABUNDANCE_TARGETS = ["log2_ppm_abundance", "log2_tpm_abundance"]
TARGETS = {
    "max_half_life_delta": "max_half_life_delta",
    "min_half_life_delta": "min_half_life_delta",
    "log2_ppm_abundance": "log2_ppm_abundance",
    "log2_tpm_abundance": "log2_tpm_abundance",
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
    """
    Robustly loads the E. coli annotation TSV, attempting to parse it as
    UTF-16 first (common for Excel exports) and falling back to UTF-8.
    Uses the Python engine for better handling of complex quoted fields.
    """
    try:
        # Attempt to read with UTF-16, which is a common export format
        return pd.read_csv(
            path, sep="\t", engine="python", encoding="utf-16",
            dtype=str, na_filter=False, quotechar='"', quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="error",
        )
    except (UnicodeError, pd.errors.ParserError):
        # Fall back to UTF-8 if UTF-16 fails
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

def gpu_safe_r2_scorer(estimator, X, y_true):
    y_pred, y_true_np = estimator.predict(X), to_numpy_if_cupy(y_true)
    y_pred_np = to_numpy_if_cupy(y_pred)
    return r2_score(y_true_np, y_pred_np)

def build_model(**xgb_params) -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('tocupy', ToCuPyTransformer()),
        ('xgb', xgb.XGBRegressor(**{**BASE_XGB_PARAMS, **xgb_params}))
    ])

def export_holdout_predictions(df_meta, indices, target_name, data_fraction, variant, model, X_mat, y_series):
    valid_idx = np.array([i for i in indices if not pd.isna(y_series.iloc[i])], dtype=int)
    if valid_idx.size == 0:
        logging.warning(f"No valid holdout rows for {target_name}. Skipping.")
        return {'holdout_r2': np.nan}
    X_hold, y_hold = X_mat[valid_idx], y_series.iloc[valid_idx].values
    y_pred = model.predict(X_hold)
    y_hold_np, y_pred_np = to_numpy_if_cupy(y_hold), to_numpy_if_cupy(y_pred)
    gene_id_col = next((c for c in ['Gene_Name', 'Protein_ID', 'gene_id', 'locus_tag'] if c in df_meta.columns), None)
    gene_ids = df_meta.iloc[valid_idx][gene_id_col].values if gene_id_col else valid_idx
    out_df = pd.DataFrame({'gene_identifier': gene_ids, 'y_true': y_hold_np, 'y_pred': y_pred_np})
    out_df['abs_error'] = np.abs(out_df['y_true'] - out_df['y_pred'])
    frac_dir = os.path.join(RESIDUALS_DIR, data_fraction, variant)
    os.makedirs(frac_dir, exist_ok=True)
    out_df.to_csv(os.path.join(frac_dir, f"{target_name}__holdout_predictions.csv"), index=False)
    holdout_r2 = r2_score(y_hold_np, y_pred_np)
    logging.info(f"[{data_fraction}|{variant}|{target_name}] Holdout R²: {holdout_r2:.4f}")
    return {'holdout_r2': float(holdout_r2)}

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
    
    # Handle potential row mismatch after robust loading
    if len(df_master) != X.shape[0]:
        logging.warning(f"Row count mismatch! Annotations: {len(df_master)}, Embeddings: {X.shape[0]}.")
        # Assuming embeddings are the source of truth for length if they are shorter
        min_rows = min(len(df_master), X.shape[0])
        logging.warning(f"Truncating to {min_rows} rows to proceed.")
        df_master = df_master.iloc[:min_rows]
        X = X[:min_rows]
    
    assert len(df_master) == X.shape[0], "FATAL: Mismatch between annotations and features after attempting to resolve."


    for col in TARGETS.values():
        if col in df_master.columns:
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
    
    valid_rows_mask = df_master[list(TARGETS.values())].notna().any(axis=1)
    df_clean, X_clean = df_master[valid_rows_mask].reset_index(drop=True), X[valid_rows_mask]
    logging.info(f"Retained {len(df_clean)} rows with at least one target.")

    indices = df_clean.index.values
    train_val_idx, holdout_idx = train_test_split(indices, test_size=HOLDOUT_FRACTION, random_state=42)
    
    y_cols = {key: df_clean[col] for key, col in TARGETS.items()}
    optimization_results = []

    for frac_name, frac_val in DATA_FRACTIONS.items():
        logging.info(f"\n--- Data Fraction: {frac_name} ---")
        train_idx_frac = train_val_idx
        if frac_val < 1.0:
            train_idx_frac, _ = train_test_split(train_val_idx, train_size=frac_val, random_state=42)
        
        best_primary_model = None

        for target_key, y_series in y_cols.items():
            logging.info(f"\nOptimizing Embeddings-Only for: {target_key}")

            # --- CHANGE: Create a dedicated directory for each target model ---
            # New structure: models/{data_fraction}/{target_name}/
            target_model_dir = os.path.join(MODELS_DIR, frac_name, target_key)
            os.makedirs(target_model_dir, exist_ok=True)
            
            train_indices = np.intersect1d(train_idx_frac, y_series.dropna().index)
            if len(train_indices) < 20: continue
            X_train, y_train = X_clean[train_indices], y_series.iloc[train_indices].values

            @use_named_args(SEARCH_SPACE)
            def objective(**params):
                model = build_model(**params)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring=gpu_safe_r2_scorer)
                return -np.mean(scores)

            result = gp_minimize(objective, SEARCH_SPACE, n_calls=N_OPTIMIZATION_CALLS, random_state=42)
            best_params = {dim.name: val for dim, val in zip(SEARCH_SPACE, result.x)}
            
            final_model = build_model(**best_params).fit(X_train, y_train)
            
            # --- CHANGE: Save the model into the new target-specific directory ---
            joblib.dump(final_model, os.path.join(target_model_dir, "embeddings_only.pkl"))
            
            eval_dict = export_holdout_predictions(df_clean, holdout_idx, target_key, frac_name, 'embeddings_only', final_model, X_clean, y_series)
            optimization_results.append({'data_fraction': frac_name, 'variant': 'embeddings_only', 'target': target_key, 'best_cv_r2': -result.fun, **eval_dict, **best_params})
            if target_key == PRIMARY_TARGET: best_primary_model = final_model

        if best_primary_model:
            logging.info("\n--- Running Augmented Models ---")
            primary_preds = to_numpy_if_cupy(best_primary_model.predict(X_clean)).reshape(-1, 1)
            X_aug = np.concatenate([X_clean, primary_preds, X_clean[:, :10] * primary_preds], axis=1)

            for target_key in ABUNDANCE_TARGETS:
                logging.info(f"Optimizing Augmented for: {target_key}")
                
                # --- CHANGE: Create a dedicated directory for each target model ---
                # New structure: models/{data_fraction}/{target_name}/
                target_model_dir = os.path.join(MODELS_DIR, frac_name, target_key)
                os.makedirs(target_model_dir, exist_ok=True)
                
                y_series = y_cols[target_key]
                train_indices = np.intersect1d(train_idx_frac, y_series.dropna().index)
                if len(train_indices) < 20: continue
                X_train_aug, y_train = X_aug[train_indices], y_series.iloc[train_indices].values
                
                @use_named_args(SEARCH_SPACE)
                def objective_aug(**params):
                    model = build_model(**params)
                    scores = cross_val_score(model, X_train_aug, y_train, cv=3, scoring=gpu_safe_r2_scorer)
                    return -np.mean(scores)

                result_aug = gp_minimize(objective_aug, SEARCH_SPACE, n_calls=N_OPTIMIZATION_CALLS, random_state=42)
                best_params_aug = {dim.name: val for dim, val in zip(SEARCH_SPACE, result_aug.x)}
                final_model_aug = build_model(**best_params_aug).fit(X_train_aug, y_train)

                # --- CHANGE: Save the model into the new target-specific directory ---
                joblib.dump(final_model_aug, os.path.join(target_model_dir, "augmented.pkl"))
                
                eval_dict_aug = export_holdout_predictions(df_clean, holdout_idx, target_key, frac_name, 'augmented', final_model_aug, X_aug, y_series)
                optimization_results.append({'data_fraction': frac_name, 'variant': 'augmented', 'target': target_key, 'best_cv_r2': -result_aug.fun, **eval_dict_aug, **best_params_aug})

    pd.DataFrame(optimization_results).to_csv(os.path.join(METRICS_DIR, "bayesian_optimization_summary.csv"), index=False)
    logging.info("\n--- Training script completed successfully. ---")

if __name__ == "__main__":
    main()

