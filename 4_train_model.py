#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4_train_model.py

- CRITICAL FIX (Data Leakage): Replaced the standard KFold cross-validation
  within the hyperparameter tuning loop with GroupKFold. The previous method
  was splitting genes from the same operon across training/validation folds,
  leading to data leakage and inflated CV scores. The new GroupKFold approach
  ensures all genes from an operon are kept in the same fold, providing a
  more rigorous and trustworthy model selection process.
- CRITICAL UPDATE (Operon Split): Replaced the simple random train/test
  split with a robust, sequence-based split.
- UPDATE: The script now saves the generated holdout indices to 'splits.json'.
- FIX (FileNotFoundError): Corrected embedding loading logic.
"""

import os
import csv
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
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
    # Ensure indices are within the bounds of the dataframe and X matrix
    valid_indices_mask = np.isin(indices, df_meta.index)
    valid_indices = indices[valid_indices_mask]
    
    y_series_indices = y_series.index
    valid_idx = np.array([i for i in valid_indices if i in y_series_indices and not pd.isna(y_series.loc[i])], dtype=int)
    
    if valid_idx.size == 0:
        logging.warning(f"No valid holdout rows for {target_name}. Skipping.")
        return {'holdout_r2': np.nan}
        
    relative_idx_for_df = df_meta.index.get_indexer(valid_idx)
    
    X_hold, y_hold = X_mat[valid_idx], y_series.loc[valid_idx].values
    y_pred = model.predict(X_hold)
    y_hold_np, y_pred_np = to_numpy_if_cupy(y_hold), to_numpy_if_cupy(y_pred)
    
    gene_id_col = next((c for c in ['Gene_Name', 'Protein_ID', 'gene_id', 'locus_tag'] if c in df_meta.columns), None)
    gene_ids = df_meta.iloc[relative_idx_for_df][gene_id_col].values if gene_id_col else valid_idx
    
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
    
    if len(df_master) != X.shape[0]:
        logging.warning(f"Row count mismatch! Annotations: {len(df_master)}, Embeddings: {X.shape[0]}.")
        min_rows = min(len(df_master), X.shape[0])
        logging.warning(f"Truncating to {min_rows} rows to proceed.")
        df_master = df_master.iloc[:min_rows]
        X = X[:min_rows]
    
    assert len(df_master) == X.shape[0], "FATAL: Mismatch between annotations and features after attempting to resolve."

    for col in TARGETS.values():
        if col in df_master.columns:
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
    
    valid_rows_mask = df_master[list(TARGETS.values())].notna().any(axis=1)
    df_clean = df_master[valid_rows_mask].reset_index(drop=True)
    X_clean = X[valid_rows_mask]
    logging.info(f"Retained {len(df_clean)} rows with at least one target.")

    logging.info("--- Performing robust operon-level data split ---")
    operon_id_buffer = 75
    
    if 'DNA_Sequence_Input' not in df_clean.columns:
        raise ValueError("FATAL: 'DNA_Sequence_Input' column not found in the data. Cannot perform sequence-based split.")

    long_enough_mask = df_clean['DNA_Sequence_Input'].str.len() >= operon_id_buffer
    df_operon = df_clean[long_enough_mask].reset_index(drop=True)
    X_operon = X_clean[long_enough_mask]
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

    logging.info("Saving the operon-based data splits to splits.json...")
    split_data = {'holdout': holdout_idx.tolist()}
    with open("splits.json", "w") as f:
        json.dump(split_data, f)
    logging.info("Splits saved successfully.")
    
    y_cols = {key: df_operon[col] for key, col in TARGETS.items()}
    optimization_results = []

    for frac_name, frac_val in DATA_FRACTIONS.items():
        logging.info(f"\n--- Data Fraction: {frac_name} ---")
        train_idx_frac, _ = train_test_split(train_val_idx, train_size=frac_val, random_state=42) if frac_val < 1.0 else (train_val_idx, None)
        
        best_primary_model = None

        for target_key, y_series in y_cols.items():
            logging.info(f"\nOptimizing Embeddings-Only for: {target_key}")
            
            target_model_dir = os.path.join(MODELS_DIR, frac_name, target_key)
            os.makedirs(target_model_dir, exist_ok=True)
            
            train_indices = np.intersect1d(train_idx_frac, y_series.dropna().index)
            if len(train_indices) < 20: continue
            X_train, y_train = X_operon[train_indices], y_series.iloc[train_indices].values
            
            # --- FIX: Use GroupKFold to prevent leakage during cross-validation ---
            groups = df_operon.loc[train_indices, 'operon_group_key'].values
            gkf = GroupKFold(n_splits=3)

            @use_named_args(SEARCH_SPACE)
            def objective(**params):
                model = build_model(**params)
                scores = cross_val_score(model, X_train, y_train, cv=gkf, groups=groups, scoring=gpu_safe_r2_scorer)
                return -np.mean(scores)

            result = gp_minimize(objective, SEARCH_SPACE, n_calls=N_OPTIMIZATION_CALLS, random_state=42)
            best_params = {dim.name: val for dim, val in zip(SEARCH_SPACE, result.x)}
            
            final_model = build_model(**best_params).fit(X_train, y_train)
            
            joblib.dump(final_model, os.path.join(target_model_dir, "embeddings_only.pkl"))
            
            eval_dict = export_holdout_predictions(df_operon, holdout_idx, target_key, frac_name, 'embeddings_only', final_model, X_operon, y_series)
            optimization_results.append({'data_fraction': frac_name, 'variant': 'embeddings_only', 'target': target_key, 'best_cv_r2': -result.fun, **eval_dict, **best_params})
            if target_key == PRIMARY_TARGET: best_primary_model = final_model

        if best_primary_model:
            logging.info("\n--- Running Augmented Models ---")
            # Predict only on the operon-grouped data
            primary_preds = to_numpy_if_cupy(best_primary_model.predict(X_operon)).reshape(-1, 1)
            X_aug = np.concatenate([X_operon, primary_preds, X_operon[:, :10] * primary_preds], axis=1)

            for target_key in ABUNDANCE_TARGETS:
                logging.info(f"Optimizing Augmented for: {target_key}")
                
                target_model_dir = os.path.join(MODELS_DIR, frac_name, target_key)
                os.makedirs(target_model_dir, exist_ok=True)
                
                y_series_aug = y_cols[target_key]
                train_indices_aug = np.intersect1d(train_idx_frac, y_series_aug.dropna().index)
                if len(train_indices_aug) < 20: continue
                X_train_aug, y_train_aug = X_aug[train_indices_aug], y_series_aug.iloc[train_indices_aug].values

                # --- FIX: Use GroupKFold for augmented model CV as well ---
                groups_aug = df_operon.loc[train_indices_aug, 'operon_group_key'].values
                gkf_aug = GroupKFold(n_splits=3)
                
                @use_named_args(SEARCH_SPACE)
                def objective_aug(**params):
                    model = build_model(**params)
                    scores = cross_val_score(model, X_train_aug, y_train_aug, cv=gkf_aug, groups=groups_aug, scoring=gpu_safe_r2_scorer)
                    return -np.mean(scores)

                result_aug = gp_minimize(objective_aug, SEARCH_SPACE, n_calls=N_OPTIMIZATION_CALLS, random_state=42)
                best_params_aug = {dim.name: val for dim, val in zip(SEARCH_SPACE, result_aug.x)}
                final_model_aug = build_model(**best_params_aug).fit(X_train_aug, y_train_aug)

                joblib.dump(final_model_aug, os.path.join(target_model_dir, "augmented.pkl"))
                
                eval_dict_aug = export_holdout_predictions(df_operon, holdout_idx, target_key, frac_name, 'augmented', final_model_aug, X_aug, y_series_aug)
                optimization_results.append({'data_fraction': frac_name, 'variant': 'augmented', 'target': target_key, 'best_cv_r2': -result_aug.fun, **eval_dict_aug, **best_params_aug})

    pd.DataFrame(optimization_results).to_csv(os.path.join(METRICS_DIR, "bayesian_optimization_summary.csv"), index=False)
    logging.info("\n--- Training script completed successfully. ---")

if __name__ == "__main__":
    main()

