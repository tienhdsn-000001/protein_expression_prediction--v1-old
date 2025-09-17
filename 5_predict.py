#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5_predict.py (evaluation)

- **UPDATE (Pipeline Alignment)**:
  - Overhauled the script to align with the current pipeline's outputs.
  - Replaced separate model and scaler loading with a single `joblib.load` call
    that loads the scikit-learn `Pipeline` object. The Pipeline now handles
    data scaling internally, simplifying the prediction logic.
  - Updated the model pathing logic to match the new directory structure:
    `models/{data_fraction}/{target_name}/{variant}.pkl`.
  - Replaced the data loading mechanism with the robust `load_ecoli` function
    from the training script for consistency.
  - Aligned the `TARGETS` dictionary and embedding loading logic (`.npy` files)
    with `4_train_model.py`.
  - The `splits.json` logic remains for ad-hoc evaluations on specific subsets,
    but the core data handling is now consistent with the rest of the pipeline.
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
import csv

# Try to import cupy, but fall back gracefully if not available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.info("CuPy not found. Predictions will run on CPU.")


# ------------------------------
# Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Input Paths
MODELS_DIR = "models"
EMBEDDINGS_DIR = "embeddings"
MASTER_ANNOTATION_FILE = "Ecoli_Annotation_v2_fixed.tsv" # Use the cleaned file

# Output Path
TMP_OUTPUT_DIR = "tmp_preds"
os.makedirs(TMP_OUTPUT_DIR, exist_ok=True)

# Aligned with 4_train_model.py
PRIMARY_TARGET = "max_half_life_delta"
ABUNDANCE_TARGETS = ["log2_ppm_abundance", "log2_tpm_abundance"]
TARGETS = {
    "max_half_life_delta": "max_half_life_delta",
    "min_half_life_delta": "min_half_life_delta",
    "log2_ppm_abundance": "log2_ppm_abundance",
    "log2_tpm_abundance": "log2_tpm_abundance",
}

# ------------------------------
# Helper Functions
# ------------------------------
def load_ecoli(path: str) -> pd.DataFrame:
    """
    Robustly loads the E. coli annotation TSV, attempting to parse it as
    UTF-16 first and falling back to UTF-8.
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

def to_numpy_if_cupy(a):
    """Converts a CuPy array to NumPy; passes through other types."""
    if CUPY_AVAILABLE and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)

def load_and_prep_holdout_data(splits_file, data_size):
    """Loads and prepares the feature and target data for the holdout set."""
    logging.info("Loading and preparing holdout data...")
    df_annot = load_ecoli(MASTER_ANNOTATION_FILE)

    logging.info("Converting target columns to numeric type...")
    for target_col in TARGETS.values():
        if target_col in df_annot.columns:
            df_annot[target_col] = pd.to_numeric(df_annot[target_col], errors='coerce')

    # Load embeddings as numpy arrays, consistent with training script
    x_dna = np.load(os.path.join(EMBEDDINGS_DIR, 'dnabert_embeddings.npy'))
    x_rna = np.load(os.path.join(EMBEDDINGS_DIR, 'codonbert_embeddings.npy'))
    x_prot = np.load(os.path.join(EMBEDDINGS_DIR, 'esm2_embeddings.npy'))
    X_features = np.concatenate([x_dna, x_rna, x_prot], axis=1)

    # Ensure annotation and feature lengths match
    min_rows = min(len(df_annot), X_features.shape[0])
    df_annot = df_annot.iloc[:min_rows]
    X_features = X_features[:min_rows]
    
    # --- Split parsing logic remains for targeted evaluation ---
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    if 'holdout' in splits:
        holdout_indices = splits['holdout']
    elif data_size in splits and 'holdout' in splits[data_size]:
        holdout_indices = splits[data_size]['holdout']
    else:
        raise KeyError(f"FATAL: Could not find a valid 'holdout' key in {splits_file}")

    original_indices = df_annot.index
    valid_holdout_indices = [idx for idx in holdout_indices if idx in original_indices]

    X_holdout = X_features[valid_holdout_indices]
    df_holdout_meta = df_annot.loc[valid_holdout_indices].reset_index(drop=True)
    
    logging.info(f"Holdout data prepared. Features shape: {X_holdout.shape}, Meta shape: {df_holdout_meta.shape}")
    return X_holdout, df_holdout_meta

# ------------------------------
# Main Execution
# ------------------------------

def main(data_size, splits_file):
    """Main function to run predictions for a given data size config."""
    logging.info(f"\n--- Running Predictions for Configuration: {data_size} ---")
    
    X_holdout, df_holdout = load_and_prep_holdout_data(splits_file, data_size)
    all_predictions = []

    # --- Embeddings-Only Predictions ---
    logging.info("--- Generating predictions for embeddings-only models... ---")
    for target_key in TARGETS:
        model_path = os.path.join(MODELS_DIR, data_size, target_key, "embeddings_only.pkl")
        
        if os.path.exists(model_path):
            model_pipeline = joblib.load(model_path)
            y_pred = to_numpy_if_cupy(model_pipeline.predict(X_holdout))
            
            for i, pred_val in enumerate(y_pred):
                all_predictions.append({
                    'data_size_config': data_size, 'variant': 'embeddings_only',
                    'target': target_key,
                    'gene_name': df_holdout.iloc[i].get('Gene_Name', 'N/A'),
                    'bnumber': df_holdout.iloc[i].get('Protein_ID', 'N/A'),
                    'predicted_value': pred_val
                })
        else:
            logging.warning(f"Model not found for embeddings-only target {target_key} at {model_path}. Skipping.")

    # --- Augmented Predictions ---
    logging.info("--- Generating predictions for augmented models... ---")
    primary_model_path = os.path.join(MODELS_DIR, data_size, PRIMARY_TARGET, "embeddings_only.pkl")
    
    if os.path.exists(primary_model_path):
        primary_model_pipeline = joblib.load(primary_model_path)
        primary_preds = to_numpy_if_cupy(primary_model_pipeline.predict(X_holdout))

        # Create augmented feature set
        X_holdout_aug = np.concatenate([X_holdout, primary_preds.reshape(-1, 1), X_holdout[:, :10] * primary_preds.reshape(-1, 1)], axis=1)

        for target_key in ABUNDANCE_TARGETS:
            model_path_aug = os.path.join(MODELS_DIR, data_size, target_key, "augmented.pkl")
            
            if os.path.exists(model_path_aug):
                model_aug_pipeline = joblib.load(model_path_aug)
                y_pred_aug = to_numpy_if_cupy(model_aug_pipeline.predict(X_holdout_aug))
                
                for i, pred_val in enumerate(y_pred_aug):
                    all_predictions.append({
                        'data_size_config': data_size, 'variant': 'augmented',
                        'target': target_key,
                        'gene_name': df_holdout.iloc[i].get('Gene_Name', 'N/A'),
                        'bnumber': df_holdout.iloc[i].get('Protein_ID', 'N/A'),
                        'predicted_value': pred_val
                    })
            else:
                logging.warning(f"Model not found for augmented target {target_key} at {model_path_aug}. Skipping.")
    else:
        logging.warning(f"Primary model for augmentation not found at {primary_model_path}. Skipping all augmented predictions.")


    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        output_file = os.path.join(TMP_OUTPUT_DIR, f"{data_size}_predictions.csv")
        pred_df.to_csv(output_file, index=False)
        logging.info(f"Saved {len(pred_df)} predictions to temporary file: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions for a specific model configuration.")
    parser.add_argument("input_file", type=str, help="Path to the splits.json file.")
    parser.add_argument("data_size", type=str, help="The data size configuration to run (e.g., '100pct_data').")
    args = parser.parse_args()
    
    main(args.data_size, args.input_file)
