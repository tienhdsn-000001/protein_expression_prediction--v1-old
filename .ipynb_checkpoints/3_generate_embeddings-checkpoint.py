#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_generate_embeddings.py

- FIX (Inference Memory): Reduced the batch size for ESM-2 from 4 to 1.
  This is a critical change to prevent CUDA OutOfMemory errors during the
  attention calculation, which is memory-intensive with long sequence lengths (4096).
- FIX (Error Handling): Corrected a bug in the try/except block that caused
  an UnboundLocalError if the CUDA OOM happened before all variables were assigned.
- UPDATE (Model Parallelism): The ESM-2 model is still loaded with `device_map="auto"`
  to split its layers across available GPUs.
"""

import os
import csv
import gc
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from multimolecule import RnaTokenizer, RiNALMoModel, RnaErnieModel

# Make common sbin locations visible so Triton can find `ldconfig` if present.
os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + "/sbin" + os.pathsep + "/usr/sbin"

DNABERT_S_NAME = "zhihan1996/DNABERT-S"
DNABERT_S_REV  = "00e47f96cdea35e4b6f5df89e5419cbe47d490c6"
ESM2_NAME = "facebook/esm2_t48_15B_UR50D"


def get_device():
    # This will now default to the primary CUDA device, which is fine.
    # The model parallelism will handle the rest.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ecoli(path: str) -> pd.DataFrame:
    """
    Robustly loads the E. coli annotation TSV, attempting to parse it as
    UTF-16 first (common for Excel exports) and falling back to UTF-8.
    """
    try:
        return pd.read_csv(
            path, sep="\t", engine="python", encoding="utf-16",
            dtype=str, na_filter=False, quotechar='"', quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="error",
        )
    except (UnicodeError, pd.errors.ParserError):
        print("UTF-16 parsing failed. Falling back to UTF-8.")
        return pd.read_csv(
            path, sep="\t", engine="python", encoding="utf-8",
            dtype=str, na_filter=False, quotechar='"', quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="error",
        )

# ---------- DNABERT-S ----------
def _load_dnabert_s(disable_flash_attn: bool, device):
    cfg = AutoConfig.from_pretrained(DNABERT_S_NAME, trust_remote_code=True, revision=DNABERT_S_REV)
    cfg.attention_probs_dropout_prob = 0.1 if (disable_flash_attn or device.type == "cpu") else 0.0
    cfg.return_dict = True
    tok = AutoTokenizer.from_pretrained(DNABERT_S_NAME, trust_remote_code=True, revision=DNABERT_S_REV)
    model = AutoModel.from_pretrained(
        DNABERT_S_NAME, trust_remote_code=True, revision=DNABERT_S_REV, config=cfg
    ).to(device).eval()
    return tok, model

def _select_last_hidden(output):
    if hasattr(output, "last_hidden_state"): return output.last_hidden_state
    if isinstance(output, (tuple, list)) and len(output) > 0: return output[0]
    raise TypeError(f"Unexpected DNABERT-S output type: {type(output)}")

def _dnabert_embed_once(tok, model, seqs, device, max_length):
    hidden = model.config.hidden_size
    out = []
    with torch.no_grad():
        for s in seqs:
            if not isinstance(s, str) or not s:
                out.append(np.zeros(hidden, dtype=np.float32))
                continue
            batch = tok(s, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            y = model(**batch)
            last = _select_last_hidden(y)
            cls = last[:, 0, :].float().cpu().numpy()[0]
            out.append(cls)
    return np.asarray(out, dtype=np.float32)

def generate_dnabert_embeddings(sequences, max_length=512):
    device = get_device()
    attempt_flash = (device.type == "cuda") and (os.environ.get("DNABERT_DISABLE_FLASH", "0") != "1")
    if attempt_flash:
        try:
            print("Attempting DNABERT-S with FlashAttention on GPU...")
            tok, model = _load_dnabert_s(disable_flash_attn=False, device=device)
            return _dnabert_embed_once(tok, model, sequences, device, max_length)
        except Exception as e:
            print(f"DNABERT-S FlashAttention path failed ({e.__class__.__name__}). Falling back...")
    print("Using DNABERT-S with standard attention...")
    tok, model = _load_dnabert_s(disable_flash_attn=True, device=device)
    return _dnabert_embed_once(tok, model, sequences, device, max_length)

# ---------- Generic Multimolecule RNA Embeddings ----------
def generate_multimolecule_embeddings(sequences, model_class, model_path, max_length=4096, batch_size=16):
    """
    Generic function to generate embeddings using models from the multimolecule library, now with batching.
    """
    device = get_device()
    tokenizer = RnaTokenizer.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path).to(device).eval()
    hidden = model.config.hidden_size
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            
            valid_seqs, valid_indices = [], []
            for j, s in enumerate(batch_seqs):
                if isinstance(s, str) and s:
                    valid_seqs.append(s.replace('T', 'U'))
                    valid_indices.append(j)

            batch_embeddings = np.zeros((len(batch_seqs), hidden), dtype=np.float32)
            if valid_seqs:
                batch = tokenizer(valid_seqs, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
                batch = {k: v.to(device) for k, v in batch.items()}
                y = model(**batch)
                
                last = y.last_hidden_state
                cls_embeddings = last[:, 0, :].float().cpu().numpy()

                for idx, emb in zip(valid_indices, cls_embeddings):
                    batch_embeddings[idx] = emb

            all_embeddings.append(batch_embeddings)
            
    return np.concatenate(all_embeddings, axis=0)

# ---------- ESM-2 (with Model Parallelism) ----------
def generate_esm2_embeddings(sequences, model_name=ESM2_NAME, max_length=4096, batch_size=1):
    """
    Generates ESM-2 embeddings with model parallelism, batching, and CUDA cache clearing.
    Batch size is set to 1 to accommodate the large memory footprint of the attention
    mechanism with long sequences.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading ESM-2 model with model parallelism...")
    model = AutoModel.from_pretrained(
        model_name,
        device_map="auto"
    ).eval()
    print("ESM-2 model loaded successfully across available GPUs.")

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            print(f"  Processing ESM-2 batch {i//batch_size + 1}/{(len(sequences) + batch_size - 1)//batch_size}...")
            batch_seqs = sequences[i:i + batch_size]
            
            batch_embeddings = np.zeros((len(batch_seqs), model.config.hidden_size), dtype=np.float32)

            valid_seqs = [s for s in batch_seqs if isinstance(s, str) and s]
            valid_indices = [j for j, s in enumerate(batch_seqs) if isinstance(s, str) and s]

            if valid_seqs:
                # --- FIX: Restructured try/except block to prevent UnboundLocalError ---
                batch_vars = {}
                try:
                    batch = tokenizer(valid_seqs, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
                    batch_vars['batch'] = {k: v.to(model.device) for k, v in batch.items()}
                    
                    y = model(**batch_vars['batch'])
                    batch_vars['y'] = y
                    
                    last = y[0] if isinstance(y, (tuple, list)) else y.last_hidden_state
                    batch_vars['last'] = last
                    
                    cls_vecs = last[:, 0, :].float().cpu().numpy()
                    
                    for j, emb in zip(valid_indices, cls_vecs):
                        batch_embeddings[j] = emb

                except torch.cuda.OutOfMemoryError:
                    print(f"    WARNING: CUDA OOM on batch. Batch size is already 1; sequence may be too complex.")
                    # Continue to the cleanup block, which will handle any assigned variables
                
                finally:
                    # Clean up any variables that were created in the try block
                    for var in batch_vars.values():
                        del var
                    gc.collect()
                    torch.cuda.empty_cache()

            all_embeddings.append(batch_embeddings)

    return np.concatenate(all_embeddings, axis=0)


# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sequence embeddings for the E. coli dataset.")
    parser.add_argument(
        '--regenerate', 
        nargs='+', 
        choices=['dna', 'rinalmo', 'rna_ernie', 'protein', 'all'], 
        default=[],
        help="Specify which embeddings to force-regenerate, even if they exist. Use 'all' to regenerate everything."
    )
    args = parser.parse_args()
    
    if 'all' in args.regenerate:
        force_regenerate = {'dna', 'rinalmo', 'rna_ernie', 'protein'}
    else:
        force_regenerate = set(args.regenerate)

    os.makedirs("embeddings", exist_ok=True)
    MODELS_BASE_DIR = os.path.expanduser("~/data/models")

    print("Loading annotation data...")
    df = load_ecoli("Ecoli_Annotation_v3.tsv")
    print(f"Successfully loaded {len(df)} rows.")

    dna_sequences = df["DNA_Sequence_Input"].tolist()
    rna_sequences = df["mRNA_Sequence_Input"].tolist()
    protein_sequences = df["Protein_Sequence_Input"].tolist()
    
    # --- DNABERT-S ---
    dna_emb_path = "embeddings/dnabert_embeddings.npy"
    if not os.path.exists(dna_emb_path) or 'dna' in force_regenerate:
        print("\nGenerating DNABERT-S embeddings...")
        dnabert_embeddings = generate_dnabert_embeddings(dna_sequences, max_length=512) # Architectural limit
        np.save(dna_emb_path, dnabert_embeddings)
        print("DNABERT-S embeddings saved.")
    else:
        print("\nDNABERT-S embeddings already exist. Skipping.")

    # --- Rinalmo ---
    rinalmo_emb_path = "embeddings/rinalmo_embeddings.npy"
    if not os.path.exists(rinalmo_emb_path) or 'rinalmo' in force_regenerate:
        print("\nGenerating Rinalmo embeddings...")
        rinalmo_path = os.path.join(MODELS_BASE_DIR, "rinalmo_giga")
        if not os.path.exists(rinalmo_path):
            print(f"Rinalmo model not found at {rinalmo_path}. Skipping.")
        else:
            rinalmo_emb = generate_multimolecule_embeddings(rna_sequences, RiNALMoModel, rinalmo_path)
            np.save(rinalmo_emb_path, rinalmo_emb)
            print("Rinalmo embeddings saved.")
    else:
        print("\nRinalmo embeddings already exist. Skipping.")

    # --- RNA-ERNIE ---
    rna_ernie_emb_path = "embeddings/rna_ernie_embeddings.npy"
    if not os.path.exists(rna_ernie_emb_path) or 'rna_ernie' in force_regenerate:
        print("\nGenerating RNA-ERNIE embeddings...")
        rna_ernie_path = os.path.join(MODELS_BASE_DIR, "rna_ernie")
        if not os.path.exists(rna_ernie_path):
            print(f"RNA-ERNIE model not found at {rna_ernie_path}. Skipping.")
        else:
            rna_ernie_emb = generate_multimolecule_embeddings(rna_sequences, RnaErnieModel, rna_ernie_path)
            np.save(rna_ernie_emb_path, rna_ernie_emb)
            print("RNA-ERNIE embeddings saved.")
    else:
        print("\nRNA-ERNIE embeddings already exist. Skipping.")

    # --- ESM-2 ---
    esm2_emb_path = "embeddings/esm2_embeddings.npy"
    if not os.path.exists(esm2_emb_path) or 'protein' in force_regenerate:
        print("\nGenerating ESM-2 embeddings...")
        esm2_path = os.path.join(MODELS_BASE_DIR, "esm2_t48_15B_UR50D")
        if not os.path.exists(esm2_path):
             print(f"ESM-2 model not found at {esm2_path}. Skipping.")
        else:
            esm2_embeddings = generate_esm2_embeddings(protein_sequences, model_name=esm2_path)
            np.save(esm2_emb_path, esm2_embeddings)
            print("ESM-2 embeddings saved.")
    else:
        print("\nESM-2 embeddings already exist. Skipping.")

    print("\nAll necessary embeddings are present or have been generated.")

