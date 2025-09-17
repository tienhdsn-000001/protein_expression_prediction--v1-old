#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_generate_embeddings.py — resilient + tuple-safe

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
- DNABERT-S:
  * Try FlashAttention on CUDA.
  * On ANY error, reload with standard attention (works CPU/GPU).
  * Force return_dict=True and also handle tuple outputs defensively.

- CodonBERT: manual codon tokenization (expects HF-style local folder).
- ESM-2: standard HF path.
"""

import os
# Make common sbin locations visible so Triton can find `ldconfig` if present.
os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + "/sbin" + os.pathsep + "/usr/sbin"

import csv
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

DNABERT_S_NAME = "zhihan1996/DNABERT-S"
DNABERT_S_REV  = "00e47f96cdea35e4b6f5df89e5419cbe47d490c6"  # commit you pinned earlier

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ---------- CodonBERT ----------
def load_codon_vocab(vocab_file):
    with open(vocab_file, "r") as f: toks = [line.strip() for line in f]
    return {t: i for i, t in enumerate(toks)}

def generate_codonbert_embeddings(sequences, model_path, max_length=512):
    device = get_device()
    vocab_file = os.path.join(model_path, "vocab.txt")
    if not os.path.exists(vocab_file): raise FileNotFoundError(f"Missing vocab.txt at {vocab_file}")
    codon_to_id = load_codon_vocab(vocab_file)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    hidden = model.config.hidden_size
    cls_id = getattr(model.config, "cls_token_id", codon_to_id.get("[CLS]", 0))
    pad_id = getattr(model.config, "pad_token_id", codon_to_id.get("[PAD]", 0))
    out = []
    with torch.no_grad():
        for seq in sequences:
            if not isinstance(seq, str) or len(seq) < 3:
                out.append(np.zeros(hidden, dtype=np.float32))
                continue
            codons = [seq[i:i+3] for i in range(0, len(seq) - (len(seq) % 3), 3)]
            ids = [codon_to_id.get(c, pad_id) for c in codons]
            input_ids = ([cls_id] + ids)[:max_length]
            input_ids += [pad_id] * (max_length - len(input_ids))
            attn = [1 if t != pad_id else 0 for t in input_ids]
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.tensor([attn], dtype=torch.long, device=device)
            y = model(input_ids=input_tensor, attention_mask=attention_mask)
            last = y[0] if isinstance(y, (tuple, list)) else y.last_hidden_state
            cls = last[:, 0, :].float().cpu().numpy()[0]
            out.append(cls)
    return np.asarray(out, dtype=np.float32)

# ---------- ESM-2 ----------
def generate_esm2_embeddings(sequences, model_name="facebook/esm2_t6_8M_UR50D", max_length=512):
    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for s in sequences:
            if not isinstance(s, str) or not s:
                out.append(np.zeros(model.config.hidden_size, dtype=np.float32))
                continue
            batch = tok(s, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            y = model(**batch)
            last = y[0] if isinstance(y, (tuple, list)) else y.last_hidden_state
            mean_vec = last[:, 1:-1, :].mean(dim=1).float().cpu().numpy()[0]
            out.append(mean_vec)
    return np.asarray(out, dtype=np.float32)

# ---------- Main ----------
if __name__ == "__main__":
    max_length = 512
    os.makedirs("embeddings", exist_ok=True)

    print("Loading annotation data...")
    df = load_ecoli("Ecoli_Annotation_v2.tsv")
    print(f"Successfully loaded {len(df)} rows.")

    dna_sequences = df["DNA_Sequence_Input"].tolist()
    rna_sequences = df["mRNA_Sequence_Input"].tolist()
    protein_sequences = df["Protein_Sequence_Input"].tolist()

    print("\nGenerating DNABERT-S embeddings...")
    dnabert_embeddings = generate_dnabert_embeddings(dna_sequences, max_length=max_length)
    np.save("embeddings/dnabert_embeddings.npy", dnabert_embeddings)
    print("DNABERT-S embeddings saved.")

    print("\nGenerating CodonBERT embeddings...")
    codonbert_path = os.path.expanduser("~/data/models/codonbert_bfd")
    needed = [os.path.join(codonbert_path, x) for x in ("config.json", "pytorch_model.bin", "vocab.txt")]
    if not all(os.path.exists(p) for p in needed):
        print(f"CodonBERT model incomplete/not found at {codonbert_path}. Skipping.")
    else:
        codon_emb = generate_codonbert_embeddings(rna_sequences, codonbert_path, max_length=max_length)
        np.save("embeddings/codonbert_embeddings.npy", codon_emb)
        print("CodonBERT embeddings saved.")

    print("\nGenerating ESM-2 embeddings...")
    esm2_embeddings = generate_esm2_embeddings(protein_sequences, max_length=max_length)
    np.save("embeddings/esm2_embeddings.npy", esm2_embeddings)
    print("ESM-2 embeddings saved.")

    print("\nAll embeddings generated and saved successfully.")

