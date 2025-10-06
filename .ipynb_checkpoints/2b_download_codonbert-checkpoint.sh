#!/bin/bash
# 2b_download_codonbert.sh
# Purpose: To manually download and set up the CodonBERT model from its
#          official GitHub repository source, structuring it to match the
#          Hugging Face format for compatibility with downstream scripts.
#
# -- UPDATE v57.0 (AI Assistant Task) --
# - FIX: Reverted the vocab.txt download URL from the Hugging Face Hub back to the
#   original author's GitHub repository. The Hugging Face URL was causing a
#   '401 Unauthorized' error. The direct GitHub raw content URL resolves this
#   authentication failure and aligns with the project's direct-from-source methodology.

echo "--- Starting Manual Download for CodonBERT ---"

# --- Configuration ---
DEST_DIR="$HOME/data/models/codonbert_bfd"
DOWNLOAD_URL="https://cdn.prod.accelerator.sanofi/llm/CodonBERT.zip"
TEMP_ZIP_FILE="/tmp/codonbert.zip"
# --- FIX (v57.0): Corrected URL to point to the raw GitHub source ---
VOCAB_URL="https://raw.githubusercontent.com/jerryji1993/CodonBERT/master/CodonBERT-BFD-10M/vocab.txt"


# --- Pre-flight Checks ---
echo "Checking for existing CodonBERT setup in: $DEST_DIR"
echo "Verifying presence of: config.json, pytorch_model.bin, vocab.txt"

if [ -d "$DEST_DIR" ] && [ -f "$DEST_DIR/config.json" ] && [ -f "$DEST_DIR/pytorch_model.bin" ] && [ -f "$DEST_DIR/vocab.txt" ]; then
    echo "CodonBERT model and tokenizer already appear to be correctly set up. Skipping download."
    exit 0
fi

# If the directory exists but files are missing, we should clear it to ensure a clean slate.
if [ -d "$DEST_DIR" ]; then
    echo "Incomplete installation found. Clearing directory for a fresh download."
    rm -rf "$DEST_DIR"/*
else
    mkdir -p "$DEST_DIR"
    echo "Destination directory created at: $DEST_DIR"
fi


# --- Download the core model files ---
echo "Downloading CodonBERT model from official source..."
wget -O "$TEMP_ZIP_FILE" "$DOWNLOAD_URL"

if [ $? -ne 0 ]; then
    echo "FATAL: Download failed. Please check the URL and your internet connection."
    rm -f "$TEMP_ZIP_FILE"
    exit 1
fi

# --- Unpack and Organize ---
echo "Download complete. Unzipping model files..."
unzip -o "$TEMP_ZIP_FILE" -d "$DEST_DIR"

if [ $? -ne 0 ]; then
    echo "FATAL: Unzip failed. The downloaded file may be corrupt."
    rm -f "$TEMP_ZIP_FILE"
    exit 1
fi

# Check for a subdirectory and move contents up if necessary
if [ -d "$DEST_DIR/CodonBERT-BFD-10M" ]; then
    echo "Moving model files from subdirectory 'CodonBERT-BFD-10M' to the correct location..."
    mv "$DEST_DIR/CodonBERT-BFD-10M/"* "$DEST_DIR/"
    rm -rf "$DEST_DIR/CodonBERT-BFD-10M"
fi
if [ -d "$DEST_DIR/codonbert" ]; then
    echo "Moving model files from subdirectory 'codonbert' to the correct location..."
    mv "$DEST_DIR/codonbert/"* "$DEST_DIR/"
    rm -rf "$DEST_DIR/codonbert"
fi
# Clean up any junk folders
rm -rf "$DEST_DIR/__MACOSX"


echo "Cleaning up temporary zip file..."
rm -f "$TEMP_ZIP_FILE"

# --- Download tokenizer vocab file from the stable GitHub source ---
echo "Downloading tokenizer vocabulary file from the GitHub repository..."
wget -O "$DEST_DIR/vocab.txt" "$VOCAB_URL"

if [ $? -ne 0 ]; then
    echo "FATAL: Failed to download vocab.txt from GitHub."
    echo "Please check your internet connection and the URL: $VOCAB_URL"
    exit 1
fi

# --- Verification ---
if [ -f "$DEST_DIR/config.json" ] && [ -f "$DEST_DIR/pytorch_model.bin" ] && [ -f "$DEST_DIR/vocab.txt" ]; then
    echo "--- CodonBERT setup completed successfully. ---"
else
    echo "--- CodonBERT setup failed. Required files are missing. ---"
    exit 1
fi

