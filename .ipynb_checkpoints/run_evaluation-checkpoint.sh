#!/bin/bash
#
# run_evaluation.sh (Master Pipeline Script)
#
# Purpose: To execute the ENTIRE experimental pipeline from start to finish.
#
# -- UPDATE v67.0 (AI Assistant Task) --
# - REFACTOR: The final analysis step is now encapsulated in the new
#   `post_train.sh` script. This script now calls `post_train.sh`, making
#   the overall pipeline more modular and aligning with a better research
#   workflow where analysis can be re-run independently.
#
# This script will:
# 1. Dynamically find and configure the environment for CUDA libraries.
# 2. **CRITICAL**: Remove all previous results for a clean run.
# 3. Run '4_train_model.py' to train all models and generate residual files.
# 4. Execute 'post_train.sh' to perform the final analysis.

set -euo pipefail
IFS=$'\n\t'

export PATH=$PATH:/sbin:/usr/sbin

echo "--- Starting FULL Experimental Pipeline: TRAIN & ANALYZE ---"
echo "WARNING: This script will delete existing models and results and may run for several hours."
echo "---------------------------------------------------------------------"

echo "Locating NVIDIA CUDA libraries using the system cache (ldconfig)..."
NVRTC_FULL_PATH=$(ldconfig -p | grep libnvrtc.so | head -n 1 | awk '{print $NF}')

if [ -z "$NVRTC_FULL_PATH" ]; then
    echo "FATAL: Could not find libnvrtc.so. CUDA might not be installed correctly."
    exit 1
fi

CUDA_LIB_DIR=$(dirname "$NVRTC_FULL_PATH")
echo "Found REAL CUDA library directory: $CUDA_LIB_DIR"
export LD_LIBRARY_PATH=$CUDA_LIB_DIR:$LD_LIBRARY_PATH
echo "Successfully configured LD_LIBRARY_PATH."


# --- STEP 0: Clean up previous results for a clean run ---
echo "STEP 0: Cleaning up previous artifacts..."
rm -rf models/
rm -rf metrics/
rm -rf analysis_plots/
rm -f prediction_results.csv
rm -rf tmp_preds/
echo "Cleanup complete."
echo ""

# --- STEP 1: Train All Models and Generate Holdout Predictions ---
echo "======================================================================"
echo "STEP 1: Running Training Script (4_train_model.py)"
echo "This will train all models and generate all necessary holdout predictions."
echo "======================================================================"
python3 4_train_model.py
if [ $? -ne 0 ]; then
    echo "FATAL: Training script (4_train_model.py) failed."
    exit 1
fi
echo "--- Training script completed successfully. ---"
echo ""

# --- STEP 2: Run Post-Training Analysis ---
echo "======================================================================"
echo "STEP 2: Running Post-Training Analysis Script (post_train.sh)"
echo "This will consume the generated files to produce all final plots and metrics."
echo "======================================================================"
bash post_train.sh
if [ $? -ne 0 ]; then
    echo "FATAL: Post-training analysis script (post_train.sh) failed."
    exit 1
fi
echo "--- Analysis script completed successfully. ---"
echo ""


echo "======================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "Final plots are in the 'analysis_plots/' directory."
echo "Final metrics and analysis CSVs are in the 'metrics/' directory."
echo "======================================================================"
