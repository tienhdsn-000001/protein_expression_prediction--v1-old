#!/bin/bash
#
# post_train.sh (Post-Training Analysis Script)
#
# Purpose: To run only the analysis part of the pipeline.
#
# -- NEW (v67.0, AI Assistant Task) --
# - This script was created to formalize the separation between the long-running
#   training phase and the rapid, iterative analysis phase. It can be run on its
#   own after `4_train_model.py` has successfully completed, allowing you to
#   tweak plots and metrics without re-training.
#
# This script will:
# 1. Run '6_analyze_results.py' to consume residual files from the 'metrics/'
#    directory and generate all final plots and analysis CSVs.

set -euo pipefail
IFS=$'\n\t'

echo "======================================================================"
echo "STEP 2: Running Analysis Script (6_analyze_results.py)"
echo "This will consume the summary file and generate all final plots and metrics."
echo "======================================================================"
python3 6_analyze_results.py
if [ $? -ne 0 ]; then
    echo "FATAL: Analysis script (6_analyze_results.py) failed."
    exit 1
fi
