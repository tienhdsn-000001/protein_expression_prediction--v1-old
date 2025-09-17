#!/bin/bash


bash 1_setup_environment.sh

bash 2b_download_codonbert.sh
python 2_download_models.py

python 3_generate_embeddings.py

bash run_evaluation.sh
