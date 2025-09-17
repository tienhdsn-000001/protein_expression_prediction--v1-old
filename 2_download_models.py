# 2_download_models.py
# Purpose: To intelligently download and cache the specified models for the pipeline.
#
# -- UPDATE v54.0 (AI Assistant Task) --
# - REVERSION: The model list has been completely changed to support the new
#   three-model (DNA, RNA, Protein) pipeline.
# - REMOVED: NT, DNABERT2, Evo2, Aidorna, and Orthrus were removed.
# - ADDED: The ESM-2 650M model ('facebook/esm2_t33_650M_UR50D') has been added.
# - RETAINED: 'dnabert_s' and 'codonbert' are still included.

# Import necessary libraries
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download, HfApi, HfFolder
import os
import logging
import shutil
import glob
import sys
import stat
import json
try:
    from huggingface_hub import HfHubHTTPError
except ImportError:
    from requests import HTTPError as HfHubHTTPError


# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CACHE_DIR = "/tmp/hf_cache"
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
    logging.info(f"--- Set Hugging Face cache directory to: {CACHE_DIR} ---")
except OSError as e:
    logging.error(f"FATAL: Could not create or access cache directory at {CACHE_DIR}. Please check permissions.")
    logging.error(f"Underlying error: {e}")
    sys.exit(1)

MODELS_BASE_DIR = os.path.expanduser("~/data/models")
os.makedirs(MODELS_BASE_DIR, exist_ok=True)
logging.info(f"--- Models will be saved to: {MODELS_BASE_DIR} ---")


# --- UPDATE (v54.0): New model list for the 3-part pipeline ---
MODELS_TO_DOWNLOAD = {
    "dnabert_s": {
        "name": "zhihan1996/DNABERT-S",
        "path": os.path.join(MODELS_BASE_DIR, "dnabert_s"),
        "type": "dna"
    },
    "codonbert": {
        "name": "microsoft/codon-bert-bfd",
        "path": os.path.join(MODELS_BASE_DIR, "codonbert_bfd"),
        "type": "rna"
    },
    "esm2": {
        "name": "facebook/esm2_t33_650M_UR50D",
        "path": os.path.join(MODELS_BASE_DIR, "esm2_t33_650M_UR50D"),
        "type": "protein"
    }
}


# --- Functions ---

def check_hf_login():
    """Checks if the user is logged into Hugging Face and provides instructions if not."""
    token = HfFolder.get_token()
    if token is None:
        logging.info("Hugging Face token not found. Most public models will still download fine.")
        logging.info("If you encounter issues with private/gated models, run: `huggingface-cli login`")
    else:
        logging.info("Hugging Face token found.")
    return True

def force_remove_directory(directory_path):
    """Forcefully removes a directory and its contents."""
    if not os.path.isdir(directory_path):
        return
    logging.warning(f"Attempting to forcefully remove incomplete directory: {directory_path}")
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for name in files:
            filepath = os.path.join(root, name)
            os.chmod(filepath, stat.S_IWUSR)
            os.remove(filepath)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    shutil.rmtree(directory_path)
    logging.info(f"Successfully cleaned up directory: '{directory_path}'")

def is_model_downloaded(local_path):
    """Performs a basic check to see if a model directory seems to exist and has content."""
    if not os.path.isdir(local_path):
        return False

    if not os.path.isfile(os.path.join(local_path, "config.json")):
        logging.warning(f"Verification check failed for '{local_path}': missing config.json.")
        return False

    has_weights = (
        glob.glob(os.path.join(local_path, "*.bin")) or
        glob.glob(os.path.join(local_path, "*.safetensors")) or
        glob.glob(os.path.join(local_path, "*.pt*"))
    )
    if not has_weights:
        logging.warning(f"Verification check failed for '{local_path}': missing model weight files.")
        return False

    logging.info(f"Model in '{local_path}' appears to be present and downloaded.")
    return True


def download_model(model_name, local_path, model_key):
    """Downloads a specified Hugging Face model, with pre-flight checks."""
    if is_model_downloaded(local_path):
        logging.info(f"Model '{model_name}' already downloaded in '{local_path}'. Skipping.")
        return True

    if model_key == 'codonbert':
        logging.warning(f"'{model_key}' is manually downloaded. Please run '2b_download_codonbert.sh'. Skipping automated download.")
        return is_model_downloaded(local_path)

    api = HfApi()
    try:
        logging.info(f"Verifying repository '{model_name}' exists on Hugging Face Hub...")
        api.model_info(repo_id=model_name, token=True)
        logging.info(f"Repository '{model_name}' verified successfully.")
    except HfHubHTTPError as e:
        if hasattr(e, 'response') and (e.response.status_code == 401 or e.response.status_code == 404):
            logging.error(f"FATAL: Repository '{model_name}' not found or access is denied.")
            logging.error(f"Please check the model name and ensure you are logged in (`huggingface-cli login`) with an account that has access.")
            return False
        else:
            raise

    logging.info(f"Starting download for model '{model_name}' to '{local_path}'...")
    
    if os.path.isdir(local_path):
        force_remove_directory(local_path)

    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.h5", "*.flax_model.msgpack", "*.gguf"],
            token=True,
        )
        logging.info(f"Model '{model_name}' downloaded successfully to '{local_path}'.")

    except Exception as e:
        logging.error(f"Download failed for '{model_name}': {e}", exc_info=False)
        if "No space left on device" in str(e):
            logging.critical("FATAL: Not enough disk space to download the model.")
        force_remove_directory(local_path)
        return False
    
    if not is_model_downloaded(local_path):
        logging.error(f"Post-download verification failed for '{model_name}'. The downloaded files may be incomplete.")
        return False
        
    return True

# --- Main Execution Logic ---
if __name__ == "__main__":
    logging.info("--- Starting Model Download and Verification Process ---")
    
    check_hf_login()

    for model_key, model_info in MODELS_TO_DOWNLOAD.items():
        print("-" * 50)
        logging.info(f"Processing model: {model_key.upper()}")
        download_model(
            model_info['name'],
            model_info['path'],
            model_key=model_key,
        )

    print("-" * 50)
    
    all_models_ok = True
    logging.info("--- Running Final Verification for All Models ---")
    for key, info in MODELS_TO_DOWNLOAD.items():
        if not is_model_downloaded(info['path']):
            logging.error(f"Final verification FAILED for model: {key.upper()} at '{info['path']}'")
            all_models_ok = False
        else:
            logging.info(f"Final verification PASSED for model: {key.upper()}")


    if all_models_ok:
        logging.info("--- All specified models are downloaded and ready for use. ---")
        try:
            logging.info(f"Cleaning up cache directory: {CACHE_DIR}")
            shutil.rmtree(CACHE_DIR)
            logging.info("Cleanup successful.")
        except OSError as e:
            logging.error(f"Could not remove cache directory {CACHE_DIR}: {e}")
    else:
        logging.error("--- Model download process failed or is incomplete. Not all models are ready. ---")
        sys.exit(1)
