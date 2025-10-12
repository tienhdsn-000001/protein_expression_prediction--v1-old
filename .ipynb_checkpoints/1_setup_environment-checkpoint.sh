#!/bin/bash
# 1_setup_environment.sh
# Purpose: To create a consistent, modern Python environment for the project.
#
# -- UPDATE v21.0 (AI Assistant Task) --
# - ADDED: The `scikit-optimize` library has been added to support Bayesian
#   Optimization for hyperparameter tuning in the training script.
#
# -- UPDATE v20.0 (AI Assistant Task) --
# - REMOVED: Unnecessary libraries (timm, mamba-ssm, causal-conv1d).

echo "--- Starting Environment Setup ---"

# --- Fix apt-get update warning ---
echo "Disabling problematic bullseye-backports repository..."
sudo sed -i -e 's/^deb http:\/\/deb.debian.org\/debian bullseye-backports main/#deb http:\/\/deb.debian.org\/debian bullseye-backports main/g' /etc/apt/sources.list

# --- Add sbin directories to PATH for ldconfig utility ---
export PATH=$PATH:/usr/sbin:/sbin
echo "Updated PATH environment variable to include /usr/sbin and /sbin."

# --- Part 1: Install PyTorch for CUDA 11.8 ---
echo "Installing PyTorch 2.1.2 with CUDA 11.8..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# --- Part 2: Install all other packages ---
echo "Installing all other required packages..."
pip install --upgrade pip

# --- Pin transformers and accelerate for compatibility ---
echo "Installing specific versions of transformers and accelerate for PyTorch 2.1.2 compatibility..."
pip install transformers==4.35.2
pip install accelerate==0.25.0

# --- Install a compatible version of bitsandbytes ---
echo "Installing bitsandbytes==0.41.1 for PyTorch 2.1.2 compatibility..."
pip install bitsandbytes==0.41.1

pip install scikit-learn
pip install pandas

# --- Explicitly set NumPy version for compatibility ---
echo "Installing NumPy with version < 2.0 to avoid compatibility issues..."
pip uninstall -y numpy
pip install matplotlib
pip install seaborn
pip install numpy==1.26.4
pip install joblib
pip install tqdm
pip install einops
pip install safetensors
pip install triton==2.1.0
pip install xgboost
pip install bayesian-optimization
pip install lightgbm
pip install multimolecule

# --- NEW (v21.0): Install scikit-optimize for Bayesian Optimization ---
echo "Installing scikit-optimize for Bayesian Optimization..."
pip install scikit-optimize

# --- Install CuPy version that matches PyTorch's CUDA version ---
echo "Uninstalling any existing CuPy to prevent conflicts..."
pip uninstall -y cupy cupy-cuda11x cupy-cuda12x || true
echo "Installing cupy-cuda11x to match PyTorch CUDA 11.8..."
pip install "cupy-cuda11x>=13.0,<14.0"


# --- Part 3: Verify the installation ---
echo "Verifying PyTorch and CUDA installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo "Verifying CuPy installation..."
python -c "import cupy as cp; print(f'CuPy version: {cp.__version__}'); print('CuPy array created on GPU successfully.') if cp.array([1,2,3]).device.id == 0 else print('CuPy GPU test FAILED.')"


echo "--- Environment setup complete. ---"

