#!/bin/bash
#SBATCH --job-name=Build-PyLib
#SBATCH --partition=gpu_a40
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out-build.log
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

# --- 1. LOAD MODULES ---
module purge
module load gcc/12.4.0
module load nvhpc-hpcx-cuda12/25.1

# !!! CHANGE THIS to the module you found in Step 1 !!!
# Examples: module load python/3.11   OR   module load anaconda3
module load miniconda3/3.13.25

# --- 2. FORCE CMAKE TO USE GCC ---
export CC=gcc
export CXX=g++
export CUDACXX=nvcc

# --- 3. CLEAN START (Critical) ---
# We must delete the old venv because it points to the broken system python
echo "Cleaning old environment..."
rm -rf venv
rm -rf sw_cuda_py/build sw_cuda_py/_skbuild sw_cuda_py/*.egg-info

# --- 4. SETUP NEW VIRTUAL ENVIRONMENT ---
# We use 'python' (from the module), not 'python3' (system)
echo "Creating new venv with: $(which python)"
python -m venv venv
source venv/bin/activate

# --- 5. INSTALL DEPENDENCIES ---
pip install --upgrade pip
pip install pybind11 cmake scikit-build ninja

# --- 6. BUILD ---
echo "Building sw_cuda_py..."
cd sw_cuda_py
pip install . -v

if [ $? -eq 0 ]; then
    echo "SUCCESS: Library installed!"
else
    echo "FAILURE: Build failed check logs."
    exit 1
fi