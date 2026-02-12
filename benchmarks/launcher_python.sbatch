#!/bin/bash
#SBATCH --job-name=Run-SwPy
#SBATCH --partition=gpu_a40
#SBATCH --time=00:010:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./out-benchmarks-benchmark.log
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

# --- 1. LOAD MODULES ---
# MUST match the build modules exactly
module purge
module load gcc/12.4.0
module load nvhpc-hpcx-cuda12/25.1
module load miniconda3/3.13.25

# --- 2. ACTIVATE LIBRARY ---
# The venv is in the PARENT directory (root), so we use ../venv
source ../venv/bin/activate

# --- 3. RUN PYTHON ---
echo "Starting Python Benchmark..."
python benchmark.py

echo "Job Finished."