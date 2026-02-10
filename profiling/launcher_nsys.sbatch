#!/bin/bash
#SBATCH --job-name=Profile-SW
#SBATCH --partition=gpu_a40
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=./out-profile.log

# 1. Load Modules (Must include cuda/nsight)
module purge
module load gcc/12.4.0
module load nvhpc-hpcx-cuda12/25.1
module load miniconda3/3.13.25

# 2. Activate Environment
source ../venv/bin/activate

# 3. Run Profiler
# -o: Output filename
# --trace=cuda,osrt,nvtx: Capture CUDA, OS Runtime, and Markers
# --force-overwrite: Overwrite old reports
echo "Starting Profiling..."


nsys profile \
    -o report_streams \
    --trace=cuda,osrt,nvtx \
    --force-overwrite true \
    python profile_streams.py

echo "Profiling Done"