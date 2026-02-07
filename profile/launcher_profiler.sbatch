#!/bin/bash
#SBATCH --job-name=CudaSw-Lib
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpu_group5@hpc-legionlogin.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out-lib-test.log
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1

# --- 1. LOAD MODULES ---
module load gcc/12.4.0
module load nvhpc-hpcx-cuda12/25.1
module load nvhpc-hpcx-2.20-cuda12/25.1 

echo "Loaded NVIDIA SDK !!!"
nvidia-smi

# --- 2. COMPILATION ---
echo "Compiling..."

# Compile with -lineinfo
nvcc -O3 -lineinfo -arch=sm_86 main.cpp sw_cuda.cu sw_cuda_diagonal.cu sw_cpu.cpp -o sw_benchmark.exe

# 1. Run standard benchmark (for time/GCUPS)
./sw_benchmark.exe

# 2. Run Nsight Systems (for Timeline)
echo "Profiling Timeline..."
nsys profile -t cuda,osrt -o sw_timeline ./sw_benchmark.exe

# 3. Run Nsight Compute (for Kernel Details)
echo "Profiling Kernel..."
ncu --set full --launch-skip 1000 --launch-count 1 -o sw_kernel -f ./sw_benchmark.exe