#!/bin/bash
#SBATCH --job-name=CudaSw-Lib
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpu_group5@hpc-legionlogin.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out-streams.log
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

# We compile main.cpp AND the CUDA implementations together.
# -O3 enables optimization (critical for benchmarking).
# -arch: 
#        + sm_86 A40 
#        + sm_80 A100 
#        + sm_70 V100 
#        + sm_53 Jetson nano 
# If -arch fails, remove it or use -arch=all
nvcc -O3 -lineinfo -arch=native streams_benchmark.cpp sw_cuda_tiled.cu sw_cuda_diagonal.cu sw_cpu.cpp sw_cuda_o2m.cu -o sw_benchmark.exe

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
else
    echo "Compilation failed!"
    exit 1
fi

# --- 3. EXECUTION ---
echo "Running Benchmark..."
./sw_benchmark.exe 


echo "Done!"