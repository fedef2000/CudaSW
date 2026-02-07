#!/bin/bash
#SBATCH --job-name=Cpp-Bench
#SBATCH --partition=gpu_a40
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --output=../out-cpp-run.log

# Load necessary modules
module load gcc/12.4.0
module load nvhpc-hpcx-cuda12/25.1
module load nvhpc-hpcx-2.20-cuda12/25.1 

echo "Compiling C++ Benchmark..."

# 1. Define Paths
BENCH_SRC="cpp_benchmark.cpp"
LIB_DIR="../sw_cuda_py/src"

# 2. Compile
# -std=c++17 is required for std::filesystem
# We compile the benchmark AND the library sources together
nvcc -O3 -arch=native -std=c++17 -o cpp_benchmark.exe \
    $BENCH_SRC \
    $LIB_DIR/sw_cpu.cpp \
    $LIB_DIR/sw_cuda_tiled.cu \
    $LIB_DIR/sw_cuda_diagonal.cu \
    $LIB_DIR/sw_cuda_o2m.cu

if [ $? -eq 0 ]; then
    echo "Compilation successful! Created 'cpp_benchmark.exe'"
else
    echo "Compilation failed."
    exit 1
fi

echo "Starting C++ Benchmark..."
./cpp_benchmark.exe
