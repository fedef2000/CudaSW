#!/bin/bash
#SBATCH --job-name=CudaSw-optimized
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpu_group5@hpc-legionlogin.polito.it
#SBATCH --partition=gpu_a100
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out-opt.log
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1


# load cuda module
module load gcc/12.4.0
module load nvhpc-hpcx-cuda12/25.1
module load nvhpc-hpcx-2.20-cuda12/25.1 

echo "Loaded NVIDIA SDK !!!"
nvidia-smi

# check the current version of the compiler:
#nvc --version              #for C
#nvc++ --version            #for C++
#nvfortran --version        #for Fortran

#nvcc --version

echo "Compiling!"

nvcc gpu_opt_version.cu -o opt.exe -O3 -arch=sm_80

./opt.exe
