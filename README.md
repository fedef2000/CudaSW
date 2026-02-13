# CudaSW
A python library to execute Smith-Waterman on a GPU.

All the source files are inside the folder "sw_cuda_py/src".

To build the python library is sufficient to launch the file "launcher_build.sbatch" inside the HPC server with the command `sbatch launcher_build.sbatch`. This will build the python library.

Inside the folder "benchmarks" there are two files to perform benchmarks of one-to-one and on one-to-many. These benchmarks can be launcher using the corresponding "launcher_python.sbatch"

Inside the folder "profiling" there's a script for each implemented version and, it can be selected which one to profile updating the script "launcher_ncu.sbatch" and "launcher-nsys.sbatch".
