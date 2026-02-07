cmake_minimum_required(VERSION 3.18)
project(sw_cuda_py LANGUAGES CXX CUDA)

# --- 1. Find Dependencies ---
# CRITICAL FIX: Explicitly find the CUDA Toolkit libraries
find_package(CUDAToolkit REQUIRED)

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# --- 2. Define Source Files ---
# Ensure these match your actual files in sw_cuda_py/src/
set(SOURCE_FILES 
    sw_cuda_py/src/bindings.cpp 
    sw_cuda_py/src/sw_cpu.cpp
    sw_cuda_py/src/sw_cuda_diagonal.cu
    sw_cuda_py/src/sw_cuda_tiled.cu
    sw_cuda_py/src/sw_cuda_o2m.cu
)

# --- 3. Create the Target ---
python_add_library(sw_cuda_py MODULE ${SOURCE_FILES})

# --- 4. Link Libraries ---
# Now CUDA::cudart will exist because we called find_package(CUDAToolkit)
target_link_libraries(sw_cuda_py PRIVATE pybind11::headers CUDA::cudart)

# --- 5. Compilation Properties ---
set_target_properties(sw_cuda_py PROPERTIES
    CXX_STANDARD 14
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Optimization flags
if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
    target_compile_options(sw_cuda_py PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()