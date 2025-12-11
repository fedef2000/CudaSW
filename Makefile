# --------------------------------------------------------
# Makefile for CuSW (Robust Linker Version)
# --------------------------------------------------------

# 1. Compiler Configurations
CXX      := g++
NVCC     := nvcc
PYTHON   := python3

# 2. Auto-detect Paths
# --------------------
# Python Includes
PY_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_SUFFIX  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or sysconfig.get_config_var('SO'))")

# CUDA Library Path (CRITICAL FIX)
# We ask 'which nvcc' to find the path, then strip '/bin/nvcc' and add '/lib64'
CUDA_PATH := $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_LIB_PATH := /share/apps/hpc_sdk/Linux_x86_64/25.1/cuda/12.6/targets/x86_64-linux/lib/ 

# 3. Project Paths
INC_FLAGS := -Iinclude -Isrc -Iextern/pybind11/include -I$(PY_INCLUDE)
LIB_FLAGS := -L$(CUDA_LIB_PATH) -lcudart  # <--- Added -L here

# 4. Flags
CXXFLAGS  := -O3 -Wall -shared -std=c++17 -fPIC $(INC_FLAGS)
NVCCFLAGS := -O3 -std=c++17 --compiler-options '-fPIC' -Iinclude -Isrc

# 5. File Lists
CPP_SRCS := src/aligner.cpp python/bindings.cpp
CU_SRCS  := src/kernels.cu

OBJ_DIR  := build/obj
CPP_OBJS := $(CPP_SRCS:%.cpp=$(OBJ_DIR)/%.o)
CU_OBJS  := $(CU_SRCS:%.cu=$(OBJ_DIR)/%.o)

TARGET   := cusw$(PY_SUFFIX)

# --------------------------------------------------------
# Build Rules
# --------------------------------------------------------

all: $(TARGET)

# Link Step: Note the $(LIB_FLAGS) at the end
$(TARGET): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIB_FLAGS)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf build $(TARGET)

.PHONY: all clean
