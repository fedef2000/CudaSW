# --------------------------------------------------------
# Makefile for CuSW (Robust Version)
# --------------------------------------------------------

# 1. Compiler Configurations
CXX      := g++
NVCC     := nvcc
PYTHON   := python3

# 2. Auto-detect Python Paths (Fixed for HPC)
# -------------------------------------------
# Get the include directory (e.g., /usr/include/python3.9)
PY_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")

# Get the extension suffix (e.g., .cpython-39-x86_64-linux-gnu.so) WITHOUT python3-config
PY_SUFFIX  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or sysconfig.get_config_var('SO'))")

# 3. Project Paths
INC_FLAGS := -Iinclude -Isrc -Iextern/pybind11/include -I$(PY_INCLUDE)

# 4. Flags
# -fPIC is mandatory for shared libraries
CXXFLAGS  := -O3 -Wall -shared -std=c++17 -fPIC $(INC_FLAGS)
NVCCFLAGS := -O3 -std=c++17 --compiler-options '-fPIC' -Iinclude -Isrc

# 5. File Lists
CPP_SRCS := src/aligner.cpp python/bindings.cpp
CU_SRCS  := src/kernels.cu

OBJ_DIR  := build/obj
CPP_OBJS := $(CPP_SRCS:%.cpp=$(OBJ_DIR)/%.o)
CU_OBJS  := $(CU_SRCS:%.cu=$(OBJ_DIR)/%.o)

# The final library name
TARGET   := cusw$(PY_SUFFIX)

# --------------------------------------------------------
# Build Rules
# --------------------------------------------------------

all: $(TARGET)

# Link Step
$(TARGET): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lcudart

# Compile C++
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA
$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf build $(TARGET)

.PHONY: all clean
