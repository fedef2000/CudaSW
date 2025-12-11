# --------------------------------------------------------
# Makefile for CuSW (C++ + CUDA + Python)
# --------------------------------------------------------

# 1. Compiler Configurations
# --------------------------
CXX      := g++
NVCC     := nvcc
PYTHON   := python3

# 2. Auto-detect Python Paths
# ---------------------------
# We use python itself to find where the header files are
PY_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_SUFFIX  := $(shell $(PYTHON)-config --extension-suffix)

# 3. Project Paths
# ----------------
# Where your headers are
INC_FLAGS := -Iinclude -Isrc -Iextern/pybind11/include -I$(PY_INCLUDE)

# 4. Flags
# --------
# C++ Flags: -fPIC is mandatory for shared libraries
CXXFLAGS  := -O3 -Wall -shared -std=c++17 -fPIC $(INC_FLAGS)

# CUDA Flags: --compiler-options '-fPIC' passes the flag to the underlying C++ compiler
NVCCFLAGS := -O3 -std=c++17 --compiler-options '-fPIC' -Iinclude -Isrc

# 5. File Lists
# -------------
# We manually list the files we created earlier
CPP_SRCS := src/aligner.cpp python/bindings.cpp
CU_SRCS  := src/kernels.cu

# Define where object files go
OBJ_DIR  := build/obj
CPP_OBJS := $(CPP_SRCS:%.cpp=$(OBJ_DIR)/%.o)
CU_OBJS  := $(CU_SRCS:%.cu=$(OBJ_DIR)/%.o)

# The final library name (e.g., cusw.cpython-38-x86_64-linux-gnu.so)
TARGET   := cusw$(PY_SUFFIX)

# --------------------------------------------------------
# Build Rules
# --------------------------------------------------------

all: $(TARGET)

# Link Step: Combine C++ and CUDA objects into the Python library
$(TARGET): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lcudart

# Compile C++ Files
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA Files
$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean Up
clean:
	rm -rf build $(TARGET)

.PHONY: all clean
