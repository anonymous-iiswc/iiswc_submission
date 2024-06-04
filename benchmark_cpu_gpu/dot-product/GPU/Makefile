DATA_TYPE=float

# Compiler settings
NVCC := nvcc
CUDA_DIR ?= $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INCLUDE := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
CUDA_LIB := -lcublas
ARCH_FLAG := -arch=sm_80

# Target executable
TARGET := dp

# Source files
SOURCES := dot-prod.cu

# Compiler flags
CFLAGS := -G

# Default target
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(SOURCES)
	$(NVCC) $(SOURCES) -I$(CUDA_INCLUDE) $(ARCH_FLAG) -L$(CUDA_LIB_DIR) $(CUDA_LIB) $(CFLAGS) -DDATA_TYPE=$(DATA_TYPE) -o $(TARGET)

# Rule to clean the project
clean:
	rm -f $(TARGET)

