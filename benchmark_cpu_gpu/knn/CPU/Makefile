# Data type definition
DATA_TYPE := int32_t

# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -Wextra -Werror -std=c++11 -O3 -fopenmp

# Output directory
BUILD_DIR := ./build

# Executable name
TARGET := $(BUILD_DIR)/knn

# Source files
SRCS := knn.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DDATA_TYPE=$(DATA_TYPE) -o $@ $^

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
