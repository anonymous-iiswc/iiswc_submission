DATA_TYPE=int32_t

# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -Wextra -Werror -std=c++11 -O3 -fopenmp

# Output directory
BUILD_DIR := ./build

# Executable name
TARGET := $(BUILD_DIR)/va

.PHONY: all clean

all: $(TARGET)

$(TARGET): | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DDATA_TYPE=${DATA_TYPE} -o $@ $^ vec_mul.cpp

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

clean:
	rm -r $(BUILD_DIR)
