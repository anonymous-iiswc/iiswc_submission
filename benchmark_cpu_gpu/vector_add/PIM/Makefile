.PHONY: all clean

DATA_TYPE=double

TOML_LIB_DIR=../../util/tomlplusplus/include

SIM_LIB_DIR=/home/farzana/research/DRAMsim3/src

SIM_LIB_DIR1=/home/farzana/research/DRAMsim3/ext/headers

all:
	g++ -o va app.cpp -ldramsim3 -L/home/farzana/research/DRAMsim3 -I$(TOML_LIB_DIR) -I$(SIM_LIB_DIR1) -I$(SIM_LIB_DIR) ../../util/pim.cpp -DDATA_TYPE=${DATA_TYPE}

clean:
	rm va
