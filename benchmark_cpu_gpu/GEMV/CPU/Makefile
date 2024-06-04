.PHONY: all clean

DATA_TYPE=int32_t

all:
	g++ -o gemv -fopenmp -O3 gemv.cpp -DDATA_TYPE=${DATA_TYPE}

clean:
	rm gemv 
