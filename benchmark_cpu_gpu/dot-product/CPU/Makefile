.PHONY: all clean

DATA_TYPE=int32_t

all:
	g++ -o dp -fopenmp -O3 dot-prod.cpp -DDATA_TYPE=${DATA_TYPE}

clean:
	rm dp
