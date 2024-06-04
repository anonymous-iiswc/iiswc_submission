.PHONY: all clean

DATA_TYPE=int32_t

all:
	g++ -o hist -fopenmp -O3 hist-pthread.cpp -DDATA_TYPE=${DATA_TYPE}

clean:
	rm hist 
    
