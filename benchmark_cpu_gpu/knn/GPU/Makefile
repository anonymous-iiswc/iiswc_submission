all:
	nvcc -o kNN-CUDA.out -I../../common/inc  -I/sw/ubuntu-22.04/cuda/extras/CUPTI/include -L/sw/ubuntu-22.04/cuda/extras/CUPTI/lib64 -DONLY_TIME_KERNELS test.cpp knncuda.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets

clean:
	rm -f kNN-CUDA.out
