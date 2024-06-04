/* File:     vec_add.cu
 * Purpose:  Implement vector addition on a gpu using cuda
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "../../util/common.h"

using namespace std;

#define BLOCK_SIZE 512

void initData(uint64_t numPoints, data_t *dataPoints, uint64_t n_pad)
{
    // Providing a seed value
    srand((unsigned)time(NULL));
    for (uint64_t i = 0; i < n_pad; i++)
    {
        for (uint64_t j = 0; j < 2; j++)
        {
            uint64_t index = j * n_pad + i;
            if (i >= numPoints)
            {
                dataPoints[index] = 0;
            }
            else
            {
                dataPoints[index] = rand() % 16;
            }
        }
    }
}

__device__ void warpReduce(uint64_t tid, volatile data_t *dataArray)
{
    dataArray[tid] += dataArray[tid + 32];
    dataArray[tid] += dataArray[tid + 16];
    dataArray[tid] += dataArray[tid + 8];
    dataArray[tid] += dataArray[tid + 4];
    dataArray[tid] += dataArray[tid + 2];
    dataArray[tid] += dataArray[tid + 1];
}

__global__ void LR(data_t *points, data_t *SX, data_t *SY, data_t *SXX, data_t *SYY, data_t *SXY, uint64_t n_pad, uint64_t n)
{
    uint64_t threadID = threadIdx.x;
    uint64_t index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    __shared__ data_t partialSumSX[BLOCK_SIZE], partialSumSY[BLOCK_SIZE], partialSumSXX[BLOCK_SIZE], partialSumSYY[BLOCK_SIZE], partialSumSXY[BLOCK_SIZE];
    if (index + blockDim.x < n)
    {

        partialSumSX[threadID] = points[index + n_pad * 0] + points[index + blockDim.x + n_pad * 0];
        partialSumSY[threadID] = points[index + n_pad * 1] + points[index + blockDim.x + n_pad * 1];
        partialSumSXX[threadID] = (points[index + n_pad * 0] * points[index + n_pad * 0]) + (points[index + blockDim.x + n_pad * 0] * points[index + blockDim.x + n_pad * 0]);
        partialSumSYY[threadID] = points[index + n_pad * 1] * points[index + n_pad * 1] + points[index + blockDim.x + n_pad * 1] * points[index + blockDim.x + n_pad * 1];
        partialSumSXY[threadID] = points[index + n_pad * 0] * points[index + n_pad * 1] + points[index + blockDim.x + n_pad * 0] * points[index + blockDim.x + n_pad * 1];
    }
    else
    {
        partialSumSX[threadID] = points[index + n_pad * 0];
        partialSumSY[threadID] = points[index + n_pad * 1];
        partialSumSXX[threadID] = (points[index + n_pad * 0] * points[index + n_pad * 0]);
        partialSumSYY[threadID] = points[index + n_pad * 1] * points[index + n_pad * 1];
        partialSumSXY[threadID] = points[index + n_pad * 0] * points[index + n_pad * 1];
    }
    __syncthreads();
    for (int i = blockDim.x / 2; i > 32; i >>= 1)
    {
        uint64_t currIndex = threadID;
        if (currIndex < i)
        {
            partialSumSX[currIndex] += partialSumSX[currIndex + i];
            partialSumSY[currIndex] += partialSumSY[currIndex + i];
            partialSumSXX[currIndex] += partialSumSXX[currIndex + i];
            partialSumSYY[currIndex] += partialSumSYY[currIndex + i];
            partialSumSXY[currIndex] += partialSumSXY[currIndex + i];
        }
        __syncthreads();
    }

    if (threadID < 32)
    {
        warpReduce(threadID, partialSumSX);
        warpReduce(threadID, partialSumSY);
        warpReduce(threadID, partialSumSXX);
        warpReduce(threadID, partialSumSYY);
        warpReduce(threadID, partialSumSXY);
    }

    if (threadID == 0)
    {
        SX[blockIdx.x] = partialSumSX[0];
        SY[blockIdx.x] = partialSumSY[0];
        SXX[blockIdx.x] = partialSumSXX[0];
        SYY[blockIdx.x] = partialSumSYY[0];
        SXY[blockIdx.x] = partialSumSXY[0];
    }
}

__global__ void LR_Wrap(data_t *SX, data_t *SY, data_t *SXX, data_t *SYY, data_t *SXY, uint64_t n)
{
    int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int gridSize = BLOCK_SIZE * 2 * gridDim.x;
    __shared__ data_t partialSumSX[BLOCK_SIZE], partialSumSY[BLOCK_SIZE], partialSumSXX[BLOCK_SIZE], partialSumSYY[BLOCK_SIZE], partialSumSXY[BLOCK_SIZE];
    partialSumSX[threadIdx.x] = 0;
    partialSumSY[threadIdx.x] = 0;
    partialSumSXX[threadIdx.x] = 0;
    partialSumSYY[threadIdx.x] = 0;
    partialSumSXY[threadIdx.x] = 0;
    while (index < n)
    {
        if (index + blockDim.x < n)
        {
            partialSumSX[threadIdx.x] += SX[index] + SX[index + blockDim.x];
            partialSumSY[threadIdx.x] += SY[index] + SY[index + blockDim.x];
            partialSumSXX[threadIdx.x] += SXX[index] + SXX[index + blockDim.x];
            partialSumSYY[threadIdx.x] += SYY[index] + SYY[index + blockDim.x];
            partialSumSXY[threadIdx.x] += SXY[index] + SXY[index + blockDim.x];
        }
        else
        {
            partialSumSX[threadIdx.x] += SX[index];
            partialSumSY[threadIdx.x] += SY[index];
            partialSumSXX[threadIdx.x] += SXX[index];
            partialSumSYY[threadIdx.x] += SYY[index];
            partialSumSXY[threadIdx.x] += SXY[index];
        }
        index += gridSize;
    }
    __syncthreads();
    for (int i = blockDim.x / 2; i > 32; i >>= 1)
    {
        if (threadIdx.x < i)
        {
            partialSumSX[threadIdx.x] += partialSumSX[threadIdx.x + i];
            partialSumSY[threadIdx.x] += partialSumSY[threadIdx.x + i];
            partialSumSXX[threadIdx.x] += partialSumSXX[threadIdx.x + i];
            partialSumSYY[threadIdx.x] += partialSumSYY[threadIdx.x + i];
            partialSumSXY[threadIdx.x] += partialSumSXY[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        warpReduce(threadIdx.x, partialSumSX);
        warpReduce(threadIdx.x, partialSumSY);
        warpReduce(threadIdx.x, partialSumSXX);
        warpReduce(threadIdx.x, partialSumSYY);
        warpReduce(threadIdx.x, partialSumSXY);
    }

    if (threadIdx.x == 0)
    {
        SX[blockIdx.x] = partialSumSX[0];
        SY[blockIdx.x] = partialSumSY[0];
        SXX[blockIdx.x] = partialSumSXX[0];
        SYY[blockIdx.x] = partialSumSYY[0];
        SXY[blockIdx.x] = partialSumSXY[0];
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << argc << "\n";
        printf("Vector size required.\n");
        printf("Syntax: %s <vector_size>.\n", argv[0]);
        exit(1);
    }

    uint64_t n = atoll(argv[1]);
    data_t *dataPoints;
    data_t *SX, *SY, *SXX, *SYY, *SXY;

    int numBlock = ceil((ceil((n * 1.0) / BLOCK_SIZE)) / 2);

    uint64_t n_pad = numBlock * BLOCK_SIZE * 2;
    cudaError_t errorCode;

    errorCode = cudaMallocManaged(&dataPoints, n_pad * 2 * sizeof(data_t *));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    initData(n, dataPoints, n_pad);
    errorCode = cudaMallocManaged(&SX, numBlock * sizeof(data_t));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    errorCode = cudaMallocManaged(&SY, numBlock * sizeof(data_t));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    errorCode = cudaMallocManaged(&SXX, numBlock * sizeof(data_t));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    errorCode = cudaMallocManaged(&SYY, numBlock * sizeof(data_t));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    errorCode = cudaMallocManaged(&SXY, numBlock * sizeof(data_t));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    cout << "Num Block: " << numBlock << "\tn_pad: " << n_pad << "\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float timeElapsed = 0;
    // Start timer
    cudaEventRecord(start, 0);

    /* Kernel Call */
    LR<<<numBlock, BLOCK_SIZE>>>(dataPoints, SX, SY, SXX, SYY, SXY, n_pad, n);
    LR_Wrap<<<1, BLOCK_SIZE>>>(SX, SY, SXX, SYY, SXY, numBlock);
    cudaDeviceSynchronize();
    // Calculate slope and intercept
    auto slopeD = (n * SXY[0] - SX[0] * SY[0]) / (n * SXX[0] - SX[0] * SX[0]);
    auto interceptD = (SY[0] - slopeD * SX[0]) / n;
    // End timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeElapsed, start, stop);
    printf("Execution time = %f ms\n", timeElapsed);
    data_t SX_ll = 0, SY_ll = 0, SXX_ll = 0, SYY_ll = 0, SXY_ll = 0;
    for (int i = 0; i < n; i++)
    {
        SX_ll += dataPoints[i + n_pad * 0];
        SXX_ll += dataPoints[i + n_pad * 0] * dataPoints[i + n_pad * 0];
        SY_ll += dataPoints[i + n_pad * 1];
        SYY_ll += dataPoints[i + n_pad * 1] * dataPoints[i + n_pad * 1];
        SXY_ll += dataPoints[i + n_pad * 0] * dataPoints[i + n_pad * 1];
    }

    auto slopeH = (n * SXY_ll - SX_ll * SY_ll) / (n * SXX_ll - SX_ll * SX_ll);
    auto interceptH = (SY_ll - slopeH * SX_ll) / n;

    if (interceptH != interceptD) {
        cout << "Wrong Answer\n";
    }
    
    /* Free memory */
    cudaFree(SX);
    cudaFree(SY);
    cudaFree(SXX);
    cudaFree(SYY);
    cudaFree(SXY);
    cudaFree(dataPoints);

    return 0;
} /* main */