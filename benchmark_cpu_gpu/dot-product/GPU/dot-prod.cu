/* File:     matrix vector multiplication cuda
 * Purpose:  Implement  on a gpu using cuda
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cublas_v2.h>

#include "../../util/common.h"

#define TOLERANCE 200.0f

using namespace std;

vector<data_t> A;
vector<data_t> B;

// Params ---------------------------------------------------------------------
typedef struct Params
{
    uint64_t vector_length;
    bool shouldVerify;
} Params;

void usage()
{
    fprintf(stderr,
            "\nUsage:  ./dp [options]"
            "\n"
            "\n    -l   vector size"
            "\n    -h   print usage"
            "\n    -v   t = verifies PIM output with host output. (default=false)"
            "\n");
}

struct Params input_params(int argc, char **argv)
{
    struct Params p;
    p.vector_length = 65536;
    p.shouldVerify = false;

    int opt;
    while ((opt = getopt(argc, argv, ":h:l:v:")) >= 0)
    {
        switch (opt)
        {
        case 'h':
            usage();
            exit(0);
            break;
        case 'l':
            p.vector_length = atoll(optarg);
            break;
        case 'v':
            p.shouldVerify = (*optarg == 't') ? true : false;
            break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
        }
    }

    return p;
}

int main(int argc, char *argv[])
{
    struct Params p = input_params(argc, argv);

    u64 vector_length = p.vector_length;
    initVector(vector_length, A);
    initVector(vector_length, B);

    float *x, *y;

    cudaError_t errorCode;

    errorCode = cudaMalloc((void **)&x, vector_length * sizeof(data_t));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    errorCode = cudaMalloc((void **)&y, vector_length * sizeof(data_t));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    errorCode = cudaMemcpy(x, A.data(), vector_length * sizeof(float), cudaMemcpyHostToDevice);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    errorCode = cudaMemcpy(y, B.data(), vector_length * sizeof(float), cudaMemcpyHostToDevice);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "CUBLAS initialization failed\n";
        exit(1);
    }

    // Event creation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float timeElapsed = 0;
    float result_device;

    // Start timer
    cudaEventRecord(start, 0);
    /* Kernel Call */
    status = cublasSdot(handle, vector_length, x, 1, y, 1, &result_device);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "CUBLAS SGEMV failed\n";
        exit(1);
    }

    // Check for kernel launch errors
    errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    // End timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeElapsed, start, stop);

    printf("Execution time = %f ms\n", timeElapsed);

    if (p.shouldVerify)
    {
        cout.precision(0);
        data_t result_host = 0;
        for (int i = 0; i < vector_length; ++i)
        {
            result_host += A[i] * B[i];
        }
        if (abs(result_device - result_host) > TOLERANCE)
        {
            cout << fixed << "Do product failed. Expected: " << result_host << "\tReceived: " << result_device << endl;
        }
    }

    /* Free memory */
    cublasDestroy(handle);
    cudaFree(x);
    cudaFree(y);

    return 0;
} /* main */
