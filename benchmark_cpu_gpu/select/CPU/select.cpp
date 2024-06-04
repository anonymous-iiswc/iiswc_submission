/**
 * @file axpy.cpp
 * @brief Template for a Host Application Source File.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>
#include <iomanip>
#include <chrono>

#include <omp.h>
#include "../../util/common.h"

/**
 * @brief cpu vector addition kernel
 */
u64 select(const std::vector<data_t> &x, std::vector<data_t> &y)
{
    u64 pos = 0;

#pragma omp parallel for shared(pos)
    for (size_t idx = 0; idx < x.size(); idx++)
    {
        if ((x[idx] % 2))
        {
            u64 local_pos;
#pragma omp atomic capture
            local_pos = pos++;

            y[local_pos] = x[idx];
        }
    }
    return pos;
}

// Params ---------------------------------------------------------------------
typedef struct Params
{
    u64 vector_size;
    int n_threads;
} Params;

void usage()
{
    fprintf(stderr,
            "\nUsage:  ./select [options]"
            "\n"
            "\n    -t <T>    # of threads (default=8)"
            "\n    -i <I>    input size (default=8M elements)"
            "\n");
}

struct Params input_params(int argc, char **argv)
{
    struct Params p;
    p.vector_size = 16 << 20;
    p.n_threads = 8;

    int opt;
    while ((opt = getopt(argc, argv, "i:t:")) >= 0)
    {
        switch (opt)
        {
        case 'h':
            usage();
            exit(0);
            break;
        case 'i':
            p.vector_size = atoi(optarg);
            break;
        case 't':
            p.n_threads = atoi(optarg);
            break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
        }
    }
    assert(p.n_threads > 0 && "Invalid # of ranks!");

    return p;
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{

    struct Params p = input_params(argc, argv);

    u64 vector_size = p.vector_size;

    vector<data_t> X;
    vector<data_t> Y(vector_size);

    initVector(vector_size, X);
    u64 totalCount = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (i32 i = 0; i < WARMUP; i++)
    {
        totalCount = select(X, Y);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
    cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms.\n" << "Total Element Slected: " << totalCount << endl;

    return 0;
}
