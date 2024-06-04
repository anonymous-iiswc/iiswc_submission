/**
 * @file app.c
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

using namespace std;

vector<data_t> A;
vector<data_t> B;
vector<data_t> C;

/**
 * @brief cpu vector addition kernel
 */
static void vectorXOR(u64 nr_elements, int t)
{
    omp_set_num_threads(t);
#pragma omp parallel for
    for (u64 i = 0; i < nr_elements; i++)
    {
        C[i] = A[i] ^ B[i];
    }
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
            "\nUsage:  ./program [options]"
            "\n"
            "\n    -t <T>    # of threads (default=8)"
            "\n    -i <I>    input size (default=8M elements)"
            "\n");
}

struct Params input_params(int argc, char **argv)
{
    struct Params p;
    p.vector_size = 65536;
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

    initVector(vector_size, A);
    initVector(vector_size, B);
    C.resize(vector_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (i32 i = 0; i < WARMUP; i++)
    {
        vectorXOR(vector_size, p.n_threads);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
    cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

    return 0;
}
