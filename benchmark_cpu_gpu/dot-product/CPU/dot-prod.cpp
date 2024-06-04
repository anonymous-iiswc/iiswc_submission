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
#include <bits/stdc++.h>

#include <omp.h>
#include "../../util/common.h"

vector<data_t> A;
vector<data_t> B;

/**
 * @brief compute output in the host
 */
static void dot_product(uint64_t vector_length)
{
    data_t result = 0.0;
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < A.size(); ++i)
    {
        result += A[i] * B[i];
    }
}

// Params ---------------------------------------------------------------------
typedef struct Params
{
    uint64_t vectorLength;
} Params;

void usage()
{
    fprintf(stderr,
            "\nUsage:  ./dp [options]"
            "\n"
            "\n    -l    vector length"
            "\n    -h    print usage"
            "\n");
}

struct Params input_params(int argc, char **argv)
{
    struct Params p;
    p.vectorLength = 65536;

    int opt;
    while ((opt = getopt(argc, argv, ":l:h:")) >= 0)
    {
        switch (opt)
        {
        case 'h':
            usage();
            exit(0);
            break;
        case 'l':
            p.vectorLength = atoll(optarg);
            break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
        }
    }

    return p;
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{

    struct Params p = input_params(argc, argv);

    uint64_t vectorLength = p.vectorLength;

    initVector(vectorLength, A);
    initVector(vectorLength, B);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < WARMUP; i++)
    {
        dot_product(vectorLength);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsedTime = (end - start) / WARMUP;
    cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

    return 0;
}
