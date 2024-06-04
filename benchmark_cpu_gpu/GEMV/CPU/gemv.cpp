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

static data_t **A;
static data_t *B;
static data_t *C;

/**
 * @brief creates a "test file" by filling a buffer of 64MB with pseudo-random values
 * @param nr_elements how many 32-bit elements we want the file to be
 * @return the buffer address
 */
void create_data(uint64_t row, uint64_t col)
{
    A = (data_t **)malloc(row * sizeof(data_t *));
    B = (data_t *)malloc(col * sizeof(data_t));
    C = (data_t *)malloc(row * sizeof(data_t));

    for (int i = 0; i < col; i++)
    {
        B[i] = 1.0;
    }

    for (int i = 0; i < row; i++)
    {
	C[i] = 0.0;
        A[i] = (data_t *)malloc(col * sizeof(data_t));
        for (int j = 0; j < col; j++)
        {
            A[i][j] = 2.0;
        }
    }
}

/**
 * @brief compute output in the host
 */
static void matrix_vector_multiplication(uint64_t row, uint64_t col)
{
#pragma omp parallel for
    for (size_t i = 0; i < row; i++)
    {
        for (size_t j = 0; j < col; j++)
        {
            C[i] += A[i][j] * B[j];
        }
    }
}

// Params ---------------------------------------------------------------------
typedef struct Params
{
    uint64_t row, column;
    int threads;
} Params;

void usage()
{
    fprintf(stderr,
            "\nUsage:  ./program [options]"
            "\n"
            "\n    -t <T>    # of threads (default=8)"
            "\n    -r <R>    row size"
            "\n    -c <C>    column size"
            "\n");
}

struct Params input_params(int argc, char **argv)
{
    struct Params p;
    p.row = 65536;
    p.column = 65536;
    p.threads = 8;

    int opt;
    while ((opt = getopt(argc, argv, ":r:c:t:")) >= 0)
    {
        switch (opt)
        {
        case 'h':
            usage();
            exit(0);
            break;
        case 'r':
            p.row = atoll(optarg);
            break;
        case 'c':
            p.column = atoll(optarg);
            break;
        case 't':
            p.threads = atoi(optarg);
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

    uint64_t r = p.row, c = p.column;

    create_data(r, c);

    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < WARMUP; i++)
    {
        matrix_vector_multiplication(r, c);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
    cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

    free(A);
    free(B);
    free(C);

    return 0;
}
