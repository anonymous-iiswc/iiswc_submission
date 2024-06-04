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
#include <stdint.h>
#include <bits/stdc++.h>

#include <chrono>

#include <omp.h>
#include "../../util/common.h"
#include <iomanip>

vector<vector<data_t>> A;
vector<vector<data_t>> B;
vector<vector<data_t>> C;

#define BLOCK_SIZE 64
/**
 * @brief compute output in the host
 */
static void matrix_matrix_multiplication(u64 row_A, u64 col_A,  u64 col_B)
{
    u64 i, j, k, ii, jj, kk;
    #pragma omp parallel for collapse(2) private(i, j, k, ii, jj, kk) shared(A, B, C)
    for (ii = 0; ii < row_A; ii += BLOCK_SIZE) {
        for (jj = 0; jj < col_B; jj += BLOCK_SIZE) {
            for (kk = 0; kk < col_A; kk += BLOCK_SIZE) {
                for (i = ii; i < ii + BLOCK_SIZE && i < row_A; ++i) {
                    for (j = jj; j < jj + BLOCK_SIZE && j < col_B; ++j) {
                        for (k = kk; k < kk + BLOCK_SIZE && k < col_A; ++k) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

// Params ---------------------------------------------------------------------
typedef struct Params
{
    uint64_t row, columnA, columnB;
} Params;

void usage()
{
    fprintf(stderr,
            "\nUsage:  ./program [options]"
            "\n"
            "\n    -r <R>    row size of Matrix A"
            "\n    -c <C>    column size of Matrix A"
            "\n    -d <D>    column size of Matrix B"
            "\n");
}

struct Params input_params(int argc, char **argv)
{
    struct Params p;
    p.row = 6553;
    p.columnA = 6553;
    p.columnB = 6553;

    int opt;
    while ((opt = getopt(argc, argv, ":r:c:d:")) >= 0)
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
            p.columnA = atoll(optarg);
            break;
        case 'd':
            p.columnB = atoll(optarg);
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

    initMatrix(p.row, p.columnA, A);
    initMatrix(p.columnA, p.columnB, B);
    C.resize(p.row, vector<data_t>(p.columnB, 0));
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < WARMUP; i++)
    {
        matrix_matrix_multiplication(p.row, p.columnA, p.columnB);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
    cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

    return 0;
}
