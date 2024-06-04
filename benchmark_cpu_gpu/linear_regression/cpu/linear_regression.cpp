/* File:     vec_add.cu
 * Purpose:  Implement vector addition on a gpu using cuda
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>

#include <omp.h>
#include "../../util/common.h"
#include <iomanip>

using namespace std;

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
    vector<vector<data_t>> dataPoints;
    initMatrix(n, 2, dataPoints);
    cout << "Done initializing data\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (i32 w = 0; w < WARMUP; w++) 
    {
        data_t SX = 0, SY = 0, SXX = 0, SYY = 0, SXY = 0;
#pragma omp parallel for reduction(+ : SX, SXX, SY, SYY, SXY)
        for (u64 i = 0; i < n; i++)
        {
            SX += dataPoints[i][0];
            SXX += dataPoints[i][0] * dataPoints[i][0];
            SY += dataPoints[i][1];
            SYY += dataPoints[i][1] * dataPoints[i][1];
            SXY += dataPoints[i][0] * dataPoints[i][1];
        }
        // Calculate slope and intercept
        auto slope = (n * SXY - SX * SY) / (n * SXX - SX * SX);
        auto intercept = (SY - slope * SX) / n;
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
    cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;
    return 0;
} /* main */