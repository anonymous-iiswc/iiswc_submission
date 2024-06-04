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
#include <algorithm>
#include <climits>
#include <omp.h>

#include "../../util/common.h"
#include <iomanip>

using namespace std;

// Function to perform counting sort on the array based on the digit represented by exp
void countingSort(std::vector<data_t> &dataArray, int exp, std::vector<data_t> &output, std::vector<data_t> &count)
{
    u64 n = dataArray.size();
    int numThreads = omp_get_max_threads();
    std::vector<std::vector<int>> localCount(numThreads, std::vector<int>(10, 0));

// Store count of occurrences in localCount[]
#pragma omp parallel
    {
        int threadNum = omp_get_thread_num();
#pragma omp for nowait
        for (u64 i = 0; i < n; i++)
        {
            localCount[threadNum][(dataArray[i] / exp) % 10]++;
        }
    }

    // Aggregate counts from all threads
    for (int i = 0; i < 10; i++)
    {
        for (int t = 0; t < numThreads; t++)
        {
            count[i] += localCount[t][i];
        }
    }

    // Change count[i] so that count[i] now contains the actual
    // position of this digit in output[]
    for (int i = 1; i < 10; i++)
    {
        count[i] += count[i - 1];
    }

    for (u64 i = 0; i < n; i++)
    {
        int digit = (dataArray[n - i - 1] / exp) % 10;
        int idx = count[digit]--;
        output[idx - 1] = dataArray[n - i - 1];
    }

    // Copy the output array to dataArray[], so that dataArray[] now
    // contains sorted numbers according to the current digit
    std::copy(output.begin(), output.end(), dataArray.begin());
}

void radixSort(const vector<data_t> &dataArray, vector<data_t> &sortedArray)
{

    // Find the maximum number to know the number of digits
    data_t m = *std::max_element(dataArray.begin(), dataArray.end());

    // Output array to store sorted numbers temporarily
    std::vector<data_t> tempArray(dataArray.size());
    std::copy(dataArray.begin(), dataArray.end(), sortedArray.begin());
    // Count array to store the count of occurrences of digits
    std::vector<int> count(10, 0);

    // Do counting sort for every digit. Note that instead
    // of passing the digit number, exp is passed. exp is 10^i
    // where i is the current digit number
    for (int exp = 1; m / exp > 0; exp *= 10)
    {
        countingSort(sortedArray, exp, tempArray, count);
        std::fill(count.begin(), count.end(), 0);
    }
}

void printArray(const std::vector<data_t> &dataArray)
{
    for (auto num : dataArray)
    {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << argc << "\n";
        printf("Array size required.\n");
        printf("Syntax: %s <array_size>.\n", argv[0]);
        exit(1);
    }

    u64 n = atoll(argv[1]);
    vector<data_t> dataArray;
    initVector(n, dataArray);
    vector<data_t> sortedArray(n);
    cout << "Done initializing data\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (i32 w = 0; w < WARMUP; w++)
    {
        radixSort(dataArray, sortedArray);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
    cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;
    // cout << "Original Array:\n";
    // printArray(dataArray);
    // cout << "\n\nSorted array:\n";
    // printArray(sortedArray);
    return 0;
} /* main */