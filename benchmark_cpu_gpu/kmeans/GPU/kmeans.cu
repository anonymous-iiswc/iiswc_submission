/* File:     kmeans.cu
 * Purpose:  Implement vector addition on a gpu using cuda
 *
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <chrono>
#include <vector>
#include <iomanip>
#include <float.h>

#include "../../util/common.h"
using namespace std;

struct Data
{
  Data(int size) : size(size), bytes(size * sizeof(data_t))
  {
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMemset(x, 0, bytes);
    cudaMemset(y, 0, bytes);
  }

  Data(int size, std::vector<data_t> &h_x, std::vector<data_t> &h_y)
      : size(size), bytes(size * sizeof(data_t))
  {
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMemcpy(x, h_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, h_y.data(), bytes, cudaMemcpyHostToDevice);
  }

  ~Data()
  {
    cudaFree(x);
    cudaFree(y);
  }

  data_t *x{nullptr};
  data_t *y{nullptr};
  int size{0};
  int bytes{0};
};

__device__ data_t
squared_l2_distance(data_t x_1, data_t y_1, data_t x_2, data_t y_2)
{
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

__global__ void fine_reduce(const data_t *__restrict__ data_x,
                            const data_t *__restrict__ data_y,
                            int data_size,
                            const data_t *__restrict__ means_x,
                            const data_t *__restrict__ means_y,
                            data_t *__restrict__ new_sums_x,
                            data_t *__restrict__ new_sums_y,
                            int k,
                            int *__restrict__ counts)
{
  extern __shared__ data_t shared_data[];

  const int local_index = threadIdx.x;
  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index >= data_size)
    return;

  // Load the mean values into shared memory.
  if (local_index < k)
  {
    shared_data[local_index] = means_x[local_index];
    shared_data[k + local_index] = means_y[local_index];
  }

  __syncthreads();

  // Load once here.
  const data_t x_value = data_x[global_index];
  const data_t y_value = data_y[global_index];

  data_t best_distance = FLT_MAX;
  int best_cluster = -1;
  for (int cluster = 0; cluster < k; ++cluster)
  {
    const data_t distance = squared_l2_distance(x_value,
                                                y_value,
                                                shared_data[cluster],
                                                shared_data[k + cluster]);
    if (distance < best_distance)
    {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

  __syncthreads();

  // reduction

  const int x = local_index;
  const int y = local_index + blockDim.x;
  const int count = local_index + blockDim.x + blockDim.x;

  for (int cluster = 0; cluster < k; ++cluster)
  {
    shared_data[x] = (best_cluster == cluster) ? x_value : 0;
    shared_data[y] = (best_cluster == cluster) ? y_value : 0;
    shared_data[count] = (best_cluster == cluster) ? 1 : 0;
    __syncthreads();

    // Reduction for this cluster.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
      if (local_index < stride)
      {
        shared_data[x] += shared_data[x + stride];
        shared_data[y] += shared_data[y + stride];
        shared_data[count] += shared_data[count + stride];
      }
      __syncthreads();
    }

    // Now shared_data[0] holds the sum for x.

    if (local_index == 0)
    {
      const int cluster_index = blockIdx.x * k + cluster;
      new_sums_x[cluster_index] = shared_data[x];
      new_sums_y[cluster_index] = shared_data[y];
      counts[cluster_index] = shared_data[count];
    }
    __syncthreads();
  }
}

__global__ void coarse_reduce(data_t *__restrict__ means_x,
                              data_t *__restrict__ means_y,
                              data_t *__restrict__ new_sum_x,
                              data_t *__restrict__ new_sum_y,
                              int k,
                              int *__restrict__ counts)
{
  extern __shared__ data_t shared_data[];

  const int index = threadIdx.x;
  const int y_offset = blockDim.x;

  shared_data[index] = new_sum_x[index];
  shared_data[y_offset + index] = new_sum_y[index];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= k; stride /= 2)
  {
    if (index < stride)
    {
      shared_data[index] += shared_data[index + stride];
      shared_data[y_offset + index] += shared_data[y_offset + index + stride];
    }
    __syncthreads();
  }

  if (index < k)
  {
    const int count = max(1, counts[index]);
    means_x[index] = new_sum_x[index] / count;
    means_y[index] = new_sum_y[index] / count;
    new_sum_y[index] = 0;
    new_sum_x[index] = 0;
    counts[index] = 0;
  }
}

typedef struct Params
{
  uint64_t numPoints;
  int maxItr;
  int dimension;
  int k;
  int numThreads;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./km [options]"
          "\n"
          "\n    -t    # of threads (default=8)"
          "\n    -p    number of points (default=10M points)"
          "\n    -k    value of K (default=20)"
          "\n    -d    number of features (default=2 dimensions)"
          "\n    -i    max iteration (default=5 iteration)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.numPoints = 10000000;
  p.k = 20;
  p.dimension = 2;
  p.numThreads = 6;
  p.maxItr = 2;

  int opt;
  while ((opt = getopt(argc, argv, "p:k:d:i:t:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'p':
      p.numPoints = atoll(optarg);
      break;
    case 'k':
      p.k = atoi(optarg);
      break;
    case 'd':
      p.dimension = atoi(optarg);
      break;
    case 't':
      p.numThreads = atoi(optarg);
      break;
    case 'i':
      p.maxItr = atoi(optarg);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }

  return p;
}

int main(int argc, char **argv)
{

  struct Params p = input_params(argc, argv);

  int k = p.k, number_of_elements = p.numPoints, dim = p.dimension, number_of_iterations = p.maxItr;

  std::vector<data_t> h_x;
  std::vector<data_t> h_y;
  initVector(number_of_elements, h_x);
  initVector(number_of_elements, h_x);

  Data d_data(number_of_elements, h_x, h_y);

  std::mt19937 rng(std::random_device{}());
  std::shuffle(h_x.begin(), h_x.end(), rng);
  std::shuffle(h_y.begin(), h_y.end(), rng);
  Data d_means(k, h_x, h_y);

  const int threads = 1024;
  const int blocks = (number_of_elements + threads - 1) / threads;

  std::cerr << "Processing " << number_of_elements << " points on " << blocks
            << " blocks x " << threads << " threads" << std::endl;

  // * 3 for x, y and counts.
  const int fine_shared_memory = 3 * threads * sizeof(data_t);
  // * 2 for x and y. Will have k * blocks threads for the coarse reduction.
  const int coarse_shared_memory = 2 * k * blocks * sizeof(data_t);

  Data d_sums(k * blocks);
  int *d_counts;
  cudaMalloc(&d_counts, k * blocks * sizeof(int));
  cudaMemset(d_counts, 0, k * blocks * sizeof(int));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timeElapsed = 0;
  // Start timer
  cudaEventRecord(start, 0);
  for (size_t iteration = 0; iteration < number_of_iterations; ++iteration)
  {
    fine_reduce<<<blocks, threads, fine_shared_memory>>>(d_data.x,
                                                         d_data.y,
                                                         d_data.size,
                                                         d_means.x,
                                                         d_means.y,
                                                         d_sums.x,
                                                         d_sums.y,
                                                         k,
                                                         d_counts);
    cudaDeviceSynchronize();

    coarse_reduce<<<1, k * blocks, coarse_shared_memory>>>(d_means.x,
                                                           d_means.y,
                                                           d_sums.x,
                                                           d_sums.y,
                                                           k,
                                                           d_counts);

    cudaDeviceSynchronize();
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeElapsed, start, stop);
  printf("Execution time = %f ms\n", timeElapsed);

  cudaFree(d_counts);

  std::vector<data_t> mean_x(k, 0);
  std::vector<data_t> mean_y(k, 0);
  cudaMemcpy(mean_x.data(), d_means.x, d_means.bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(mean_y.data(), d_means.y, d_means.bytes, cudaMemcpyDeviceToHost);

  for (size_t cluster = 0; cluster < k; ++cluster)
  {
    std::cout << mean_x[cluster] << " " << mean_y[cluster] << std::endl;
  }
}