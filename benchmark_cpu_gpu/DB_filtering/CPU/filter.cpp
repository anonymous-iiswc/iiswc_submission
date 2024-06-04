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

#define MY_RANGE 100

using namespace std;

/**
 * @brief cpu database filtering kernel
 */

void filterByKey(std::vector<int> &Vector, u64 vector_size, int key, std::vector<bool> & bitMap)
{
#pragma omp parallel for
  for (u64 i = 0; i < vector_size; ++i)
  {
    if (key > Vector[i])
      bitMap[i] = true;
  }
}

typedef struct Params
{
    u64 inVectorSize;
    int key;
    bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./filter [options]"
          "\n"
          "\n    -n    database size (default=65536 elements)"
          "\n    -k    value of key (default = 70)"
          "\n    -v    t = print output vector. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.inVectorSize = 65536;
  p.key = 70;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:n:k:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'n':
      p.inVectorSize = strtoull(optarg, NULL, 0);
      break;
    case 'k':
      p.key = strtoull(optarg, NULL, 0);
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

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv){

    struct Params p = getInputParams(argc, argv);

    u64 inVectorSize = p.inVectorSize;

    vector<data_t> inVector(inVectorSize);
    vector<data_t> outVector;

    std::cout << "DB element size: " << inVectorSize << std::endl;

    srand(8746219);
#pragma omp parallel for
    for (u64 i = 0; i < inVectorSize; i++){
        inVector[i] = rand() % MY_RANGE;
    }
    std::vector<bool> bitMap(inVectorSize, false);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // run scan
    filterByKey(inVector, inVectorSize, p.key, bitMap);
    
    // select data whose bitmap is '1'
// #pragma omp parallel for
    for (u64 i = 0; i < inVectorSize; i++){
        if(bitMap[i] == true){
            outVector.push_back(inVector[i]);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;

    cout << outVector.size() <<" selected" << endl;
    if(p.shouldVerify == true){
      for (u64 i = 0; i < inVectorSize; i++){
        cout << inVector[i] << " ";
      }
      cout << endl;
      for (u64 i = 0; i < inVectorSize; i++){
        cout << bitMap[i] << " ";
      }
      cout << endl;
      for (u64 i = 0; i < outVector.size(); i++){
        cout << outVector[i] << " ";
      }
      cout << endl;
    }
    cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

    return 0;
}
