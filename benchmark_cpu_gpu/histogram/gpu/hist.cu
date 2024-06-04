/* File:     hist.cu
 * Purpose:  Implement histogram on a gpu using cuda
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <cub/cub.cuh>

#define CUB_STDERR
#include <cub/cub.cuh>

#include "../../util/common.h"

#define IMG_DATA_OFFSET_POS 10
#define BITS_PER_PIXEL_POS 28
 
using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t dataSize;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./lr [options]"
          "\n"
          "\n    -l    input size (default=65536 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing 2D matrix (default=generates matrix with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dataSize = NULL;
  p.configFile = nullptr;
  p.inputFile = "sample1.bmp";
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:l:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'l':
      p.dataSize = strtoull(optarg, NULL, 0);
      break;
    case 'c':
      p.configFile = optarg;
      break;
    case 'i':
      p.inputFile = optarg;
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
    struct Params p = getInputParams(argc, argv);

    // Begin data parsing
    int fd, imgdata_bytes;
    struct stat finfo;
    char *fdata;
    unsigned short *data_pos;
    fd = open(p.inputFile, O_RDONLY);
    if (fd < 0)
    {
        perror("Failed to open input file, or file doesn't exist");
        return 1;
    }
    if (fstat(fd, &finfo) < 0) 
    {
        perror("Failed to get file info");
        return 1;
    }
    fdata = static_cast<char *>(mmap(0, finfo.st_size + 1, PROT_READ, MAP_PRIVATE, fd, 0));
    if (fdata == 0) 
    {
        perror("Failed to memory map the file");
        return 1;
    }
    data_pos = (unsigned short *)(&(fdata[IMG_DATA_OFFSET_POS]));
    imgdata_bytes = (int)finfo.st_size - (int)(*(data_pos));
    printf("This file has %d bytes of image data, %d pixels\n", imgdata_bytes,
                                                            imgdata_bytes / 3);

    std::vector<uint8_t> imgData(fdata + *data_pos, fdata + finfo.st_size);
    std::vector<int> imgDataToInt;

    unsigned char* h_samples = new unsigned char[(imgData.size()/3) * 4];

    for (size_t i = 0, j = 0; i < imgData.size(); i+=3, j+=4)
    {
        h_samples[j] = static_cast<int> (imgData[i]);
        h_samples[j + 1] = static_cast<int> (imgData[i + 1]);
        h_samples[j + 2] = static_cast<int> (imgData[i + 2]);
        h_samples[j + 3] = 0; 
    }

    for (int i = 0; i < imgdata_bytes; ++i) {

        imgDataToInt.push_back(static_cast<int> (imgData[i]));
    }
    // End data parsing

    unsigned char* d_samples;
    cudaMalloc(&d_samples, (imgData.size() / 3) * 4 * sizeof(unsigned char));
    cudaMemcpy(d_samples, h_samples, (imgData.size() / 3) * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int num_samples = imgDataToInt.size() / 3;
    int *d_histogram[3];
    int num_levels[3] = {257, 257, 257};
    int lower_level[3] = {0, 0, 0};
    int upper_level[3] = {256, 256, 256};


    for (int i = 0; i < 3; ++i) {
        cudaMalloc(&d_histogram[i], num_levels[i] * sizeof(int));
        cudaMemset(d_histogram[i], 0, num_levels[i] * sizeof(int));
    }

    cudaError_t errorCode;

    int h_histogram[3][256] = {0};

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    errorCode = cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes,
                        d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    errorCode = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float timeElapsed = 0;

    // Start timer
    cudaEventRecord(start, 0);

    errorCode = cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes,
    d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples);
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

    for (int i = 0; i < 3; ++i) {
        errorCode = cudaMemcpy (h_histogram[i], d_histogram[i], 256 * sizeof(int), cudaMemcpyDeviceToHost);
        if (errorCode != cudaSuccess)
        {
            cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
            std::cout << "TESt" << std::endl;
            exit(1);
        }
    }

    // Uncomment for verification
    // int i;
    // printf("\n\nBlue\n");
    // printf("----------\n\n");
    // for (i = 0; i < 256; i++) {
    //     printf("%d - %d\n", i, h_histogram[0][i]);        
    // }

    // printf("\n\nGreen\n");
    // printf("----------\n\n");
    // for (i = 0; i < 256; i++) {
    //     printf("%d - %d\n", i, h_histogram[1][i]);        
    // }
   

    // printf("\n\nRed\n");
    // printf("----------\n\n");
    // for (i = 0; i < 256; i++) {
    //     printf("%d - %d\n", i, h_histogram[2][i]);        
    // }

    if(p.shouldVerify)
    {
        int red_cpu[256];
        int green_cpu[256];
        int blue_cpu[256];

        memset(&(red_cpu[0]), 0, sizeof(int) * 256);
        memset(&(green_cpu[0]), 0, sizeof(int) * 256);
        memset(&(blue_cpu[0]), 0, sizeof(int) * 256);
   
        for (int i=*data_pos; i < finfo.st_size; i+=3) {      
            unsigned char *val = (unsigned char *)&(fdata[i]);
            blue_cpu[*val]++;
      
            val = (unsigned char *)&(fdata[i+1]);
            green_cpu[*val]++;
      
            val = (unsigned char *)&(fdata[i+2]);
            red_cpu[*val]++;   
        }

        int errorFlag = 0;
        for (int i = 0; i < 256; ++i)
        {
            if (red_cpu[i] != h_histogram[2][i]) {
                std::cout << "Wrong answer for red: " << h_histogram[2][i] << " | " << i << " (expected " << red_cpu[i] << ")" << std::endl;
                errorFlag = 1;
            }
            if (green_cpu[i] != h_histogram[1][i]) {
                std::cout << "Wrong answer for green: " << h_histogram[1][i] << " | " << i << " (expected " << green_cpu[i] << ")" << std::endl;
                errorFlag = 1;
            }
            if (blue_cpu[i] != h_histogram[0][i]) {
                std::cout << "Wrong answer for blue: " << h_histogram[0][i] << " | " << i << " (expected " << blue_cpu[i] << ")" << std::endl;
                errorFlag = 1;
            }
        }

        if (errorFlag == 1)
            std::cout << "At least one wrong answer" << std::endl;
        else
            std::cout << "Correct!" << std::endl;
    }

    cudaFree(d_samples); 
    for (int i = 0; i < 3; ++i) {
        cudaFree(d_histogram[i]);
    }
    cudaFree(d_temp_storage);

    printf("\n");
    return 0;

} /* main */