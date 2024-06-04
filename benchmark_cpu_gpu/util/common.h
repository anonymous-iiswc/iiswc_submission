#include <iostream>
#include <vector>
using namespace std;


#ifndef _COMMON_H_
#define _COMMON_H_

#define WARMUP 1

#define MAX_NUMBER 1024

typedef 	uint64_t	u64;
typedef		int64_t		i64;
typedef		uint32_t	u32;
typedef		int32_t		i32;
typedef		uint16_t	u16;
typedef		int16_t		i16;
typedef		uint8_t		u8;
typedef		int8_t		i8;

// Params ---------------------------------------------------------------------                                                                                             
typedef struct PIMParams {
    string device;
    bool shouldVerify;
    string memoryConfig;
}PIMParams;

#ifndef DATA_TYPE
typedef int32_t data_t;
#else
typedef DATA_TYPE data_t;
#endif

/**
* @brief creates a vector with random values
* @param vectorSize size of the vector
* @param vectorPoints 
*/
void initVector(u64 vectorSize, vector<data_t>& vectorPoints)
{
    vectorPoints.resize(vectorSize);
    // Using a fixed seed instead of time for reproducibility
    //srand((unsigned)time(NULL));
    srand(8746219);
    #pragma omp parallel for
    for (u64 i = 0; i < vectorSize; i++)
    {
        vectorPoints[i] = rand() % MAX_NUMBER;
    }
}

/**
* @brief creates a vector with random values
* @param row number of rows in the matrix
* @param col number of columns in the matrix
*/
void initMatrix(u64 row,  u64 col, vector<vector<data_t>>& mat)
{
    // Providing a seed value
    srand((unsigned)time(NULL));
    mat.resize(row, vector<data_t>(col));
    #pragma omp parallel for
    for (u64 i = 0; i < row; i++)
    {
        for (u64 j = 0; j < col; j++) {
            mat[i][j] = rand() % MAX_NUMBER;
        }
    }
}

#endif
