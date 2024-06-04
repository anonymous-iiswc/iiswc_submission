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
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <unordered_map>
#include <chrono>
#include <vector>
#include <string>
#include <queue>
#include <iomanip>
using namespace std;

#include <omp.h>
#include "../../util/common.h"
#include <cfloat>

vector<vector<data_t>> dataPoints;
vector<vector<data_t>> testPoints;

// Params ---------------------------------------------------------------------
typedef struct Params
{
    int numTestPoints;
    int numDataPoints;
    int dimension;
    int k;
    int numThreads;
    char *inputTestFile;
    char *inputDataFile;
    int target;
} Params;

void usage()
{
    fprintf(stderr,
            "\nUsage:  ./program [options]"
            "\n"
            "\n    -t    # of threads (default=8)"
            "\n    -n    number of data points (default=65536 points)"
            "\n    -m    number of test points (default=100 points)"
            "\n    -d    dimension (default=2)"
            "\n    -k    value of K (default=20)"
            "\n    -x    target index of the data set(default=12)"
            "\n    -i    input file containing training datapoints (default=generates datapoints with random numbers)"
            "\n    -j    input file containing testing datapoints (default=generates datapoints with random numbers)"
            "\n");
}

struct Params input_params(int argc, char **argv)
{
    struct Params p;
    p.numDataPoints = 65536;
    p.numTestPoints = 100;
    p.dimension = 2;
    p.k = 20;
    p.numThreads = 8;
    p.target = 12;
    p.inputTestFile = nullptr;
    p.inputDataFile = nullptr;

    int opt;
    while ((opt = getopt(argc, argv, "h:k:t:n:m:d:x:i:j:")) >= 0)
    {
        switch (opt)
        {
        case 'h':
            usage();
            exit(0);
            break;
        case 'n':
            p.numDataPoints = atoll(optarg);
            break;
        case 'm':
            p.numTestPoints = atoll(optarg);
            break;
        case 'd':
            p.dimension = atoll(optarg);
            break;
        case 'k':
            p.k = atoi(optarg);
            break;
        case 't':
            p.numThreads = atoi(optarg);
            break;
        case 'x':
            p.target = atoi(optarg);
            break;
        case 'i':
            p.inputDataFile = optarg;
            break;
        case 'j':
            p.inputTestFile = optarg;
            break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
        }
    }

    return p;
}

inline data_t calculateDistance(const vector<data_t> &pointA, const vector<data_t> &pointB, int dim, int target) {
    data_t sum = 0;
    for (int i = 0; i < dim; i++)
    {
        if (i == target) {continue;}
        sum += abs(pointA[i] - pointB[i]);
    }
    return sum;
}

struct DistancePoint {
    double distance;
    int index;
    
    DistancePoint(double d, int i) : distance(d), index(i) {}
};

struct CompareDistance {
    bool operator()(const DistancePoint& dp1, const DistancePoint& dp2) {
        return dp1.distance > dp2.distance;
    }
};

void runKNN(int numPoints, int numTests, int k, int dim, int numThreads, int target, vector<data_t> testPredictions)
{
    omp_set_num_threads(numThreads);
#pragma omp parallel
        {
            // Use thread-private variables
            vector<priority_queue<DistancePoint, vector<DistancePoint>, CompareDistance>> localMinHeaps(numTests);

#pragma omp for schedule(static)
            for (int i = 0; i < numTests; ++i)
            {
                for (int j = 0; j < numPoints; ++j) {
                    double dist = calculateDistance(dataPoints[i], dataPoints[j], dim, target);
                    if (int(localMinHeaps[i].size()) < k) {
                        localMinHeaps[i].emplace(dist, j);
                    } else if (dist < localMinHeaps[i].top().distance) {
                        localMinHeaps[i].pop();
                        localMinHeaps[i].emplace(dist, j);
                    }

                }
            }
#pragma omp critical
            {
                for (int i = 0; i < numTests; ++i) {
                    // Tally the labels of the k nearest neighbors
                    unordered_map<int, int> labelCount;
                    while (!localMinHeaps[i].empty()) {
                        int index = localMinHeaps[i].top().index;
                        int label = dataPoints[index][target];
                        labelCount[label]++;
                        localMinHeaps[i].pop();
                    }
                    
                    // Find the label with the highest count
                    int maxCount = 0;
                    int bestLabel = -1;
                    for (const auto& entry : labelCount) {
                        if (entry.second > maxCount) {
                            maxCount = entry.second;
                            bestLabel = entry.first;
                        }
                    }

                    // Assign the most frequent label to the test point
                    testPredictions[i] = bestLabel;
                }
            }
        }

    
}

vector<vector<int>> readCSV(const string& filename) {
    vector<vector<int>> data;
    ifstream file(filename);
    string line;
    
    if (!file.is_open()) {
        throw runtime_error("Could not open file");
    }


    while (getline(file, line)) {
        vector<int> row;
        stringstream ss(line);
        string value;
        
        while (getline(ss, value, ',')) {
            try {
                int intValue = stoi(value);
                row.push_back(intValue);
            } catch (const invalid_argument& e) {
                cerr << "Invalid argument: " << e.what() << " for value " << value << '\n';
            } catch (const out_of_range& e) {
                cerr << "Out of range: " << e.what() << " for value " << value << '\n';
            }
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

void printData(data_t **dataArray, int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            cout << dataArray[i][j] << "\t";
        }
        cout << "\n";
    }
}

// void getMatrix(int row, int column, int padding, vector<vector<int>> &inputMatrix)
// {
//   srand((unsigned)time(NULL));
//   inputMatrix.resize(row + 2 * padding, vector<int>(column + 2 * padding, 0));
// #pragma omp parallel for
//   for (int i = padding; i < row + padding; ++i)
//   {
//     for (int j = padding; j < column + padding; ++j)
//     {
//       inputMatrix[i][j] = (rand() % ((i * j) + 1)) + 1;
//     }
//   }
// }

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{
    struct Params params = input_params(argc, argv);


    if (params.inputTestFile == nullptr)
    {
        //getMatrix(params.numTestPoints, params.dimension, 0, testPoints);
        initMatrix(params.numTestPoints, params.dimension, testPoints);
    }
    else
    {
        vector<vector<int>> test_data_int = readCSV(params.inputTestFile);
        params.dimension = test_data_int[0].size();
        params.numTestPoints = test_data_int.size();

        testPoints = vector<vector<data_t>>(test_data_int.begin(), test_data_int.end());;
    }
    if (params.inputDataFile == nullptr)
    {
        //getMatrix(params.numDataPoints, params.dimension, 0, dataPoints);
        initMatrix(params.numDataPoints, params.dimension, dataPoints);
    }
    else
    {
        vector<vector<int>> train_data_int = readCSV(params.inputDataFile);

        params.dimension = train_data_int[0].size();
        params.numDataPoints = train_data_int.size();

        dataPoints = vector<vector<data_t>>(train_data_int.begin(), train_data_int.end());
    }
    int k = params.k, numPoints = params.numDataPoints, numTests = params.numTestPoints, dim = params.dimension;
    int target = params.target;
    vector<int> testPredictions(numTests);

    cout << "Set up done!\n";
    
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < WARMUP; i++)
    {
        runKNN(numPoints, numTests, k, dim, params.numThreads, target, testPredictions);
    }
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> elapsedTime = (end - start)/WARMUP;
    cout << "Duration: " << fixed << setprecision(3) << elapsedTime.count() << " ms." << endl;
    return 0;
}
