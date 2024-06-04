#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <getopt.h>
#include <string.h>
#include <toml++/toml.hpp>
#include <iomanip>
#include <boost/program_options.hpp>
using namespace std;

#include "../../util/timer.h"
#include "../../util/pim.h"
#include "../../util/common.h"
#include "cpu.h"

static data_t *A;
static data_t *B;
static data_t *C;

static double kernel_runtime = 0.0, data_movement = 0.0;
toml::table tbl;

// Params ---------------------------------------------------------------------
typedef struct Params
{
    int vector_size;
} Params;

void create_vectors(u64 nr_elements)
{
    A = (data_t *)malloc(nr_elements * sizeof(data_t));
    B = (data_t *)malloc(nr_elements * sizeof(data_t));
    C = (data_t *)malloc(nr_elements * sizeof(data_t));
    srand((unsigned)time(NULL));
    for (u64 i = 0; i < nr_elements; i++)
    {
        A[i] = rand() % 16;
        B[i] = rand() % 64;
    }
}

void hostVectorAdd(u64 vectorLength)
{
    for (u64 i = 0; i < vectorLength; i++)
    {
        C[i] = A[i] + B[i];
    }
}

void vectorAdd(int numRows, data_t **src1, data_t **src2, data_t **dest, SubArray *subArray)
{
    subArray->pimAdd(src1, src2, dest, numRows, sizeof(data_t) * 8);
    optional<int> r = tbl["pimAdd"]["R"].value<int>(), w = tbl["pimAdd"]["W"].value<int>(), l = tbl["pimAdd"]["L"].value<int>();
    kernel_runtime += (r.value() * READ_LATENCY + w.value() * WRITE_LATENCY + l.value() * LOGIC_LATENCY) * numRows;
}

int main(int argc, char **argv)
{
    struct Params appParam;
    struct PIMParams pimParam;
    pimParam.shouldVerify = true;
    pimParam.device = "BitSIMD";
    appParam.vector_size = 1234567;

    int opt;
    while ((opt = getopt(argc, argv, "i:d:")) >= 0)
    {
        switch (opt)
        {
        case 'i':
            appParam.vector_size = atoll(optarg);
            break;
        case 'd':
            pimParam.device = optarg;
            break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            exit(0);
        }
    }

    create_vectors(appParam.vector_size);
    int dataSize = sizeof(data_t) * 8;

    string configFile = "../../config/" + pimParam.device + ".toml";
    tbl = toml::parse_file(configFile);

    dramsim3::BenchmarkCPU *hostCPU = new dramsim3::BenchmarkCPU("/home/farzana/research/DRAMsim3/configs/DDR4_8Gb_x16_2133_2.ini", "./");
    dramsim3::BenchmarkCPU *deviceCPU = new dramsim3::BenchmarkCPU("/home/farzana/research/DRAMsim3/configs/DDR3_8Gb_x16_1866.ini", "./");

    PIM *pim = new PIM(deviceCPU);

    Bank *bank = pim->getBank();
    auto subArrayList = bank->getSubArrayList();

    u64 totalRowRequired = ceil(appParam.vector_size * 1.0 / bank->getNumberOfColumns());

    u64 loadPerSubArray = ceil(totalRowRequired * 1.0 / SUBARRAY_PER_BANK) * 2;

    cout << "Total row required: " << totalRowRequired << " Load Per Subarray: " << loadPerSubArray << "\n";

    u64 idx = 0;
    unordered_map<SubArray *, u64> loadBalanceMap;
    data_t *C_device = (data_t *)malloc(appParam.vector_size * sizeof(data_t));

    for (auto subArray : subArrayList)
    {
        loadBalanceMap[subArray] = loadPerSubArray;
    }

    u64 baseAddrA = 0;
    u64 baseAddrB = baseAddrA + appParam.vector_size * sizeof(data_t) + 64;
    cout << "Printing base addr: " << baseAddrA << "\t" << baseAddrB << "\n";
    while (idx < appParam.vector_size)
    {
        unordered_map<SubArray *, pair<MemMatrix *, MemMatrix *>> opMap;
        for (auto subArray : subArrayList)
        {
            if (idx >= appParam.vector_size)
                break;
            u64 availableRows = subArray->getAvailableRowCount();
            u64 allocatedRow = availableRows < loadPerSubArray ? (availableRows / 2) / dataSize : loadPerSubArray / 2;
            MemMatrix *pimA, *pimB;
            pimA = subArray->pimAlloc(allocatedRow * dataSize);
            if (pimA == nullptr)
            {
                exit(1);
            }
            pimB = subArray->pimAlloc(allocatedRow * dataSize);
            if (pimB == nullptr)
            {
                exit(1);
            }
            // cout << "Printing base row. PIMA: " << pimA->baseRowID << "\tPIMB: " << pimB->baseRowID << "\n";
            u64 stride = bank->getNumberOfColumns();
            pimA->baseAddressHost = idx;
            pimB->baseAddressHost = idx;
            u64 numB = 0;
            for (int i = 0; i < allocatedRow * dataSize; i += dataSize)
            {
                if (idx + stride > appParam.vector_size)
                {
                    stride = appParam.vector_size - idx;
                }
                subArray->pimCopyToDevice(A + idx, pimA, i, stride, dataSize);
                subArray->pimCopyToDevice(B + idx, pimB, i, stride, dataSize);
                numB += sizeof(data_t) * (stride);
                data_movement += subArray->elapsedTime;
                idx += stride;
            }
            pimA->numberOfELements = idx;
            pimB->numberOfELements = idx;
            opMap[subArray] = std::make_pair(pimA, pimB);
            loadBalanceMap[subArray] = (loadPerSubArray - (allocatedRow * 2));
            while (numB > 0)
            {
                baseAddrA += 64;
                baseAddrB += 64;
                hostCPU->addRequest(baseAddrA, false);
                hostCPU->addRequest(baseAddrB, false);
                numB -= 64;
            }
            u64 start = hostCPU->getClock();
            hostCPU->runAllPendingReq();
            u64 stop = hostCPU->getClock();
            data_movement += (stop - start) * hostCPU->GetTCK();
        }
        for (auto elem : opMap)
        {
            auto pimA = elem.second.first, pimB = elem.second.second;
            auto src1 = pimA->bitMat, src2 = pimB->bitMat;
            auto subArray = elem.first;
            vectorAdd(pimA->row, src1, src2, src1, subArray);
            u64 startIDx = elem.second.first->baseAddressHost, endIDx = elem.second.first->numberOfELements;
            for (int i = 0; i < pimA->row && startIDx < endIDx; i += dataSize)
            {
                for (int j = 0; j < bank->getNumberOfColumns() && startIDx < endIDx; j++)
                {
                    C_device[startIDx] = src1[i][j];
                    startIDx += 1;
                }
            }
            subArray->pimFree(pimA);
            subArray->pimFree(pimB);
        }
    }
    if (pimParam.shouldVerify)
    {

        hostVectorAdd(appParam.vector_size);
        for (u64 i = 0; i < appParam.vector_size; i++)
        {
            if (C[i] != C_device[i])
            {
                cout << "Incorrect result for index: " << i << ".\t"
                     << "Host value: " << C[i] << ",\tdevice value: " << C_device[i] << "\n";
                break;
            }
        }
    }

    cout << "Total Runtime: " << (kernel_runtime + data_movement) << " ns.\n"
         << "Data movement latency: " << data_movement << " ns.\n"
         << "Kernel runtime: " << kernel_runtime << " ns.\n"
         << endl;

    free(A);
    free(B);
    free(C);
    free(C_device);
    return 0;
}