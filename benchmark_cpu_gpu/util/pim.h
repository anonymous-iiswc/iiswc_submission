#include <iostream>
#include <vector>
#include <cassert>

#include "common.h"
#include "cpu.h"

#define SUBARRAY_PER_BANK 32
#define READ_LATENCY 27.72
#define WRITE_LATENCY 28.98
#define LOGIC_LATENCY 2.52

class MemMatrix
{
public:
    MemMatrix(u64 _row, u64 _column, u64 baseRow)
    {
        row = _row;
        column = _column;
        baseRowID = baseRow;
        bitMat = (data_t **)malloc(_row * sizeof(data_t *));
        for (u64 i = 0; i < _row; i++)
        {
            bitMat[i] = (data_t *)malloc(_column * sizeof(data_t));
        }
    }
    data_t **bitMat;
    MemMatrix *next;
    u64 row, column, baseAddressHost, numberOfELements, baseRowID;
    ~MemMatrix()
    {
        for (u64 i = 0; i < row; i++)
        {
            delete bitMat[i];
        }
        delete[] bitMat;
    }
};

class SubArray
{
private:
    u32 id, channelID, rankID, bankID, bankGroupID;
    u64 startRowID, currRowID, remainingRowCount, totalRowCount, totalColumnCount;
    MemMatrix *head, *tail;
    void defrag();
    dramsim3::BenchmarkCPU *deviceCPU;
    dramsim3::Config *deviceConfig;

public:
    double elapsedTime = 0;
    SubArray(u64 row, u64 col, u32 subArrayID, u64 startIDx, dramsim3::BenchmarkCPU *device, dramsim3::Config *config);
    MemMatrix *pimAlloc(u64 requiredRow);
    /*
        inMat1 -> stores operand 1
        inMat2 -> stores operand 2
        outMat1 -> stores the result of the addition
        numRow -> number of rows of the subarray t o perform the addition on
    */
    void pimAdd(data_t **inMat1, data_t **inMat2, data_t **outMat, u64 numRow, u16 elementWidth);
    /*
        inMat1 -> stores operand 1
        inMat2 -> stores operand 2
        outMat1 -> stores the result of the addition
        numRow -> number of rows of the subarray to perform the addition on
    */
    void pimSub(data_t **inMat1, data_t **inMat2, data_t **outMat, u64 numRow, u16 elementWidth);
    void pimMul(data_t **inMat1, data_t **inMat2, data_t **outMat, u64 numRow, u16 elementWidth);
    void pimCopyToDevice(data_t *hostMat, MemMatrix *deviceMat, int deviceRowIdx, int numElement, u16 elementWidth);
    // void pimCopyToDevice(data_t *hostMat, data_t *deviceMat, int numElement, u16 elementWidth);
    void pimCopyAndReplicateToDevice(data_t *hostMat, data_t *deviceMat, int numElement, u16 elementWidth);
    u64 getTotalRowCount() { return totalRowCount; }
    u64 getTotalColumnCount() { return totalColumnCount; }
    u64 getAvailableRowCount() { return remainingRowCount - 1; }
    void pimFree(MemMatrix *memMatrix);
    u64 getAddress(u64 row, u64 col);
    ~SubArray();
};

class Bank
{
public:
    Bank(u64 rows, u64 columns, dramsim3::BenchmarkCPU *device, dramsim3::Config *deviceConfig)
    {
        rowCount = rows;
        columnCount = columns;
        u64 rowsPerSubarray = rows / SUBARRAY_PER_BANK;
        for (int i = 0; i < SUBARRAY_PER_BANK; i++)
        {
            SubArray *subArray = new SubArray(rowsPerSubarray, columns, i, rowsPerSubarray * i, device, deviceConfig);
            subArrayList.push_back(subArray);
        }
    }
    u64 getNumberOfRows() { return rowCount; }
    u64 getNumberOfColumns() { return columnCount; }
    vector<SubArray *> getSubArrayList() { return subArrayList; }
    ~Bank()
    {
        for (int i = 0; i < SUBARRAY_PER_BANK; i++)
        {
            delete subArrayList[i];
        }
    }

private:
    u32 id;
    u64 rowCount, columnCount;
    vector<SubArray *> subArrayList;
    dramsim3::BenchmarkCPU *deviceCPU;
    dramsim3::Config *deviceConfig;
};

class PIM
{
public:
    PIM(dramsim3::BenchmarkCPU *device)
    {
        deviceCPU = device;
        deviceConfig = deviceCPU->getMemorySystem()->getConfig();
    }
    Bank *getBank()
    {
        u64 rows_per_bank = deviceConfig->rows, column_per_row = deviceConfig->columns * deviceConfig->device_width;
        return new Bank(rows_per_bank, column_per_row, deviceCPU, deviceConfig);
    }
    dramsim3::BenchmarkCPU *deviceCPU;
    dramsim3::Config *deviceConfig;
};