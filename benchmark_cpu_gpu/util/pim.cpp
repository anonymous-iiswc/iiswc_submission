#include "pim.h"

SubArray::SubArray(u64 row, u64 col, u32 subArrayID, u64 startIDx, dramsim3::BenchmarkCPU *device, dramsim3::Config *config)
{
    totalRowCount = row;
    totalColumnCount = col;
    remainingRowCount = row;
    id = subArrayID;
    startRowID = startIDx;
    currRowID = startRowID;
    channelID = 0;
    bankGroupID = 0;
    bankID = 0;
    rankID = 0;
    head = nullptr;
    tail = nullptr;
    deviceCPU = device;
    deviceConfig = config;
}

MemMatrix *SubArray::pimAlloc(u64 requiredRow)
{
    if (remainingRowCount < requiredRow)
    {
        cerr << "Error: Out of space for subarray: " << id << "\n";
        return nullptr;
    }
    defrag();
    MemMatrix *memMat = new MemMatrix(requiredRow, totalColumnCount, currRowID);
    if (head == nullptr)
    {
        head = memMat;
    }
    else
    {
        if (tail != nullptr)
        {
            tail->next = memMat;
        }
        else
        {
            head->next = memMat;
        }
        tail = memMat;
    }
    memMat->next = nullptr;
    currRowID += requiredRow;
    remainingRowCount -= requiredRow;
    return memMat;
}

void SubArray::pimFree(MemMatrix *memMatrix)
{
    if (memMatrix == head)
    {
        head = memMatrix->next;
    }
    else
    {
        auto currMat = head;
        while (currMat->next != nullptr)
        {
            if (currMat->next == memMatrix)
            {
                currMat->next = memMatrix->next;
                if (memMatrix == tail)
                {
                    tail = currMat;
                }
                break;
            }
            currMat = currMat->next;
        }
    }
    remainingRowCount += memMatrix->row;
    delete memMatrix;
}

void SubArray::defrag()
{
    if (head != nullptr)
    {
        MemMatrix *currMat = head;
        while (currMat->next != nullptr)
        {
            auto nextMat = currMat->next;
            if (nextMat->baseRowID != currMat->baseRowID + currMat->row)
                nextMat->baseRowID = currMat->baseRowID + currMat->row;
            currMat = nextMat;
        }
        currRowID = currMat->baseRowID + currMat->row;
    }
}

void SubArray::pimAdd(data_t **inMat1, data_t **inMat2, data_t **outMat, u64 numRow, u16 elementWidth)
{
    for (int i = 0; i < numRow; i += sizeof(data_t) * 8)
    {
        for (int j = 0; j < totalColumnCount; j++)
        {
            outMat[i][j] = inMat1[i][j] + inMat2[i][j];
        }
    }
}

void SubArray::pimSub(data_t **inMat1, data_t **inMat2, data_t **outMat, u64 numRow, u16 elementWidth)
{
    for (int i = 0; i < numRow; i += sizeof(data_t) * 8)
    {
        for (int j = 0; j < totalColumnCount; j++)
        {
            outMat[i][j] = inMat1[i][j] - inMat2[i][j];
        }
    }
}

u64 SubArray::getAddress(u64 row, u64 col)
{
    u64 addr = 0;
    assert(row <= deviceConfig->ro_mask);
    assert(col <= deviceConfig->co_mask);
    assert(bankID <= deviceConfig->ba_mask);
    assert(bankGroupID <= deviceConfig->bg_mask);
    assert(channelID <= deviceConfig->ch_mask);
    assert(rankID <= deviceConfig->ra_mask);
    addr |= row << deviceConfig->ro_pos;
    addr |= col << deviceConfig->co_pos;
    addr |= bankID << deviceConfig->ba_pos;
    addr |= bankGroupID << deviceConfig->bg_pos;
    addr |= channelID << deviceConfig->ch_pos;
    addr |= rankID << deviceConfig->ra_pos;
    addr <<= deviceConfig->shift_bits;
    auto test = deviceConfig->AddressMapping(addr);
    assert(row == test.row);
    assert(col == test.column);
    assert(bankID == test.bank);
    assert(bankGroupID == test.bankgroup);
    assert(channelID == test.channel);
    assert(rankID == test.rank);
    return addr;
}

void SubArray::pimMul(data_t **inMat1, data_t **inMat2, data_t **outMat, u64 numRow, u16 elementWidth)
{
    for (int i = 0; i < numRow; i += sizeof(data_t) * 8)
    {
        for (int j = 0; j < totalColumnCount; j++)
        {
            outMat[i][j] = inMat1[i][j] * inMat2[i][j];
        }
    }
}

void SubArray::pimCopyToDevice(data_t *hostMat, MemMatrix *deviceMat, int deviceRowIdx, int numElement, u16 elementWidth)
{
    for (int i = 0; i < numElement; i++)
    {
        deviceMat->bitMat[deviceRowIdx][i] = hostMat[i];
    }
    int maxColumn = ceil(numElement * 1.0 / deviceConfig->device_width);
    maxColumn = maxColumn >> dramsim3::LogBase2(deviceConfig->BL);
    for (u64 i = deviceRowIdx; i < deviceRowIdx + elementWidth; i++)
    {
        for (int j = 0; j < maxColumn; j++)
        {
            deviceCPU->addRequest(getAddress(i + deviceMat->baseRowID, j), true);
        }
    }
    u64 start = deviceCPU->getClock();
    deviceCPU->runAllPendingReq();
    u64 stop = deviceCPU->getClock();
    elapsedTime += (stop - start) * deviceCPU->GetTCK();
}

void SubArray::pimCopyAndReplicateToDevice(data_t *hostMat, data_t *deviceMat, int numElement, u16 elementWidth)
{
    u64 numReplicate = totalColumnCount / numElement;
    for (int i = 0; i < numElement; i++)
    {
        for (int j = 0; j < numReplicate; j++)
        {
            deviceMat[numElement * j + i] = hostMat[i];
        }
        // hostCPU->addRequest((u64)&hostMat[i], false);
        // deviceCPU->addRequest((u64)&deviceMat[i], true);
    }
}

SubArray::~SubArray()
{
}
