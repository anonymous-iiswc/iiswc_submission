// Test: C++ version of vector logical

#include "libpimsim.h"
#include <iostream>
#include <vector>

int main()
{
  std::cout << "PIM test: Vector logical" << std::endl;

  unsigned numCores = 4;
  unsigned numRows = 128;
  unsigned numCols = 256;

  PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, 1, numCores, numRows, numCols);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  unsigned numElements = 512;
  unsigned bitsPerElement = 32;

  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numElements, bitsPerElement, PIM_INT32);
  if (obj1 == -1) {
    std::cout << "Abort" << std::endl;
    return 1;
  }
  PimObjId obj2 = pimAllocAssociated(bitsPerElement, obj1, PIM_INT32);
  if (obj2 == -1) {
    std::cout << "Abort" << std::endl;
    return 1;
  }
  PimObjId obj3 = pimAllocAssociated(bitsPerElement, obj1, PIM_INT32);
  if (obj3 == -1) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  std::vector<int> src1(numElements);
  std::vector<int> src2(numElements);
  std::vector<int> dest(numElements);

  // assign some initial values
  for (unsigned i = 0; i < numElements; ++i) {
    src1[i] = i;
    src2[i] = i * 2 + 10;
  }

  status = pimCopyHostToDevice((void*)src1.data(), obj1);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  status = pimCopyHostToDevice((void*)src2.data(), obj2);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  status = pimOr(obj1, obj2, obj3);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  status = pimCopyDeviceToHost(obj3, (void*)dest.data());
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  // check results
  bool ok = true;
  for (unsigned i = 0; i < numElements; ++i) {
    int result = src1[i] | src2[i];
    if (dest[i] != result) {
      std::cout << "Wrong answer for logical or: " << src1[i] << " | " << src2[i] << " = " << dest[i] << " (expected " << result << ")" << std::endl;
      ok = false;
      break;
    }
  }

  status = pimAnd(obj1, obj2, obj3);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  status = pimCopyDeviceToHost(obj3, (void*)dest.data());
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  for (unsigned i = 0; i < numElements; ++i) {
    int result = src1[i] & src2[i];
    if (dest[i] != result) {
      std::cout << "Wrong answer for logical and: " << src1[i] << " & " << src2[i] << " = " << dest[i] << " (expected " << result << ")" << std::endl;
      ok = false;
      break;
    }
  }

  status = pimXor(obj1, obj2, obj3);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  status = pimCopyDeviceToHost(obj3, (void*)dest.data());
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  for (unsigned i = 0; i < numElements; ++i) {
    int result = src1[i] ^ src2[i];
    if (dest[i] != result) {
      std::cout << "Wrong answer for xor: " << src1[i] << " ^ " << src2[i] << " = " << dest[i] << " (expected " << result << ")" << std::endl;
      ok = false;
      break;
    }
  }

  pimShowStats();
  if (ok) {
    std::cout << "All correct!" << std::endl;
  }

  return 0;
}
