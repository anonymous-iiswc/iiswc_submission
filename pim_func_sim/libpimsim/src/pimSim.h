// File: pimSim.h
// PIM Functional Simulator - PIM Simulator Main Entry
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_SIM_H
#define LAVA_PIM_SIM_H

#include "libpimsim.h"
#include "pimDevice.h"
#include "pimParamsDram.h"
#include "pimParamsPerf.h"
#include "pimStats.h"
#include <vector>


//! @class  pimSim
//! @brief  PIM simulator singleton class
class pimSim
{
public:
  static pimSim* get();
  static void destroy();

  // Device creation and deletion
  bool createDevice(PimDeviceEnum deviceType, unsigned numBanks, unsigned numSubarrayPerBank, unsigned numRows, unsigned numCols);
  bool createDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName);
  bool deleteDevice();
  bool isValidDevice(bool showMsg = true) const;

  PimDeviceEnum getDeviceType() const;
  PimDeviceEnum getSimTarget() const;
  unsigned getNumCores() const;
  unsigned getNumRows() const;
  unsigned getNumCols() const;

  void showStats() const;
  pimStatsMgr* getStatsMgr() { return m_statsMgr; }
  pimParamsDram* getParamsDram() { return m_paramsDram; }
  pimParamsPerf* getParamsPerf() { return m_paramsPerf; }

  // Resource allocation and deletion
  PimObjId pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimDataType dataType);
  PimObjId pimAllocAssociated(unsigned bitsPerElement, PimObjId ref, PimDataType dataType);
  bool pimFree(PimObjId obj);

  // Data transfer
  bool pimCopyMainToDevice(void* src, PimObjId dest);
  bool pimCopyDeviceToMain(PimObjId src, void* dest);
  bool pimCopyMainToDeviceWithType(PimCopyEnum copyType, void* src, PimObjId dest);
  bool pimCopyDeviceToMainWithType(PimCopyEnum copyType, PimObjId src, void* dest);

  // Computation
  bool pimAdd(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimSub(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimDiv(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimAbs(PimObjId src, PimObjId dest);
  bool pimMul(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimOr(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimAnd(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimXor(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimXnor(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimGT(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimLT(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimEQ(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimMin(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimMax(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimPopCount(PimObjId src, PimObjId dest);
  bool pimRedSum(PimObjId src, int* sum);
  bool pimRedSumRanged(PimObjId src, unsigned idxBegin, unsigned idxEnd, int* sum);
  bool pimBroadcast(PimObjId dest, unsigned value);
  bool pimRotateR(PimObjId src);
  bool pimRotateL(PimObjId src);
  bool pimShiftR(PimObjId src);
  bool pimShiftL(PimObjId src);

  // BitSIMD-V micro ops
  bool pimOpReadRowToSa(PimObjId src, unsigned ofst);
  bool pimOpWriteSaToRow(PimObjId src, unsigned ofst);
  bool pimOpTRA(PimObjId src1, unsigned ofst1, PimObjId src2, unsigned ofst2, PimObjId src3, unsigned ofst3);
  bool pimOpMove(PimObjId objId, PimRowReg src, PimRowReg dest);
  bool pimOpSet(PimObjId objId, PimRowReg dest, bool val);
  bool pimOpNot(PimObjId objId, PimRowReg src, PimRowReg dest);
  bool pimOpAnd(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpOr(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpNand(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpNor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpXor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpXnor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpMaj(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg src3, PimRowReg dest);
  bool pimOpSel(PimObjId objId, PimRowReg cond, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpRotateRH(PimObjId objId, PimRowReg src);
  bool pimOpRotateLH(PimObjId objId, PimRowReg src);

private:
  pimSim();
  ~pimSim();
  pimSim(const pimSim&) = delete;
  pimSim operator=(const pimSim&) = delete;

  static pimSim* s_instance;

  // support one device for now
  pimDevice* m_device = nullptr;
  pimParamsDram* m_paramsDram = nullptr;
  pimParamsPerf* m_paramsPerf = nullptr;
  pimStatsMgr* m_statsMgr = nullptr;
};

#endif

