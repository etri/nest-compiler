

#ifndef GLOW_VTARUNTIME_H
#define GLOW_VTARUNTIME_H
#include "vta/runtime.h"
#include "VTABundle.h"
#include <vector>

VTACommandHandle vtaCmdH{nullptr};
extern std::vector<void*> vGemmUOpHandle;
extern std::vector<void*> vAddUOpHandle;
extern std::vector<void*> vResetUopHandle;
extern std::vector<void*> vReluUopHandle;
extern std::vector<void*> vMaxUopHandle;
extern std::vector<void*> vMinUopHandle;
extern std::vector<void*> vShiftUopHandle;

void initVTARuntime(){
  xlnk_reset();
  vtaCmdH = VTATLSCommandHandle();

  vGemmUOpHandle.clear();
  vAddUOpHandle.clear();
  vResetUopHandle.clear();
  vReluUopHandle.clear();
  vMaxUopHandle.clear();
  vMinUopHandle.clear();
  vShiftUopHandle.clear();

}

void destroyVTARuntime(){
  VTARuntimeShutdown();

  vGemmUOpHandle.clear();
  vAddUOpHandle.clear();
  vResetUopHandle.clear();
  vReluUopHandle.clear();
  vMaxUopHandle.clear();
  vMinUopHandle.clear();
  vShiftUopHandle.clear();

}

#endif // GLOW_VTARUNTIME_H

