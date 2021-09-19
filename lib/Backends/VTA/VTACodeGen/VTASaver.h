#ifndef GLOW_VTASAVER_H
#define GLOW_VTASAVER_H


#include <string>

std::string const VTA_SAVE_COMMON =R"~(

#include "vta/runtime.h"
#include "VTABundle.h"
#include <time.h>
#include <iostream>
#include <fstream>
)~";

std::string const VTA_RUNTIME_HEADER =R"~(

#ifndef GLOW_VTARUNTIME_H
#define GLOW_VTARUNTIME_H
#include "vta/runtime.h"
#include "VTABundle.h"
#include <vector>

VTACommandHandle vtaCmdH{nullptr};
VTACommandHandle vtaCmdH1{nullptr};
VTACommandHandle vtaCmdH2{nullptr};
VTACommandHandle vtaCmdH3{nullptr};

extern std::vector<void*> vGemmUOpHandle[4];
extern std::vector<void*> vAddUOpHandle[4];
extern std::vector<void*> vResetUopHandle[4];
extern std::vector<void*> vReluUopHandle[4];
extern std::vector<void*> vMaxUopHandle[4];
extern std::vector<void*> vMinUopHandle[4];
extern std::vector<void*> vShiftUopHandle[4];

void initVTARuntime(){
  vtaCmdH = VTATLSCommandHandle();

  vGemmUOpHandle[0].clear();
  vAddUOpHandle[0].clear();
  vResetUopHandle[0].clear();
  vReluUopHandle[0].clear();
  vMaxUopHandle[0].clear();
  vMinUopHandle[0].clear();
  vShiftUopHandle[0].clear();

}

void destroyVTARuntime(){
  VTARuntimeShutdown();

  vGemmUOpHandle[0].clear();
  vAddUOpHandle[0].clear();
  vResetUopHandle[0].clear();
  vReluUopHandle[0].clear();
  vMaxUopHandle[0].clear();
  vMinUopHandle[0].clear();
  vShiftUopHandle[0].clear();

}

void initVTARuntime1(){
  vtaCmdH1 = VTATLSCommandHandle(1);

  vGemmUOpHandle[1].clear();
  vAddUOpHandle[1].clear();
  vResetUopHandle[1].clear();
  vReluUopHandle[1].clear();
  vMaxUopHandle[1].clear();
  vMinUopHandle[1].clear();
  vShiftUopHandle[1].clear();

}

void destroyVTARuntime1(){
  VTARuntimeShutdown(1);

  vGemmUOpHandle[1].clear();
  vAddUOpHandle[1].clear();
  vResetUopHandle[1].clear();
  vReluUopHandle[1].clear();
  vMaxUopHandle[1].clear();
  vMinUopHandle[1].clear();
  vShiftUopHandle[1].clear();

}

void initVTARuntime2(){
  vtaCmdH2 = VTATLSCommandHandle(2);

  vGemmUOpHandle[2].clear();
  vAddUOpHandle[2].clear();
  vResetUopHandle[2].clear();
  vReluUopHandle[2].clear();
  vMaxUopHandle[2].clear();
  vMinUopHandle[2].clear();
  vShiftUopHandle[2].clear();

}

void destroyVTARuntime2(){
  VTARuntimeShutdown(2);

  vGemmUOpHandle[2].clear();
  vAddUOpHandle[2].clear();
  vResetUopHandle[2].clear();
  vReluUopHandle[2].clear();
  vMaxUopHandle[2].clear();
  vMinUopHandle[2].clear();
  vShiftUopHandle[2].clear();

}

void initVTARuntime3(){
  vtaCmdH3 = VTATLSCommandHandle(3);

  vGemmUOpHandle[3].clear();
  vAddUOpHandle[3].clear();
  vResetUopHandle[3].clear();
  vReluUopHandle[3].clear();
  vMaxUopHandle[3].clear();
  vMinUopHandle[3].clear();
  vShiftUopHandle[3].clear();

}

void destroyVTARuntime3(){
  VTARuntimeShutdown(3);

  vGemmUOpHandle[3].clear();
  vAddUOpHandle[3].clear();
  vResetUopHandle[3].clear();
  vReluUopHandle[3].clear();
  vMaxUopHandle[3].clear();
  vMinUopHandle[3].clear();
  vShiftUopHandle[3].clear();

}

#endif // GLOW_VTARUNTIME_H

)~";

std::string const VTA_BUNDLE_HEADER_0 =R"~(

#include <stdint.h>

// ---------------------------------------------------------------
//                       Common definitions
// ---------------------------------------------------------------
#ifndef _GLOW_BUNDLE_COMMON_DEFS
#define _GLOW_BUNDLE_COMMON_DEFS

// Glow bundle error code for correct execution.
#define GLOW_SUCCESS 0

// Type describing a symbol table entry of a generated bundle.
struct SymbolTableEntry {
  // Name of a variable.
  const char *name;
  // Offset of the variable inside the memory area.
  uint64_t offset;
  // The number of elements inside this variable.
  uint64_t size;
  // Variable kind: 1 if it is a mutable variable, 0 otherwise.
  char kind;
};

// Type describing the config of a generated bundle.
struct BundleConfig {
  // Size of the constant weight variables memory area.
  uint64_t constantWeightVarsMemSize;
  // Size of the mutable weight variables memory area.
  uint64_t mutableWeightVarsMemSize;
  // Size of the activations memory area.
  uint64_t activationsMemSize;
  // Alignment to be used for weights and activations.
  uint64_t alignment;
  // Number of symbols in the symbol table.
  uint64_t numSymbols;
  // Symbol table.
  const SymbolTableEntry *symbolTable;
};

#endif

#ifdef __cplusplus
extern "C" {
#endif
)~";


std::string const VTA_BUNDLE_HEADER_1 =R"~(
#ifdef __cplusplus
}
#endif
)~";

#endif //GLOW_VTASAVER_H
