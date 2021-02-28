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
