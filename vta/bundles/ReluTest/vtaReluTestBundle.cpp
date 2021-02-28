

#include "vta/runtime.h"
#include "VTABundle.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include "vtaReluTestBundle.h"
SymbolTableEntry symbolTableEntry[2]={{"inputP",0,100352,'1'},{"outP",100352,100352,'1'}};
BundleConfig vtaReluTestBundle_config = {0, 200704, 0, 64, 2, symbolTableEntry};
int vtaReluTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  VTACommandHandle vtaCmdH{nullptr};
  int8_t* inputP = mutableWeight + 0;
  int8_t* outP = mutableWeight + 100352;
  relu(inputP, outP, 100352 );
  return 0;
}