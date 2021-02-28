

#include "vta/runtime.h"
#include "VTABundle.h"

#include "vtaElemDivTestBundle.h"
SymbolTableEntry symbolTableEntry[3]={{"inputP0",0,100352,'1'},{"inputP1",100352,100352,'1'},{"outP",200704,100352,'1'}};
BundleConfig vtaElemDivTestBundle_config = {0, 301056, 0, 64, 3, symbolTableEntry};
int vtaElemDivTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  int8_t* inputP0 = mutableWeight + 0;
  int8_t* inputP1 = mutableWeight + 100352;
  int8_t* outP = mutableWeight + 200704;
  elemdiv(inputP0, 1.0/0.250000, 0, inputP1, 1.0/1.000000, 0, outP, 1.0/0.500000, 0, 100352 );
  return 0;
}