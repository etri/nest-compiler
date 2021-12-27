

#include "vta/runtime.h"
#include "VTABundle.h"

#include "vtaMaxPoolTestBundle.h"
SymbolTableEntry symbolTableEntry[2]={{"inputP",0,3136,'1'},{"outP",3136,1024,'1'}};
BundleConfig vtaMaxPoolTestBundle_config = {0, 4160, 0, 64, 2, symbolTableEntry};
int vtaMaxPoolTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  int8_t* inputP = mutableWeight + 0;
  int8_t* outP = mutableWeight + 3136;
  maxpool(inputP, outP, 1, 14, 14, 16, 2, 2, 1, 2 );
  return 0;
}