

#include "vta/runtime.h"
#include "vta/vtalib/include/Bundle/VTABundle.h"

#include "vtaElemAddTestBundle.h"
SymbolTableEntry symbolTableEntry[3]={{"inputP0",0,100352,'1'},{"inputP1",100352,100352,'1'},{"outP",200704,100352,'1'}};
BundleConfig vtaElemAddTestBundle_config = {0, 301056, 0, 64, 3, symbolTableEntry};
int vtaElemAddTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  int8_t* inputP0 = mutableWeight + 0;
  int8_t* inputP1 = mutableWeight + 100352;
  int8_t* outP = mutableWeight + 200704;
  elemadd(inputP0, 1.0/4.000000, 0, inputP1, 1.0/2.000000, 0, outP, 1.0/1.000000, 0, 100352 );
  return 0;
}