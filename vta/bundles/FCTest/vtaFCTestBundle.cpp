

#include "vta/runtime.h"
#include "vta/vtalib/include/Bundle/VTABundle.h"

#include "vtaFCTestBundle.h"
SymbolTableEntry symbolTableEntry[2]={{"inputP",0,2048,'1'},{"outP",2048,1000,'1'}};
BundleConfig vtaFCTestBundle_config = {2052000, 3048, 0, 64, 2, symbolTableEntry};
int vtaFCTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  int8_t* filterP = constantWeight + 0;
  int8_t* biasP = constantWeight + 2048000;
  int8_t* inputP = mutableWeight + 0;
  int8_t* outP = mutableWeight + 2048;
  fullyconnected(inputP, 1.0/8.000000, 0, filterP, 1.0/128.000000, 0, biasP, 1.0/1024.000000, 0, outP, 1.0/4.000000, 0, 1, 2048, 2048, 1000, 1, 1000, 1 );
  return 0;
}