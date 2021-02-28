

#include "vta/runtime.h"
#include "vta/vtalib/include/Bundle/VTABundle.h"

#include "vtaDequantizeTestBundle.h"
SymbolTableEntry symbolTableEntry[2]={{"inputP",0,1000,'1'},{"outP",1000,4000,'1'}};
BundleConfig vtaDequantizeTestBundle_config = {0, 5000, 0, 64, 2, symbolTableEntry};
int vtaDequantizeTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  int8_t* inputP = mutableWeight + 0;
  int8_t* outP = mutableWeight + 1000;
  dequantize(inputP, outP, 1000, 1/4.000000, 0 );
  return 0;
}