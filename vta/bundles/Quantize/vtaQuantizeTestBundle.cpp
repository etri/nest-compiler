

#include "vta/runtime.h"
#include "VTABundle.h"

#include "vtaQuantizeTestBundle.h"
SymbolTableEntry symbolTableEntry[2]={{"inputP",0,602112,'1'},{"outP",602112,150528,'1'}};
BundleConfig vtaQuantizeTestBundle_config = {0, 752640, 0, 64, 2, symbolTableEntry};
int vtaQuantizeTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  int8_t* inputP = mutableWeight + 0;
  int8_t* outP = mutableWeight + 602112;
  quantize(inputP, outP, 150528, 1/32.000000, 0 );
  return 0;
}