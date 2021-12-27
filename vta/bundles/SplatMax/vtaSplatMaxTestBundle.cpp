

#include "vta/runtime.h"
#include "VTABundle.h"

#include "vtaSplatMaxTestBundle.h"
SymbolTableEntry symbolTableEntry[2]={{"outP",0,802816,'1'},{"inputP",802816,802816,'1'}};
BundleConfig vtaSplatMaxTestBundle_config = {0, 1605632, 0, 64, 2, symbolTableEntry};
int vtaSplatMaxTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  int8_t* outP = mutableWeight + 0;
  splat(outP, 802816, 0 );
  int8_t* inputP = mutableWeight + 802816;
  elemmax(inputP, 1.0/8.000000, 0, outP, 1.0/8.000000, 0, outP, 1.0/8.000000, 0, 802816 );
  return 0;
}