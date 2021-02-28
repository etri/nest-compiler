

#include "vta/runtime.h"
#include "vta/vtalib/include/Bundle/VTABundle.h"

#include "vtaSoftMaxTestBundle.h"
SymbolTableEntry symbolTableEntry[2]={{"inputP",0,4000,'1'},{"outP",4000,4000,'1'}};
BundleConfig vtaSoftMaxTestBundle_config = {0, 8000, 0, 64, 2, symbolTableEntry};
int vtaSoftMaxTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  int8_t* inputP = mutableWeight + 0;
  int8_t* outP = mutableWeight + 4000;
  softmax(inputP, outP, 1, 1000, 1, 1000 );
  return 0;
}