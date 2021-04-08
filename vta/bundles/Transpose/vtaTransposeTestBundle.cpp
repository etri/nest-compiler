

#include "vta/runtime.h"
#include "vta/vtalib/include/Bundle/VTABundle.h"

#include "vtaTransposeTestBundle.h"
SymbolTableEntry symbolTableEntry[2]={{"inputP",0,150528,'1'},{"outP",150528,150528,'1'}};
BundleConfig vtaTransposeTestBundle_config = {0, 301056, 0, 64, 2, symbolTableEntry};
int vtaTransposeTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  int8_t* inputP = mutableWeight + 0;
  int8_t* outP = mutableWeight + 150528;
  transpose(inputP, outP, 1, 3, 224, 224, 1, 224, 224, 3, 0, 2, 3, 1 );
  return 0;
}