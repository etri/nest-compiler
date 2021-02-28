

#include "vta/runtime.h"
#include "VTABundle.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include "vtaFloatConvTestBundle.h"
SymbolTableEntry symbolTableEntry[2]={{"inputP",0,150528,'1'},{"outP",602112,802816,'1'}};
BundleConfig vtaFloatConvTestBundle_config = {37888, 3813376, 0, 64, 2, symbolTableEntry};
int vtaFloatConvTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  VTACommandHandle vtaCmdH{nullptr};
  int8_t* filterP = (int8_t* )(constantWeight + 0);
  int32_t* biasP = (int32_t* )(constantWeight + 37632);
  int8_t* inputP = mutableWeight + 0;
  int8_t* outP = mutableWeight + 602112;
  convolutionFloat(inputP, filterP, (int8_t *)(biasP), outP, 1, 224, 224, 3, 64, 7, 7, 3, 2, 1, 1, 0, 1, 112, 112 );
  return 0;
}