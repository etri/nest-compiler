/*****************************************************************************
*
* Copyright Next-Generation System Software Research Group, All rights reserved.
* Future Computing Research Division, Artificial Intelligence Reserch Laboratory
* Electronics and Telecommunications Research Institute (ETRI)
*
* THESE DOCUMENTS CONTAIN CONFIDENTIAL INFORMATION AND KNOWLEDGE
* WHICH IS THE PROPERTY OF ETRI. NO PART OF THIS PUBLICATION IS
* TO BE USED FOR ANY OTHER PURPOSE, AND THESE ARE NOT TO BE
* REPRODUCED, COPIED, DISCLOSED, TRANSMITTED, STORED IN A RETRIEVAL
* SYSTEM OR TRANSLATED INTO ANY OTHER HUMAN OR COMPUTER LANGUAGE,
* IN ANY FORM, BY ANY MEANS, IN WHOLE OR IN PART, WITHOUT THE
* COMPLETE PRIOR WRITTEN PERMISSION OF ETRI.
*
* LICENSE file : LICENSE_ETRI located in the top directory
*
*****************************************************************************/


#include "vta/runtime.h"
#include "VTABundle.h"

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