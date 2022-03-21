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

#include "vtaElemSubTestBundle.h"
SymbolTableEntry symbolTableEntry[3]={{"inputP0",0,100352,'1'},{"inputP1",100352,100352,'1'},{"outP",200704,100352,'1'}};
BundleConfig vtaElemSubTestBundle_config = {0, 301056, 0, 64, 3, symbolTableEntry};
int vtaElemSubTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  int8_t* inputP0 = mutableWeight + 0;
  int8_t* inputP1 = mutableWeight + 100352;
  int8_t* outP = mutableWeight + 200704;
  elemsub(inputP0, 1.0/0.250000, 0, inputP1, 1.0/1.000000, 0, outP, 1.0/0.500000, 0, 100352 );
  return 0;
}