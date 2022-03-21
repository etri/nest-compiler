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