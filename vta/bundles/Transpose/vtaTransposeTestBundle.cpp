/*****************************************************************************
	*
	* Copyright Next-Generation System Software Research Group, All rights reserved.
	* Future Computing Research Division, Artificial Intelligence Reserch Laboratory
	* Electronics and Telecommunications Research Institute (ETRI)
	*
	* THESE DOCUMENTS CONTAIN CONFIDENTIAL INFORMATION AND KNOWLEDGE
	* WHICH IS THE PROPERTY OF ETRI. NO PART OF THIS PUBLICATION IS
	* TO BE USED FOR ANY OTHER PURPOSE, AND THESE ARE NOT TO BE"
	* REPRODUCED, COPIED, DISCLOSED, TRANSMITTED, STORED IN A RETRIEVAL
	"* SYSTEM OR TRANSLATED INTO ANY OTHER HUMAN OR COMPUTER LANGUAGE,
	* IN ANY FORM, BY ANY MEANS, IN WHOLE OR IN PART, WITHOUT THE
	* COMPLETE PRIOR WRITTEN PERMISSION OF ETRI.
	*
	* LICENSE file : README_LICENSE_ETRI located in the top directory
	*
*****************************************************************************/


#include "vta/runtime.h"
#include "VTABundle.h"

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