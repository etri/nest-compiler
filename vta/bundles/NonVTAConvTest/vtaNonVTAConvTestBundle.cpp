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

#include "vtaNonVTAConvTestBundle.h"
SymbolTableEntry symbolTableEntry[2]={{"inputP",0,150528,'1'},{"outP",150528,802816,'1'}};
BundleConfig vtaNonVTAConvTestBundle_config = {9664, 953344, 0, 64, 2, symbolTableEntry};
int vtaNonVTAConvTestMainEntry(int8_t *constantWeight, int8_t *mutableWeight, int8_t *activations){
  xlnk_reset();
  int8_t* filterP = constantWeight + 0;
  int8_t* biasP = constantWeight + 9408;
  int8_t* inputP = mutableWeight + 0;
  int8_t* outP = mutableWeight + 150528;
  nonvtaconvolution(inputP, 1.0/32.000000, 0, filterP, 1.0/64.000000, 0, biasP, 1.0/2048.000000, 0, outP, 1.0/8.000000, 0, 1, 224, 224, 3, 64, 7, 7, 3, 2, 1, 1, 0, 1, 112, 112 );
  return 0;
}