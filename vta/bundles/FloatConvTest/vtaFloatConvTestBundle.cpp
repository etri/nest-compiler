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