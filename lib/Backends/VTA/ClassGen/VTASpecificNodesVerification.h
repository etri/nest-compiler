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

#ifndef GLOW_LIB_BACKENDS_VTA_CLASSGEN_VTASPECIFICNODESVERIFICATION_H_
#define GLOW_LIB_BACKENDS_VTA_CLASSGEN_VTASPECIFICNODESVERIFICATION_H_

//#ifdef GLOW_WITH_CPU

#include "glow/Graph/VerifierHelper.h"

bool VTAConvolutionNode::verify() const {
  return true;
  //verifyConvolution(getInput(), getResult(), getFilter(), getBias(),
    //                       Kernels_, Strides_, Pads_, Group_);
}
//#endif // GLOW_WITH_CPU


#endif //GLOW_LIB_BACKENDS_VTA_CLASSGEN_VTASPECIFICNODESVERIFICATION_H_
