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
