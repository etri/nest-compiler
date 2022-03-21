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

BB.newNode("VTAConvolution")
.addInput("Input")
.addInput("Filter")
.addInput("Bias")
.addMember(MemberType::VectorUnsigned, "Kernels")
.addMember(MemberType::VectorUnsigned, "Strides")
.addMember(MemberType::VectorUnsigned, "Pads")
.addMember(MemberType::Unsigned, "Group")
.addResultFromCtorArg()
.addFusedActivationVTA()
.setDocstring("This is a VTA-specific convolution node that is "
"identical to the normal ConvolutionNode. That node "
"and convolution + relu are replaced with this one "
"for backend-specific code generation");



BB.includeBackendSpecificVerification("glow/VTASpecificNodesVerification.h");

