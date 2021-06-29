
BB.newNode("VTAInterpreterConvolution")
.addInput("Input")
.addInput("Filter")
.addInput("Bias")
.addMember(MemberType::VectorUnsigned, "Kernels")
.addMember(MemberType::VectorUnsigned, "Strides")
.addMember(MemberType::VectorUnsigned, "Pads")
.addMember(MemberType::Unsigned, "Group")
.addMember(MemberType::Boolean, "DoRelu")
.addResultFromCtorArg()
.setDocstring("This is a VTA-specific convolution node that is "
"identical to the normal ConvolutionNode. That node "
"and convolution + relu are replaced with this one "
"for backend-specific code generation");



BB.includeBackendSpecificVerification("glow/VTAInterpreterSpecificNodesVerification.h");

