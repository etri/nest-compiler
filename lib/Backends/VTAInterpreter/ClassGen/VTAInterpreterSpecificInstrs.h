
BB.newBackendSpecificInstr("VTAInterpreterConvolution")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Unsigned, "Group")
    .addMember(MEMBER_TYPE_INFO(FusedActivation), "FusedActivation")
    .autoIRGen();

BB.includeBackendSpecificVerification("glow/VTAInterpreterSpecificInstrsVerification.h");
