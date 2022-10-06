/**
 * Copyright (c) 2021, Etri.
 *
 * This program or software including the accompanying associated documentation
 * ("Software") is the proprietary software of Etri and/or its licensors.
 * Etri reserves all rights in and to the Software and all intellectual
 * property therein. The Software is distributed on an "AS IS" basis, without
 * warranties or conditions of any kind, unless required by applicable law or a
 * written agreement.
 *
 * @file: NMPSpecificInstrs.h
 * @brief description: Define specific Instructions to NMP
 * @date: 11 17, 2021
 */

#ifdef GLOW_WITH_NMP

BB.newBackendSpecificInstr("NMPFullyConnected")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Weights", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addFusedActivation()
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.includeBackendSpecificVerification("glow/NMPSpecificInstrsVerification.h");

#endif // GLOW_WITH_NMP
