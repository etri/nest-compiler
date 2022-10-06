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
 * @file: NMPSpecificnodes.h
 * @brief description: Define specific Nodes to NMP
 * @date: 11 17, 2021
 */

#ifdef GLOW_WITH_NMP

BB.newBackendSpecificNode("NMPFullyConnected")
    .addInput("Input")
    .addInput("Weights")
    .addInput("Bias")
    .addFusedActivation()
    .addResultFromCtorArg()
    .setDocstring("Creates a target-specific FullyConnected node where the"
      "input matrix and the Weights are multiplied, and then the Bias is"
      "added to it, producing the Output. Input and Bias are regularly"
      "quantized, while Weights use row-wise quantization. Also, Relu"
      "can be fused with this FullyConnected operation.");

BB.includeBackendSpecificVerification("glow/NMPSpecificNodesVerification.h");

#endif // GLOW_WITH_NMP
