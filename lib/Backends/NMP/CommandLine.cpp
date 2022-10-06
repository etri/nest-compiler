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
 * @file: CommandLine.cpp
 * @brief description: Command line definitions specific to NMP
 * @date: 11 17, 2021
 */

#include "glow/Backends/NMP/CommandLine.h"

llvm::cl::OptionCategory NMPBackendCat("Glow NMP Backend Options");

llvm::cl::opt<bool> disableFuseActvOpt(
    "disable-fusing-actv",
    llvm::cl::desc("Disable fusing activations into Convolutions"
                   " / FullyConnected ops for NMP Backend."),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(NMPBackendCat));

llvm::cl::opt<bool> rescaleBiasOpt(
    "rescale-bias",
    llvm::cl::desc("Rescale Bias based on Conv/FullyConnected result."),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(NMPBackendCat));

llvm::cl::opt<bool> recomputeConvPads(
    "recompute-conv-pads",
    llvm::cl::desc("Recompute paddings based on convolution parameters."),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(NMPBackendCat));

llvm::cl::opt<bool> recomputePoolPads(
    "recompute-pool-pads",
    llvm::cl::desc("Recompute paddings based on pooling parameters."),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(NMPBackendCat));

llvm::cl::opt<unsigned> haltLayerOpt(
    "halt-layer",
    llvm::cl::desc("Number of layers to be executed before halt. "
                   "Default value = 0 which means no halt will "
                   "be inserted before the end of inference."),
    llvm::cl::Optional, llvm::cl::init(0), llvm::cl::cat(NMPBackendCat));

llvm::cl::opt<bool> computeNMPSoftmaxOpt(
    "compute-nmp-softmax",
    llvm::cl::desc("Compute the NMP Softmax operations. Default is true."),
    llvm::cl::Optional, llvm::cl::init(true), llvm::cl::cat(NMPBackendCat));

