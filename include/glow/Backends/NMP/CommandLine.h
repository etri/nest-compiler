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
 * @file: CommandLine.h
 * @brief description: Command line specifications to NMP
 * @date: 11 17, 2021
 */

#ifndef GLOW_NMPBACKEND_COMMANDLINE_H
#define GLOW_NMPBACKEND_COMMANDLINE_H

#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetOptions.h"

/// Options for specifying the parameters of NMP Backend CodeGen
extern llvm::cl::OptionCategory NMPBackendCat;
extern llvm::cl::opt<bool> disableFuseActvOpt;
extern llvm::cl::opt<bool> rescaleBiasOpt;
extern llvm::cl::opt<bool> recomputeConvPads;
extern llvm::cl::opt<bool> recomputePoolPads;
extern llvm::cl::opt<unsigned> haltLayerOpt;
extern llvm::cl::opt<bool> computeNMPSoftmaxOpt;

#endif // GLOW_NMPBACKEND_COMMANDLINE_H
