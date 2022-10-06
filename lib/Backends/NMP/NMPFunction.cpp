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
 * @file: NMPFunction.cpp
 * @brief description: Executes a NMP Function that representes the model.
 * @date: 11 18, 2021
 */

#include "NMPFunction.h"

#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Support/Compiler.h"
#include "glow/Support/Memory.h"

using namespace glow;

NMPFunction::NMPFunction(std::unique_ptr<GlowJIT> JIT,
                         runtime::RuntimeBundle &&runtimeBundle)
    : LLVMCompiledFunction(std::move(JIT), std::move(runtimeBundle)) {}

Error NMPFunction::execute(ExecutionContext *context) {
  return LLVMCompiledFunction::execute(context);
}
