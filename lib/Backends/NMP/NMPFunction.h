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
 * @file: NMPFunction.h
 * @brief description: Executes a NMP Function that representes the model.
 * @date: 11 18, 2021
 */

#ifndef GLOW_BACKENDS_NMP_NMPFUNCTION_H
#define GLOW_BACKENDS_NMP_NMPFUNCTION_H

#include "glow/LLVMIRCodeGen/GlowJIT.h"
#include "glow/LLVMIRCodeGen/LLVMCompiledFunction.h"

#include "glow/Backend/Backend.h"
#include "glow/Backend/BackendUtils.h"
#include "glow/Backend/CompiledFunction.h"

namespace glow {
 
// A Glow IR function compiled for the NMP using LLVM.
class NMPFunction final : public LLVMCompiledFunction {
public:
  NMPFunction(std::unique_ptr<GlowJIT> JIT,
              runtime::RuntimeBundle &&runtimeBundle);

  ~NMPFunction() override = default;

  Error execute(ExecutionContext *context) override;

  // returns the backend used to compile this function.
  std::string getCompileBackendName() const override { return "NMP"; }
};

} // end namespace glow

#endif // GLOW_BACKENDS_NMP_NMPFUNCTION_H
