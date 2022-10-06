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
 * @file: NMPLLVMIRGen.h
 * @brief description: NMP specific Code Generation for the model.
 * @date: 11 18, 2021
 */

#ifndef GLOW_BACKENDS_NMP_NMPLLVMIRGEN_H
#define GLOW_BACKENDS_NMP_NMPLLVMIRGEN_H

#include "NMPTensorAnalysis.h"
#include "glow/Backends/NMP/nmp_common.h"
#include "glow/LLVMIRCodeGen/LLVMIRGen.h"
#include "llvm/Target/TargetMachine.h"

namespace glow {

class PlaceholderBindings;
class IRFunction;
class Value;
class Tensor;
class Constant;
class Instruction;
class WeightVar;
class AllocationsInfo;
class LLVMBackendOptions;

// This is a class containing a common logic for the generation of the LLVM IR
// from an IRFunction. The primary clients of this class are JITs and bundlers.
class NMPLLVMIRGen : public LLVMIRGen {

public:
  ~NMPLLVMIRGen() override = default;

  explicit NMPLLVMIRGen(const IRFunction *M, AllocationsInfo &allocationsInfo,
                        std::string mainEntryName, llvm::StringRef libjitBC);

  explicit NMPLLVMIRGen(const IRFunction *M, AllocationsInfo &allocationsInfo,
                        std::string mainEntryName, llvm::StringRef libjitBC,
                        llvm::ArrayRef<llvm::MemoryBufferRef> objectRegistry);

  // Add a bundle object \p objectName to be archived to the bundle. The object
  // must be registered in the \ref objectRegistry_ otherwise error is thrown.
  void addBundleObject(llvm::StringRef objectName) override;

  // Creates an LLVM module, the entry function, etc.
  void initCodeGen() override;

  // Emits the code of the entry function, performs optimizations, etc.
  void performCodeGen() override;

  // Emit LLVM-IR for the whole IRFunction from Glow Low-level IR. 
  void generateLLVMIRForModule(llvm::IRBuilder<> &builder) override;

  // Load base addresses of different memory areas (activations, const
  // weightvars, mutable weight vars) so that they can be reused inside
  // the body of the function.
  void loadBaseAddresses(llvm::IRBuilder<> &builder) override;

  // Generates LLVM IR that computes the offset address of val.
  llvm::Value *emitValueAddress(llvm::IRBuilder<> &builder,
                                const glow::Value *val) override;

  // \returns a libnmp API function by name.
  llvm::Function *getFunction(const std::string &name) override;

  // get the input/output offset & scales to dump header
  std::pair<unsigned, float> getInputData() { return inputAddrScale; }
  std::pair<unsigned, float> getOutputData() { return outputAddrScale; }

private:
  // Emit LLVM-IR for the instruction I, using the LLVM builder.
  int32_t generateIRCodeForInstr(llvm::IRBuilder<> &builder,
                                 const glow::Instruction *I);

  // Generates LLVM IR that computes the NMP cte value of elemKind.
  llvm::Value *emitElemKind(llvm::IRBuilder<> &builder, ElemKind elemKind);

  // Generate LLVM IR to call MAX_ or AVG_ Pool operation
  template <typename T>
  void generatePoolInstr(llvm::IRBuilder<> &builder, const int NMP_POOL_TYP,
                         const glow::Instruction *I);

  // Generate LLVM IR to call Element wise operation.
  template <typename T>
  void generateEltwiseInstr(llvm::IRBuilder<> &builder,
                            const int NMP_ELTWISE_TYP,
                            const glow::Instruction *I);

  // Emit LLVM IR code for fused activation type
  template <class InstructionTy>
  llvm::Value *emitFusedActivationType(llvm::IRBuilder<> &builder,
                                       const InstructionTy *I);

  // Save the offset and scale of the input and output data
  // This info will be used by the bundle to generate header
  std::pair<uint64_t, float> inputAddrScale;
  std::pair<uint64_t, float> outputAddrScale;

}; // class NMPLLVMIRGen

}; // namespace glow

#endif // GLOW_BACKENDS_NMP_NMPLLVMIRGEN_H
