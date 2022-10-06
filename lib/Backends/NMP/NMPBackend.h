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
 * @file: NMPBackend.h
 * @brief description: Implementation of the NMP Backend interface.
 * @date: 11 17, 2021
 */

#ifndef GLOW_BACKENDS_NMP_NMPBACKEND_H
#define GLOW_BACKENDS_NMP_NMPBACKEND_H

#include "NMPBundleSaver.h"
#include "NMPDeviceManager.h"

#include "glow/Backend/Backend.h"
#include "glow/Base/Tensor.h"
#include "glow/LLVMIRCodeGen/CommandLine.h"
#include "glow/LLVMIRCodeGen/LLVMBackend.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/IRBuilder.h"

#include <vector>

/* static llvm::cl::OptionCategory NMPUtilsCat("NMP Category"); */

namespace glow {

class NodeInfo;
class LLVMBackendOptions;

// This is the implementation of the NMP Backend interface.
class NMPBackend : public LLVMBackend {
public:
  NMPBackend();

  ~NMPBackend() override = default;

  std::string getBackendName() const override {
    return Named::getName().empty() ? getName() : Named::getName().str();
  }
  static std::string getName() { return "NMP"; }
  static unsigned numDevices();
  static std::vector<unsigned> scanDeviceIDs();

  // Used by the compiler during graph optimization and before code generation,
  // giving the backend an opportunity to transform the graph before IRGen. The
  // NMP backend may insert device-specific nodes & fix the data layout to NCHW.
  // returns an Expected True if the graph was modified.
  Expected<bool> transformPostLowering(
      Function *F, CompilationContext &cctx,
      const glow::runtime::DeviceInfo *devInfo = nullptr) const override;

  // Called after the end of the optimization pipeline and before code
  // generation, giving the backend another opportunity to transform the
  // graph before IRGen. The NMP backend will correct the scales to meet
  // the requirements of the specific arithmetic operations of the NMP.
  // returns a boolean to let you know if the graph has been modified.
  Expected<bool>
  transformPostOptPipeline(Function *F,
                           CompilationContext &cctx) const override;

  // returns whether the provided NI is supported by the backend.
  bool isOpSupported(const NodeInfo &NI) const override;

  bool verify(const Function &F, bool verbose) const override;

  TensorLayoutCommon &getTensorLayoutRequirements() const override;

  bool shouldLower(const Node *N) const override;

  bool shouldShareBuffers() const override { return false; }
  
  // returns whether the backend supports fusing activation into parent.
  bool supportsFusedActivation(Node *parent, Node *activation) const override;

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override {
    return createNMPDeviceManager(deviceConfig);
  }

  llvm::ArrayRef<llvm::MemoryBufferRef> getObjectRegistry() const override;

  // This is the implementation of the LLVMBackend interface for NMP Device.
  virtual std::unique_ptr<LLVMIRGen>
  createIRGen(const IRFunction *IR,
              AllocationsInfo &allocationsInfo) const override;

  void save(Function *F, llvm::StringRef outputDir, llvm::StringRef bundleName,
            llvm::StringRef mainEntryName) const override;

  void saveFunctions(llvm::ArrayRef<BundleEntry> entries,
                     llvm::StringRef outputDir,
                     llvm::StringRef bundleName) const override;

  std::unique_ptr<NMPBundleSaver>
  createNMPBundleSaver(const NMPBackend &nmpBackend, llvm::StringRef outputDir,
                       llvm::StringRef bundleName) const;

protected:
  std::unique_ptr<CompiledFunction>
  createCompiledFunction(std::unique_ptr<GlowJIT> JIT,
                         runtime::RuntimeBundle &&runtimeBundle) const override;

  llvm::StringRef getLibjitBitcode() const override;
};

} // namespace glow

#endif // GLOW_BACKENDS_NMP_NMPBACKEND_H
