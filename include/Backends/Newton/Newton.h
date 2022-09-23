/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * Modifications copyright (C) 2022 <ETRI/Yongin Kwon>
 */
#ifndef NESTC_BACKENDS_NEWTON_NEWTON_H
#define NESTC_BACKENDS_NEWTON_NEWTON_H

#include "NewtonDeviceManager.h"
#include "NewtonFunction.h"

#include "glow/Backend/Backend.h"

namespace glow {

/// This is the IR-interpreter. It owns the IR, and the heap, and is able to
/// execute the instructions one at a time.
class Newton final : public BackendUsingGlowIR,
                     public IRInstructionProcessingHandler {
 public:
  /// Ctor.
  Newton() = default;

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~Newton() override = default;

  std::string getBackendName() const override {
    return Named::getName().empty() ? getName() : Named::getName().str();
  }
  static std::string getName() { return "Newton"; }
  static unsigned numDevices() { return std::thread::hardware_concurrency(); }

  std::unique_ptr<CompiledFunction>
  compileIR(std::unique_ptr<IRFunction> IR) const override;

  std::unique_ptr<CompiledFunction>
  compileIRWithoutConstants(std::unique_ptr<IRFunction> IR) const;

  Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override;

  bool isOpSupported(const NodeInfo &NI) const override;

  bool verify(const Function &F, bool verbose = true) const override;
  bool verify(const IRFunction &IR) const override;

  bool shouldLower(const Node *N) const override;

  Expected<bool> transformPostLowering(
      Function *F, CompilationContext &cctx,
      const glow::runtime::DeviceInfo *devInfo = nullptr) const override;

  /// @}
  //
  /// \returns the size of metrics collected for a single TraceEvent.
  static size_t getTraceEventDataSizeStatic() { return sizeof(uint64_t); }
  size_t getTraceEventDataSize() const override {
    return Newton::getTraceEventDataSizeStatic();
  }

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override {
    return createNewtonDeviceManager(deviceConfig);
  }

  void parseBackendSpecificOptions(const BackendOptions &opts) const;

  virtual void save(Function *F, llvm::StringRef outputDir,
                    llvm::StringRef bundleName,
                    llvm::StringRef mainEntryName) const override;



  bool supportsFusedActivation(Node *parent, Node *activation) const override {
    bool V = parent->getKind() == Kinded::Kind::FullyConnectedNodeKind &&
             activation->getKind() == Kinded::Kind::ReluNodeKind;
    return V;
  }

};

} // namespace glow

#endif // NESTC_BACKENDS_NEWTON_NEWTON_H