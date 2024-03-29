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
#ifndef GLOW_BACKENDS_RELAY_RELAY_H
#define GLOW_BACKENDS_RELAY_RELAY_H

#include "RelayDeviceManager.h"
#include "RelayFunction.h"

#include "glow/Backend/Backend.h"

#include <numeric>
#include <vector>

namespace glow {

/// This is the IR-Relay. It owns the IR, and the heap, and is able to
/// execute the instructions one at a time.
class Relay final : public BackendUsingGlowIR,
                    public IRInstructionProcessingHandler {
public:
  /// Ctor.
  Relay() = default;

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~Relay() override = default;

  std::string getBackendName() const override {
    return Named::getName().empty() ? getName() : Named::getName().str();
  }
  static std::string getName() { return "Relay"; }
  static unsigned numDevices() { return std::thread::hardware_concurrency(); }
  static std::vector<unsigned> scanDeviceIDs() {
    std::vector<unsigned> deviceIDs(Relay::numDevices());
    std::iota(std::begin(deviceIDs), std::end(deviceIDs), 0);
    return deviceIDs;
  }

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

  virtual void save(Function *F, llvm::StringRef outputDir,
                    llvm::StringRef bundleName,
                    llvm::StringRef mainEntryName) const override;

  bool supportsFusedActivation(Node *parent, Node *activation) const override {
    bool V = true;
    return V;
  }

  Expected<bool> transformPostLowering(
      Function *F, CompilationContext &cctx,
      const glow::runtime::DeviceInfo *devInfo = nullptr) const override;

  /// \returns a unitless value to be used when comparing Nodes or
  /// error if no estimate can be generated.
  Expected<double> estimateNodeCost(const glow::Node *node) const override;

  /// @}
  //
  /// \returns the size of metrics collected for a single TraceEvent.
  static size_t getTraceEventDataSizeStatic() { return sizeof(uint64_t); }
  size_t getTraceEventDataSize() const override {
    return Relay::getTraceEventDataSizeStatic();
  }

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override {
    return createRelayDeviceManager(deviceConfig);
  }

  void parseBackendSpecificOptions(const BackendOptions &opts) const;
  void setTarget(std::string target) { target_ = target; }
  void setTargetHost(std::string target_host) { target_host_ = target_host; }
  void setExportOption(std::string export_option) {
    export_option_ = export_option;
  }
  void setRequiredPass(std::string req_pass) { required_pass_ = req_pass; }
  void setDisabledPass(std::string disabled_pass) {
    disabled_pass_ = disabled_pass;
  }
  void setOptLevel(uint32_t opt_level) { opt_level_ = opt_level; }
  void setDebugMode(std::string debug_mode) { debug_mode_ = debug_mode; }

  std::string getTarget() { return target_; }

private:
  std::string target_;
  std::string target_host_;
  std::string export_option_; // option for export lib
  std::string required_pass_;
  std::string disabled_pass_;
  uint32_t opt_level_;
  std::string debug_mode_;
};

} // namespace glow

#endif // GLOW_BACKENDS_RELAY_RELAY_H
