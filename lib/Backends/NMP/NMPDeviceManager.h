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
 * @file: NMPDeviceManager.h
 * @brief description: Manager memory device of NMP Backend.
 * @date: 11 18, 2021
 */
#ifndef GLOW_BACKENDS_NMP_NMPDEVICEMANAGER_H
#define GLOW_BACKENDS_NMP_NMPDEVICEMANAGER_H

#include "glow/Backends/QueueBackedDeviceManager.h"
#include "glow/Runtime/StatsExporter.h"

#include <atomic>

namespace glow {
namespace runtime {

// A class controlling a single NMP thread of execution driving the JIT
// backend. Many NMPFunctions may be added, but only one inference is executed
// at a time.
class NMPDeviceManager : public QueueBackedDeviceManager {
  // Compiled function list by name.
  FunctionMapTy functions_;

  // String constant for logging number of in-use devices.
  static constexpr const char *kDevicesUsedNMP = "glow.devices_used.nmp";

public:
  explicit NMPDeviceManager(const DeviceConfig &config)
      : QueueBackedDeviceManager(config) {
    statsExporterRegistry_->incrementCounter(kDevicesUsedNMP);
    exportMemoryCounters();
  }

  ~NMPDeviceManager() override {
    statsExporterRegistry_->incrementCounter(kDevicesUsedNMP, -1);
    zeroMemoryCounters();
  }

  // Returns the amount of memory in bytes available on the device when no
  // models are loaded.
  uint64_t getMaximumMemory() const override;

  // Returns the amount of memory in bytes currently availbe on the device.4
  uint64_t getAvailableMemory() const override;

  // Returns true if a function requiring the estimate size will fit on the
  // device. This is not a promise as memory cost could vary due to alignment,
  // etc.
  bool isMemoryAvailable(uint64_t estimate) const override;

  // Returns the DeviceInfo for this device containing peak limits for
  // compute and bandwidths (used in partitioning).
  DeviceInfo getDeviceInfo() const override;

protected:
  void addNetworkImpl(const Module *module, FunctionMapTy functions,
                      ReadyCBTy cb) override;
  void evictNetworkImpl(std::string functionName,
                        EvictFunctionCBTy evictCb) override;
  void runFunctionImpl(runtime::RunIdentifierTy id, std::string functionName,
                       std::unique_ptr<ExecutionContext> context,
                       ResultCBTy cb) override;
};

DeviceManager *createNMPDeviceManager(const DeviceConfig &config);

} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_NMP_NMPDEVICEMANAGER_H
