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
 */
#include "VTADeviceManager.h"
#include "VTA.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace glow {
    namespace runtime {

        unsigned GlowVTAMemory = 0;
        static llvm::cl::OptionCategory
                VTABackendCat("Glow VTA Backend Options");
        static llvm::cl::opt<unsigned, /* ExternalStorage */ true> vtaMaxMemOpt(
                "vta-memory",
                llvm::cl::desc("VTA DeviceManager maximum memory in kilobytes"),
                llvm::cl::location(GlowVTAMemory),
                llvm::cl::cat(VTABackendCat));

        DeviceManager *createVTADeviceManager(const DeviceConfig &config) {
            if (GlowVTAMemory) {
                // Convert command line GlowVTAMemory to bytes from kilobytes.
                auto configNew = config;
                configNew.setDeviceMemory(uint64_t{GlowVTAMemory} * 1024);
                return new VTADeviceManager(configNew);
            }
            return new VTADeviceManager(config);
        }

        uint64_t VTADeviceManager::getMaximumMemory() const {
            return maxMemoryBytes_;
        }

        uint64_t VTADeviceManager::getAvailableMemory() const {
            return maxMemoryBytes_ - usedMemoryBytes_;
        }

        bool VTADeviceManager::isMemoryAvailable(uint64_t estimate) const {
            return maxMemoryBytes_ >= (usedMemoryBytes_ + estimate);
        }

        DeviceInfo VTADeviceManager::getDeviceInfo() const {
            // TODO: these may need to be tweaked depending on vta overheads.
            DeviceInfo info = DeviceInfo();
            info.sramCapacity = 256 * 1024 * 1024;
            info.peakCompute = 2.2 * 1024 * 1024 * 1024 * 1024;
            info.peakDramBw = 110.0 * 1024 * 1024 * 1024;
            info.peakSramBw = 1024.0 * 1024 * 1024 * 1024;
            info.peakPCIeBw = 16.0 * 1024 * 1024 * 1024;
            return info;
        }

        void VTADeviceManager::addNetworkImpl(const Module *module,
                                                      FunctionMapTy functions,
                                                      ReadyCBTy readyCB) {
            DCHECK(readyCB != nullptr);

            uint64_t allFunctionsMemoryBytes{0};

            // First check for uniqueness of the function name.
            for (const auto &func : functions) {
                if (functions_.count(func.first) != 0) {
                    readyCB(
                            module,
                            MAKE_ERR(
                                    llvm::formatv(
                                            "Failed to add network: already have a function called {0}",
                                            func.first)
                                            .str()));
                    return;
                }

                if (func.second->getCompileBackendName() != VTA::getName()) {
                    readyCB(module, MAKE_ERR(llvm::formatv("Failed to add network: function "
                                                           "{0} is not a VTAFunction",
                                                           func.first)
                                                     .str()));
                    return;
                }

                allFunctionsMemoryBytes +=
                        func.second->getRuntimeBundle().getConstantWeightSize();

                // Add function name to map for static placeholders.
                VTAFunction *function =
                        static_cast<VTAFunction *>(func.second);
                for (auto PH : function->getIR()->getGraph()->findPlaceholders()) {
                    if (PH->isStatic()) {
                        staticPlaceholderToFunctions_[PH].push_back(func.first);
                    }
                }
            }

            if (usedMemoryBytes_ + allFunctionsMemoryBytes > maxMemoryBytes_) {
                readyCB(module,
                        MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_OUT_OF_DEVICE_MEMORY,
                                 "Failed to add network: not enough memory"));
                return;
            }

            // Add to the function name lookup map.
            for (const auto &func : functions) {
                if (func.second->getRuntimeBundle().getConstants() == nullptr) {
                    func.second->collectConstants(module);
                }
                functions_.emplace(func.first, func.second);
            }

            usedMemoryBytes_ += allFunctionsMemoryBytes;
            assert(usedMemoryBytes_ <= maxMemoryBytes_);

            // Export changes to memory use.
            exportMemoryCounters();
            // Fire the ready CB.
            readyCB(module, Error::success());
        }

        void VTADeviceManager::transferStaticPlaceholderToDevice(
                Placeholder *PH, Tensor *T, std::function<void(Error)> resultCB) {
            auto it = staticPlaceholderToFunctions_.find(PH);
            if (it == staticPlaceholderToFunctions_.end()) {
                resultCB(MAKE_ERR(
                                 ErrorValue::ErrorCode::RUNTIME_ERROR,
                                 llvm::formatv("Unable to transfer PH: {0}", PH->getName()).str()));
                return;
            }
            for (auto functionName : it->second) {
                VTAFunction *func =
                        static_cast<VTAFunction *>(functions_[functionName]);
                func->addConstant(PH->getName(), T);
            }
            resultCB(Error::success());
        }

        void VTADeviceManager::evictNetworkImpl(std::string functionName,
                                                        EvictFunctionCBTy evictCB) {
            DCHECK(evictCB != nullptr);

            auto it = functions_.find(functionName);

            if (it != functions_.end()) {
                usedMemoryBytes_ -= it->second->getRuntimeBundle().getConstantWeightSize();
                functions_.erase(it);
            } else {
                evictCB(functionName,
                        MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_NOT_FOUND,
                                 strFormat("Could not find function with name %s to evict",
                                           functionName.c_str())));
                return;
            }
            exportMemoryCounters();
            evictCB(functionName, Error::success());
        }

        void VTADeviceManager::runFunctionImpl(
                RunIdentifierTy id, std::string function,
                std::unique_ptr<ExecutionContext> context, ResultCBTy resultCB) {
            DCHECK(resultCB != nullptr);

            TRACE_EVENT_SCOPE_NAMED(context->getTraceContext(), TraceLevel::RUNTIME,
                                    "DeviceManager::run", dmRun);
            if (context->getTraceContext()) {
                context->getTraceContext()->setThreadName("VTA");
            }
            auto funcIt = functions_.find(function);
            if (funcIt == functions_.end()) {
                dmRun.addArg("reason", "function not found");
                TRACE_EVENT_SCOPE_END_NAMED(dmRun);
                resultCB(id,
                         MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_NOT_FOUND,
                                  llvm::formatv("Function {0} not found", function).str()),
                         std::move(context));
                return;
            }

            CompiledFunction *func = funcIt->second;

            // Run that function.
            auto executeErr = func->execute(context.get());

            // End the TraceEvent early to avoid time in the CB.
            TRACE_EVENT_SCOPE_END_NAMED(dmRun);

            // Fire the resultCB.
            resultCB(id, std::move(executeErr), std::move(context));
        }

    } // namespace runtime
} // namespace glow