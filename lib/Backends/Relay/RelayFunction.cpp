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

#include "glow/Backends/Relay/RelayFunction.h"

#include "glow/IR/IR.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/ThreadPool.h"

#include "llvm/Support/Casting.h"

using namespace glow;

RelayFunction::RelayFunction(std::unique_ptr<IRFunction> F,
                                         runtime::RuntimeBundle &&bundle)
    : CompiledFunction(std::move(bundle)), F_(std::move(F)) {}

RelayFunction::~RelayFunction() {
  for (const auto &p : constants_) {
    delete p.second;
  }
  constants_.clear();
}

void RelayFunction::collectConstants(const Module *module) {
  runtimeBundle_.collectConstants(module);
  if (constants_.empty()) {
    if (runtimeBundle_.getConstantWeightSize()) {
      for (const auto &v : F_->findConstants()) {
        auto symbolInfo = runtimeBundle_.getSymbolInfo(v);
        auto addr = runtimeBundle_.getConstants() + symbolInfo.offset;
        auto tensor = new Tensor(addr, &symbolInfo.type);
        constants_.emplace(std::string(v->getName()), tensor);
      }
    }
  }
}

void RelayFunction::addConstant(std::string name, Tensor *T) {
  Tensor *newTensor = new Tensor;
  newTensor->assign(T);
  constants_[name] = newTensor;
}

Error RelayFunction::execute(ExecutionContext *context) {
  BoundRelayFunction boundFunc(constants_);
  boundFunc.setIRInstructionProcessingHandler(
      getIRInstructionProcessingHandler());
  auto res = boundFunc.execute(F_.get(), context);
  {
    TRACE_EVENT_SCOPE(context, TraceLevel::RUNTIME, "processInstrumentation");
    translateTraceEvents(context);
  }
  return res;
}

void RelayFunction::translateTraceEvents(
    ExecutionContext *context) const {
  auto &traceInfo = getTraceInfo();
  if (!traceInfo.enabled) {
    return;
  }

  TraceContext *traceContext = context->getTraceContext();

  if (!traceContext || !traceContext->shouldLog(TraceLevel::OPERATOR)) {
    return;
  }

  PlaceholderBindings *bindings = context->getPlaceholderBindings();

  int tid = threads::getThreadId();
  auto &traceEvents = traceContext->getTraceEvents();
  for (auto &backing : traceInfo.events) {
    Tensor *backingTensor = bindings->get(backing.first);
    assert(backingTensor);

    for (const TraceInfo::Event &event : backing.second) {
      // If it's a complete event: grab both timestamps.
      if (event.type == TraceEvent::CompleteType) {
        uint64_t start{0}, end{0};
        memcpy(&start,
               backingTensor->getUnsafePtr() +
                   (event.startIndex * traceInfo.dataSize),
               traceInfo.dataSize);
        memcpy(&end,
               backingTensor->getUnsafePtr() +
                   (event.endIndex * traceInfo.dataSize),
               traceInfo.dataSize);
        traceEvents.push_back({event.name,
                               TraceLevel::OPERATOR,
                               start,
                               end - start,
                               tid,
                               {{"kind", event.kind}}});
      } else {
        uint64_t ts{0};
        memcpy(&ts,
               backingTensor->getUnsafePtr() +
                   (event.startIndex * traceInfo.dataSize),
               traceInfo.dataSize);
        traceEvents.push_back({event.name,
                               TraceLevel::OPERATOR,
                               ts,
                               event.type,
                               tid,
                               {{"kind", event.kind}}});
      }
    }
  }
}

BoundRelayFunction::~BoundRelayFunction() {
  // Delete the tensors that are owned by this backend.
  for (const auto &p : tensors_) {
    delete p.second;
  }
  tensors_.clear();
  externalTensors_.clear();
}

Tensor *BoundRelayFunction::getTensor(const Value *v) const {
  auto it = tensors_.find(v);
  if (it != tensors_.end()) {
    return it->second;
  }
  auto ic = constants_.find(std::string(v->getName()));
  if (ic != constants_.end()) {
    return ic->second;
  }

  auto ie = externalTensors_.find(v);
  assert(ie != externalTensors_.end() && "Unknown key Value.");
  return ie->second;
}

Tensor *BoundRelayFunction::getOrCreateTensor(const Value *v) {
  auto ie = externalTensors_.find(v);
  if (ie != externalTensors_.end()) {
    return ie->second;
  }
  auto ic = constants_.find(std::string(v->getName()));
  if (ic != constants_.end()) {
    return ic->second;
  }

  // Pick the tensor.
  auto it = tensors_.find(v);
  if (it == tensors_.end()) {
    auto *T = new Tensor(v->getType());
    tensors_[v] = T;
    return T;
  }
  return it->second;
}

Tensor *BoundRelayFunction::getOrCreateUnownedTensor(
    const Value *v, const Value *src, llvm::ArrayRef<dim_t> offsets) {
  assert(llvm::isa<TensorViewInst>(v) && "Expected a tensor view");

  // Pick the tensor.
  auto it = tensors_.find(v);

  // Release unowned tensors before re-creating them.
  if (it != tensors_.end()) {
    deleteTensor(v);
  }

  auto *T = new Tensor();
  *T = getTensor(src)->getUnowned(v->dims(), offsets);
  tensors_[v] = T;
  return T;
}

void BoundRelayFunction::deleteTensor(const Value *v) {
  auto it = tensors_.find(v);
  if (it == tensors_.end()) {
    return;
  }

  delete it->second;
  tensors_.erase(it);
}

Error BoundRelayFunction::execute(IRFunction *F,
                                        ExecutionContext *context) {

  return Error::success();
}
