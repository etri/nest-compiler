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

#include "EnlightFunction.h"

#include "glow/IR/IR.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/ThreadPool.h"

#include "llvm/Support/Casting.h"

using namespace glow;

EnlightFunction::EnlightFunction(std::unique_ptr<IRFunction> F,
                                         runtime::RuntimeBundle &&bundle)
        : CompiledFunction(std::move(bundle)), F_(std::move(F)) {}

EnlightFunction::~EnlightFunction() {
    for (const auto &p : constants_) {
        delete p.second;
    }
    constants_.clear();
}

void EnlightFunction::collectConstants(const Module *module) {
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

void EnlightFunction::addConstant(std::string name, Tensor *T) {
    Tensor *newTensor = new Tensor;
    newTensor->assign(T);
    constants_[name] = newTensor;
}

Error EnlightFunction::execute(ExecutionContext *context) {
    BoundEnlightFunction boundFunc(constants_);
    boundFunc.setIRInstructionProcessingHandler(
            getIRInstructionProcessingHandler());
    auto res = boundFunc.execute(F_.get(), context);
    {
        TRACE_EVENT_SCOPE(context, TraceLevel::RUNTIME, "processInstrumentation");
        translateTraceEvents(context);
    }
    return res;
}

void EnlightFunction::translateTraceEvents(
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

BoundEnlightFunction::~BoundEnlightFunction() {
    // Delete the tensors that are owned by this backend.
    for (const auto &p : tensors_) {
        delete p.second;
    }
    tensors_.clear();
    externalTensors_.clear();
}

Tensor *BoundEnlightFunction::getTensor(const Value *v) const {
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

Tensor *BoundEnlightFunction::getOrCreateTensor(const Value *v) {
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

Tensor *BoundEnlightFunction::getOrCreateUnownedTensor(
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

void BoundEnlightFunction::deleteTensor(const Value *v) {
    auto it = tensors_.find(v);
    if (it == tensors_.end()) {
        return;
    }

    delete it->second;
    tensors_.erase(it);
}

Error BoundEnlightFunction::execute(IRFunction *F,
                                        ExecutionContext *context) {
    {
        TRACE_EVENT_SCOPE(context, TraceLevel::RUNTIME, "registerTensors");
        // Make sure all referenced tensors are on the host.
        context->getPlaceholderBindings()->ensureOnHost();
        /* yongin comment out
        // Find all virtually padded tensors so they can be replaced.
        std::vector<Placeholder *> virtualPadded;
        for (auto &ph : context->getPlaceholderBindings()->pairs()) {
            if (ph.second->getUnpaddedSizeInBytes() < ph.second->getSizeInBytes()) {
                virtualPadded.push_back(ph.first);
            }
        }
        // Replace all virtually padded tensors with real padding tensors.
        for (auto &ph : virtualPadded) {
            auto oldTensor = context->getPlaceholderBindings()->get(ph);
            Tensor paddedTensor(oldTensor->getType());
            memcpy(paddedTensor.getUnsafePtr(), oldTensor->getUnsafePtr(),
                   oldTensor->getUnpaddedSizeInBytes());
            context->getPlaceholderBindings()->erase(ph);
            context->getPlaceholderBindings()->insert(ph, std::move(paddedTensor));
        }*/
        // Register the concrete tensors that back the placeholder tensors.
        for (auto &ph : context->getPlaceholderBindings()->pairs()) {
            auto *w = F->getWeightForNode(ph.first);
            // If the Placeholder has been aliased to the same Weight, just skip it.
            if (externalTensors_.count(w)) {
                continue;
            }

            externalTensors_[w] = &ph.second;
        }
    }

    // Do the forward pass.
    auto &irInstructionProcessingHandler = getIRInstructionProcessingHandler();
    // Dispatch the vta on each instruction in the program.
    for (const auto &I : F->getInstrs()) {
        // Perform custom processing if needed and proceed with standard processing
        // if required.

        if (!irInstructionProcessingHandler ||
            !irInstructionProcessingHandler(
                    &I, IRInstructionProcessingStage::PROCESSING, this)) {
          //glow::dbgs()<<I.getName() <<"    : ";
          //glow::dbgs()<<I.getKindName() <<"\n";
            switch (I.getKind()) {

#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    fwd##CLASS(llvm::cast<CLASS>(&I));                                         \
    break;                                                                     \
  }
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)                                \
  case Kinded::Kind::CLASS##Kind: {                                            \
    fwd##CLASS(llvm::cast<CLASS>(&I));                                         \
    break;                                                                     \
  }


#ifndef DEF_INSTR
#error The macro DEF_INSTR was not declared.
#endif
#ifndef DEF_VALUE
#error The macro DEF_VALUE was not declared.
#endif
#ifndef DEF_BACKEND_SPECIFIC_INSTR
#error The macro DEF_BACKEND_SPECIFIC_INSTR was not declared.
#endif
#ifndef DEF_INSTR_RANGE
#define DEF_INSTR_RANGE(ID, FIRST, LAST)
#endif
DEF_INSTR(ConvolutionInst, convolution)
DEF_INSTR(FullyConnectedInst, fullyconnected)
DEF_INSTR(AllocActivationInst, allocactivation)
DEF_INSTR(DeallocActivationInst, deallocactivation)
#undef DEF_INSTR_RANGE
#undef DEF_INSTR
#undef DEF_BACKEND_SPECIFIC_INSTR
#undef DEF_VALUE
//#include "glow/AutoGenInstr.def"
                default:
                    glow::errs() << "Invalid instruction: " << &I << "\n";
                    llvm_unreachable("Invalid instruction.");
            }
        }

        // Perform post-processing of the instruction.
        if (irInstructionProcessingHandler) {
            irInstructionProcessingHandler(
                    &I, IRInstructionProcessingStage::POSTPROCESSING, this);
        }
    }

    {

        TRACE_EVENT_SCOPE(context, TraceLevel::RUNTIME, "eraseTensors");
        // Remove the concrete tensors that back the placeholder tensors.
        for (auto &ph : context->getPlaceholderBindings()->pairs()) {
            auto *w = F->getWeightForNode(ph.first);
            externalTensors_.erase(w);
        }
    }

    return Error::success();
}

