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

#include "Newton.h"
#include "NewtonFunction.h"

#include "glow/Backend/BackendUtils.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"
#include <vector>
namespace glow {
namespace runtime {
extern unsigned GlowNewtonMemory;
}
} // namespace glow
using namespace glow;
using llvm::dyn_cast;

Expected<std::unique_ptr<CompiledFunction>>
Newton::compile(Function *F, const BackendOptions &opts) const {
  TraceInfo traceInfo = buildManualTraceInfo(F);
  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());

  if (!opts.backendSpecificOpts.empty()) {
    parseBackendSpecificOptions(opts);
  }

  if (opts.autoInstrument) {
    autoInstrument(traceInfo, IR.get());
  }

  std::unique_ptr<CompiledFunction> compiledFunc;
  if (opts.collectConstants) {
    compiledFunc = compileIR(std::move(IR));
  } else {
    compiledFunc = compileIRWithoutConstants(std::move(IR));
  }

  compiledFunc->setTraceInfo(std::move(traceInfo));
  return Expected<std::unique_ptr<CompiledFunction>>(std::move(compiledFunc));
}

std::unique_ptr<CompiledFunction>
Newton::compileIR(std::unique_ptr<IRFunction> IR) const {
  auto *mod = IR->getGraph()->getParent();
  auto function = compileIRWithoutConstants(std::move(IR));
  auto IFunction = static_cast<NewtonFunction *>(function.get());
  IFunction->collectConstants(mod);
  return function;
}

std::unique_ptr<CompiledFunction>
Newton::compileIRWithoutConstants(std::unique_ptr<IRFunction> IR) const {
  MemoryAllocator constantWeightsAllocator("ConstantWeights", 0);
  MemoryAllocator placeholderWeightsAllocator("PlaceholderWeights", 0);
  MemoryAllocator activationsAllocator("Activations", 0);
  runtime::RuntimeBundle bundle = runtime::RuntimeBundle::create(
      *IR, constantWeightsAllocator, placeholderWeightsAllocator,
      activationsAllocator);
  auto compiledFunction =
      glow::make_unique<NewtonFunction>(std::move(IR), std::move(bundle));
  compiledFunction->setIRInstructionProcessingHandler(
      getIRInstructionProcessingHandler());
  return compiledFunction;
}

bool Newton::isOpSupported(const NodeInfo &NI) const {
  switch (NI.getKind()) {
    case Kinded::Kind::FullyConnectedNodeKind:
      if (!NI.getInTy(ConvolutionNode::InputIdx)->isQuantizedType()) {
        return true;
      }
    case Kinded::Kind::SaveNodeKind:
      return true;

    default:return false;
  }
}

/// Use template meta-programming to check if typename ClassName contains
/// has_getLayout() method. Below generates a struct named has_getLayout that
/// looks for said method.
CLASS_CONTAINS_METHOD(getLayout)

template<typename T, std::enable_if_t<
    !has_getLayout<T, ConvolutionLayout>::value, int> = 0>
static bool checkLayout(const T &I) {
  (void) I;
  return true;
}

template<typename T,
    std::enable_if_t<has_getLayout<T, ConvolutionLayout>::value, int> = 0>
static bool checkLayout(const T &I) {
  if (I.getLayout() != NHWC) {
    report("Glow Newton supports only NHWC");
    return false;
  }
  return true;
}

static bool checkLayoutForNode(const Node &N) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case Kinded::Kind::CLASS##Kind: {                                            \
    const CLASS *CI = llvm::cast<CLASS>(&N);                                   \
    return checkLayout(*CI);                                                   \
    break;                                                                     \
  }
  switch (N.getKind()) {
#include "glow/AutoGenNodes.def"
    default:llvm_unreachable("Invalid instruction.");
  }
  return true;
}

bool Newton::verify(const Function &F, bool verbose) const {
  if (!F.verify(this)) {
    return false;
  }
  if (!checkAllNodesSupported(F, verbose)) {
    return false;
  }
  for (const Node &N : F.getNodes()) {
    if (!checkLayoutForNode(N)) {
      return false;
    }
    if (!(N.getKind() == Kinded::Kind::ConvolutionNodeKind &&
        llvm::cast<ConvolutionNode>(&N)->getFusedActivation() ==
            FusedActivation::RELU) &&
        !checkNoFusionForNode(N)) {
      return false;
    }
  }
  return true;
}

static bool checkLayoutForInstr(const Instruction &I) {
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    const CLASS *CI = llvm::cast<CLASS>(&I);                                   \
    return checkLayout(*CI);                                                   \
    break;                                                                     \
  }
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)
  switch (I.getKind()) {
#include "glow/AutoGenInstr.def"
    default:llvm_unreachable("Invalid instruction.");
  }
  return true;
}

bool Newton::verify(const IRFunction &IR) const {
  for (const auto &I : IR.getInstrs()) {
    // Only support convolution+relu fusions for now.
    if (!(I.getKind() == Kinded::Kind::ConvolutionInstKind &&
        llvm::cast<ConvolutionInst>(&I)->getFusedActivation() ==
            FusedActivation::RELU) &&
        !checkNoFusionForInstr(I)) {
      return false;
    }

    if (!checkLayoutForInstr(I)) {
      return false;
    }
  }
  return true;
}

bool Newton::shouldLower(const Node *N) const {
  switch (N->getKind()) {
    case Kinded::Kind::ConvolutionNodeKind:
    case Kinded::Kind::Convolution3DNodeKind:
    case Kinded::Kind::SparseLengthsSumNodeKind:
    case Kinded::Kind::FullyConnectedNodeKind:
    case Kinded::Kind::ReluNodeKind:return false;
    default:return true;
  }
}

/// Quantize the given float \p bias as int32 using \p inputScale,
/// weight \p scales and \p offset=0. \returns false if the bias was already
/// quantized and thus no change was made and true otherwise.
static bool quantizeFloatBias(Function *F, FullyConnectedNode &fullyConnected) {
  if (fullyConnected.getBias().getType()->isQuantizedType() ||
      (!fullyConnected.getWeights().getType()->isQuantizedType())) {
    return false;
  }
  assert(fullyConnected.getBias().getElementType() == ElemKind::FloatTy &&
      "Bias type must be a float in order to quantize it.");
  Constant *biasC =
      llvm::dyn_cast<Constant>(fullyConnected.getBias().getNode());
  assert(biasC && "bias input to ChannelwiseQuantizedConvolutionNode "
                  "must be a Constant in order to quantize the bias");
  const auto &biasUnquantizedH = biasC->getPayload().getHandle<float>();
  // biasQuantizedT is Int32QTy
  const float inputScale = fullyConnected.getInput().getType()->getScale();
  const float weigthScale = fullyConnected.getWeights().getType()->getScale();
  const float scale = inputScale * weigthScale;
  auto biasQuantizedT = Tensor(ElemKind::Int32QTy, biasUnquantizedH.dims(),
      /* scale */ scale, /* offset */ 0);
  auto biasQuantizedH = biasQuantizedT.getHandle<int32_t>();
  TensorQuantizationParams tqp;
  tqp.scale = scale;
  tqp.offset = 0;
  for (dim_t i = 0; i < biasQuantizedH.size(); ++i) {
    biasQuantizedH.raw(i) =
        quantization::quantize<int32_t>(biasUnquantizedH.raw(i), tqp);
  }
  auto biasQuantizedC = F->getParent()->createConstant(
      biasC->getName(), std::move(biasQuantizedT));
  auto newFullyConnectedNode = F->createFullyConnected(
      fullyConnected.getName(), fullyConnected.getInput(),
      fullyConnected.getWeights(), biasQuantizedC,
      fullyConnected.getResult().getType(), /* axis doens't matter */ 1);
  fullyConnected.getResult().replaceAllUsesOfWith(newFullyConnectedNode);
  return true;
}

/// Channelwise quantize the given float \p bias as int32 using \p inputScale,
/// per-channel \p scales and \p offsets. \returns false if the bias was already
/// quantized and thus no change was made and true otherwise.
static bool channelwiseQuantizeFloatBias(
    Function *F, ChannelwiseQuantizedConvolutionNode &channelwiseConv) {
  return false;
}

Expected<bool> Newton::transformPostLowering(
    Function *F, CompilationContext &cctx,
    const glow::runtime::DeviceInfo *devInfo) const {
  LOG_SCOPE(F->getLogContext(), "Newton::transformPostLowering")

  bool changed = false;
  for (auto &node : F->getNodes()) {
    if (auto *channelwiseConv =
        llvm::dyn_cast<ChannelwiseQuantizedConvolutionNode>(&node)) {
      changed |= channelwiseQuantizeFloatBias(F, *channelwiseConv);
    } else if (auto *fullyConnected =
        llvm::dyn_cast<FullyConnectedNode>(&node)) {
      changed |= quantizeFloatBias(F, *fullyConnected);
    }
  }
  return changed;
}

void Newton::parseBackendSpecificOptions(
    const BackendOptions &opts) const {
  auto NewtonMaxMemOpt =
      opts.backendSpecificOpts.find("Newton-memory");
  if (NewtonMaxMemOpt != opts.backendSpecificOpts.end()) {
    glow::runtime::GlowNewtonMemory =
        std::stoi(NewtonMaxMemOpt->second);
    llvm::outs() << "Newton memory set to "
                 << glow::runtime::GlowNewtonMemory << "\n";
  }
}
