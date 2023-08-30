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

#include "Relay.h"
#include "RelayFunction.h"

#include "glow/Backend/BackendUtils.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Flags/Flags.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"

using namespace glow;

Expected<std::unique_ptr<CompiledFunction>>
Relay::compile(Function *F, const BackendOptions &opts) const {
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
Relay::compileIR(std::unique_ptr<IRFunction> IR) const {
  auto *mod = IR->getParent();
  auto function = compileIRWithoutConstants(std::move(IR));
  auto IFunction = static_cast<RelayFunction *>(function.get());
  IFunction->collectConstants(mod);
  return function;
}

std::unique_ptr<CompiledFunction>
Relay::compileIRWithoutConstants(std::unique_ptr<IRFunction> IR) const {
  MemoryAllocator constantWeightsAllocator("ConstantWeights", 0);
  MemoryAllocator placeholderWeightsAllocator("PlaceholderWeights", 0);
  MemoryAllocator activationsAllocator("Activations", 0);
  runtime::RuntimeBundle bundle = runtime::RuntimeBundle::create(
      *IR, constantWeightsAllocator, placeholderWeightsAllocator,
      activationsAllocator);
  auto compiledFunction =
      glow::make_unique<RelayFunction>(std::move(IR), std::move(bundle));
  compiledFunction->setIRInstructionProcessingHandler(
      getIRInstructionProcessingHandler());
  return compiledFunction;
}

bool Relay::isOpSupported(const NodeInfo &NI) const {
  switch (NI.getKind()) {
  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::SubNodeKind:
  case Kinded::Kind::MulNodeKind:
  case Kinded::Kind::DivNodeKind:
  case Kinded::Kind::ReluNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
         ElemKind::Int8QTy, ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::AvgPoolNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Int32ITy, ElemKind::FloatTy, ElemKind::Float16Ty,
         ElemKind::BFloat16Ty, ElemKind::Int8QTy});

  case Kinded::Kind::FullyConnectedNodeKind:
    if (!NI.getInTy(FullyConnectedNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty});
    }
    return (NI.allInputsAndOutputsHaveSameElemKind(
                {ElemKind::Int8QTy}, {FullyConnectedNode::BiasIdx}) &&
            (NI.getInElemTy(FullyConnectedNode::BiasIdx) == ElemKind::Int8QTy ||
             NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
                 ElemKind::Int32QTy ||
             NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
                 ElemKind::FloatTy)) ||
           (NI.allInputsAndOutputsHaveSameElemKind(
                {ElemKind::Int16QTy}, {FullyConnectedNode::BiasIdx}) &&
            (NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
                 ElemKind::Int16QTy ||
             NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
                 ElemKind::Int32QTy));

  case Kinded::Kind::MaxPoolNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
                ElemKind::Int8QTy},
               {}, {MaxPoolNode::ArgmaxIdx}) &&
           (NI.getOutElemTy(MaxPoolNode::ArgmaxIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::LocalResponseNormalizationNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty});

  case Kinded::Kind::TouchNodeKind:
  case Kinded::Kind::InsertTensorNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
         ElemKind::Int8QTy, ElemKind::Int16QTy, ElemKind::Int32ITy,
         ElemKind::Int64ITy, ElemKind::BoolTy});

  case Kinded::Kind::ConvolutionNodeKind:
    if (!NI.getInTy(ConvolutionNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty});
    }
    return (NI.allInputsAndOutputsHaveSameElemKind(
                {ElemKind::Int8QTy}, {ConvolutionNode::BiasIdx}) &&
            (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int8QTy ||
             NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy)) ||
           (NI.allInputsAndOutputsHaveSameElemKind(
                {ElemKind::Int16QTy}, {ConvolutionNode::BiasIdx}) &&
            (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int16QTy ||
             NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy));

  case Kinded::Kind::QuantizeNodeKind:
    return ((NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::FloatTy) ||
            (NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::Float16Ty) ||
            (NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::BFloat16Ty)) &&
           ((NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int8QTy) ||
            (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::UInt8QTy) ||
            (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int16QTy) ||
            (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int32QTy));

  case Kinded::Kind::DequantizeNodeKind:
    return ((NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int8QTy) ||
            (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::UInt8QTy) ||
            (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int16QTy) ||
            (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int32QTy) ||
            (NI.getInElemTy(DequantizeNode::InputIdx) ==
             ElemKind::UInt8FusedQTy)) &&
           ((NI.getOutElemTy(DequantizeNode::ResultIdx) == ElemKind::FloatTy) ||
            (NI.getOutElemTy(DequantizeNode::ResultIdx) ==
             ElemKind::Float16Ty) ||
            (NI.getOutElemTy(DequantizeNode::ResultIdx) ==
             ElemKind::BFloat16Ty));

  // We just clip 64 to 32 SelectedIdx silently with the SoftMax
  // SelectedIdx in case dim_t is 32b.
  case Kinded::Kind::SoftMaxNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty},
               {SoftMaxNode::SelectedIdx}) &&
           (NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int32ITy ||
            NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::TransposeNodeKind:
    // These work regardless of the underlying type.
    return true;

  case Kinded::Kind::SplatNodeKind:
      return true;
//      return false;

  default:
    return false;
  }
}

/// Use template meta-programming to check if typename ClassName contains
/// has_getLayout() method. Below generates a struct named has_getLayout that
/// looks for said method.
CLASS_CONTAINS_METHOD(getLayout)

template <typename T, std::enable_if_t<
                          !has_getLayout<T, ConvolutionLayout>::value, int> = 0>
static bool checkLayout(const T &I) {
  (void)I;
  return true;
}

template <typename T,
          std::enable_if_t<has_getLayout<T, ConvolutionLayout>::value, int> = 0>
static bool checkLayout(const T &I) {
  if (I.getLayout() != NHWC && I.getLayout() != NTHWC) {
    report("Relay supports only NHWC and NTHWC");
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
    DEF_NODE(Placeholder, Placeholder)
    DEF_NODE(ConvolutionNode, Convolution)
    DEF_NODE(MaxPoolNode, MaxPool)
    DEF_NODE(AvgPoolNode, AvgPool)
    DEF_NODE(FullyConnectedNode, FullyConnected)
    DEF_NODE(LocalResponseNormalizationNode, LocalResponseNormalization)
    DEF_NODE(SoftMaxNode, SoftMax)
    DEF_NODE(AddNode, Add)
    DEF_NODE(MulNode, Mul)
    DEF_NODE(SubNode, Sub)
    DEF_NODE(DivNode, Div)
    DEF_NODE(ReluNode, Relu)
    DEF_NODE(TransposeNode, Transpose)
    DEF_NODE(InsertTensorNode, InsertTensor)
    DEF_NODE(TouchNode, Touch)
    DEF_NODE(QuantizeNode, Quantize)
    DEF_NODE(DequantizeNode, Dequantize)
#undef DEF_NODE
  default:
    // llvm_unreachable("Invalid instruction.");
    break;
  }
  return true;
}

bool Relay::verify(const Function &F, bool verbose) const {
  if (!F.verify(this)) {
    return false;
  }
  // if (!checkAllNodesSupported(F, verbose)) {
  //   return false;
  // }
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
  switch (I.getKind()) {
    DEF_VALUE(WeightVar, weightvar)
    DEF_INSTR(AllocActivationInst, allocactivation)
    DEF_INSTR(TensorViewInst, tensorview)
    DEF_INSTR(DeallocActivationInst, deallocactivation)
    DEF_INSTR(CopyInst, copy)
    DEF_INSTR(ConvolutionInst, convolution)
    DEF_INSTR(MaxPoolInst, maxpool)
    DEF_INSTR(AvgPoolInst, avgpool)
    DEF_INSTR(FullyConnectedInst, fullyconnected)
    DEF_INSTR(LocalResponseNormalizationInst, localresponsenormalization)
    DEF_INSTR(SoftMaxInst, softmax)
    DEF_INSTR(ElementAddInst, elementadd)
    DEF_INSTR(ElementSubInst, elementsub)
    DEF_INSTR(ElementMulInst, elementmul)
    DEF_INSTR(ElementDivInst, elementdiv)
    DEF_INSTR(ReluInst, relu)
    DEF_INSTR(TransposeInst, transpose)
    DEF_INSTR(TouchInst, touch)
    DEF_INSTR(InsertTensorInst, inserttensor)
    DEF_INSTR(QuantizeInst, quantize)
    DEF_INSTR(DequantizeInst, dequantize)
#undef DEF_VALUE
#undef DEF_INSTR
  default:
    // llvm_unreachable("Invalid instruction.");
    break;
  }
  return true;
}

bool Relay::verify(const IRFunction &IR) const {
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

bool Relay::shouldLower(const Node *N) const {
  return !isOpSupported(NodeInfo(*N));
}

/// Quantize the float \p bias for the given FullyConnectedNode as int32 using
/// \p inputScale, weight \p scales and \p offset=0. \returns false if the bias
/// was already quantized and thus no change was made and true otherwise.
static bool quantizeFCFloatBias(Function *F,
                                FullyConnectedNode &fullyConnected) {
  if (fullyConnected.getBias().getType()->isQuantizedType() ||
      (!fullyConnected.getWeights().getType()->isQuantizedType())) {
    return false;
  }
  assert(fullyConnected.getBias().getElementType() == ElemKind::FloatTy &&
         "Bias type must be a float in order to quantize it.");
  Constant *biasC =
      llvm::dyn_cast<Constant>(fullyConnected.getBias().getNode());
  assert(biasC && "bias input to FullyConnectedNode must be Constant in order "
                  "to quantize the bias");
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

/// Quantize the float \p bias for the given RowwiseQuantizedFullyConnectedNode
/// as int32. \returns false if the bias was already quantized and thus no
/// change was made and true otherwise.
static bool quantizeRQFCFloatBias(Function *F,
                                  RowwiseQuantizedFullyConnectedNode &rqfc) {
  if (rqfc.getBias().getType()->isQuantizedType() ||
      (!rqfc.getWeights().getType()->isQuantizedType())) {
    return false;
  }
  assert(rqfc.getBias().getElementType() == ElemKind::FloatTy &&
         "Bias type must be a float in order to quantize it.");
  Constant *biasC = llvm::dyn_cast<Constant>(rqfc.getBias().getNode());
  assert(biasC && "bias input to RowwiseQuantizedFullyConnectedNode must be a "
                  "Constant in order to quantize the bias");

  auto TQPs = getTensorQuantizationParams(
      biasC->getPayload(), quantization::Schema::Asymmetric, ElemKind::Int32QTy,
      0, biasC->dims()[0]);

  DCHECK_EQ(TQPs.size(), 1) << "Should only be one dimension to quantize on";

  auto biasQuantizedT = quantization::quantizeTensor(
      biasC->getPayload(), TQPs[0], ElemKind::Int32QTy);

  auto biasQuantizedC = F->getParent()->createConstant(
      biasC->getName(), std::move(biasQuantizedT));

  Constant *weights = llvm::dyn_cast<Constant>(rqfc.getWeights().getNode());
  Constant *scales = llvm::dyn_cast<Constant>(rqfc.getScales().getNode());
  Constant *offsets = llvm::dyn_cast<Constant>(rqfc.getOffsets().getNode());

  DCHECK(weights);
  DCHECK(scales);
  DCHECK(offsets);

  auto newRQFC = F->createRowwiseQuantizedFullyConnected(
      rqfc.getName(), rqfc.getInput(), weights, scales, offsets, biasQuantizedC,
      rqfc.getResult().getType());
  rqfc.getResult().replaceAllUsesOfWith(newRQFC);
  return true;
}

/// This function performs the channelwise quantization for the bias operand of
/// a ChannelwiseQuantizedConvolutionNode \p channelwiseConv from function \p F.
/// The quantization is done only if the bias is float. \returns false if the
/// bias was already quantized and thus no change was made and true otherwise.
static bool channelwiseQuantizeFloatBias(
    Function *F, ChannelwiseQuantizedConvolutionNode &channelwiseConv) {

  // If bias is already quantized then quit.
  if (channelwiseConv.getBias().getType()->isQuantizedType()) {
    return false;
  }

  DCHECK(channelwiseConv.getBias().getElementType() == ElemKind::FloatTy)
      << "Bias type must be a float in order to quantize it!";

  Constant *biasC =
      llvm::dyn_cast<Constant>(channelwiseConv.getBias().getNode());
  DCHECK(biasC)
      << "Bias input to ChannelwiseQuantizedConvolutionNode must be a Constant "
         "in order to quantize the bias!";

  Constant *filterScalesC =
      llvm::dyn_cast<Constant>(channelwiseConv.getFilterScales().getNode());
  DCHECK(filterScalesC)
      << "Filter scales input to ChannelwiseQuantizedConvolutionNode must be a "
         "Constant in order to quantize the bias!";

  // Create new constants for Bias, BiasScales and BiasOffsets operands.
  Constant *biasCQ = F->getParent()->createConstant(
      ElemKind::Int32QTy, biasC->getType()->dims(), 1.0, 0, biasC->getName());
  Constant *biasScalesC = F->getParent()->createConstant(
      ElemKind::FloatTy, biasC->getType()->dims(), "biasScales");
  Constant *biasOffsetsC = F->getParent()->createConstant(
      ElemKind::Int32ITy, biasC->getType()->dims(), "biasOffsets");

  // Quantize the bias operand manually from FloatTy to Int32QTy using the
  // quantization parameters biasScales[i] = inputScale * filterScales[i] and
  // biasOffsets[i] = 0.
  float inputScale = channelwiseConv.getInput().getType()->getScale();
  const auto &filterScalesH = filterScalesC->getPayload().getHandle<float>();
  const auto &biasH = biasC->getPayload().getHandle<float>();
  auto biasQH = biasCQ->getPayload().getHandle<int32_t>();
  auto biasScalesH = biasScalesC->getPayload().getHandle<float>();
  auto biasOffsetsH = biasOffsetsC->getPayload().getHandle<int32_t>();
  for (dim_t idx = 0, idxEnd = biasC->getType()->size(); idx < idxEnd; ++idx) {
    TensorQuantizationParams biasTQP;
    biasTQP.scale = inputScale * filterScalesH.raw(idx);
    biasTQP.offset = 0;
    biasQH.raw(idx) = quantization::quantize<int32_t>(biasH.raw(idx), biasTQP);
    biasScalesH.raw(idx) = biasTQP.scale;
    biasOffsetsH.raw(idx) = biasTQP.offset;
  }

  // Create new ChannelwiseQuantizedConvolutionNode with quantized bias
  // and explicit bias scales and offsets.
  auto *newChannelwiseConv = F->createChannelwiseQuantizedConv(
      channelwiseConv.getName(), channelwiseConv.getInput(),
      channelwiseConv.getFilter(), biasCQ, channelwiseConv.getFilterScales(),
      channelwiseConv.getFilterOffsets(), biasScalesC, biasOffsetsC,
      channelwiseConv.getResult().getType(), channelwiseConv.getKernels(),
      channelwiseConv.getStrides(), channelwiseConv.getPads(),
      channelwiseConv.getGroup(), channelwiseConv.getDilation());
  channelwiseConv.getResult().replaceAllUsesOfWith(newChannelwiseConv);
  return true;
}

Expected<bool>
Relay::transformPostLowering(Function *F, CompilationContext &cctx,
                             const glow::runtime::DeviceInfo *devInfo) const {
  LOG_SCOPE(F->getLogContext(), "Relay::transformPostLowering")

  bool changed = false;
  for (auto &node : F->getNodes()) {
    if (auto *channelwiseConv =
            llvm::dyn_cast<ChannelwiseQuantizedConvolutionNode>(&node)) {
      changed |= channelwiseQuantizeFloatBias(F, *channelwiseConv);
    } else if (auto *fullyConnected =
                   llvm::dyn_cast<FullyConnectedNode>(&node)) {
      changed |= quantizeFCFloatBias(F, *fullyConnected);
    } else if (auto *rowwiseFC =
                   llvm::dyn_cast<RowwiseQuantizedFullyConnectedNode>(&node)) {
      changed |= quantizeRQFCFloatBias(F, *rowwiseFC);
    }
  }
  return changed;
}

void Relay::parseBackendSpecificOptions(const BackendOptions &opts) const {}

Expected<double> Relay::estimateNodeCost(const glow::Node *node) const {
  // Using default cost from Partitioner which is 1.
  return 1.0;
}
