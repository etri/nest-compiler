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
 * @file: NMPBackend.cpp
 * @brief description: Implementation of the NMP Backend interface.
 * @date: 11 17, 2021
 */

#include <cstdio>

#include "NMPBackend.h"
#include "NMPFunction.h"
#include "NMPTensorLayout.h"

#include "glow/Backend/BackendUtils.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/Instrs.h"
#include "glow/LLVMIRCodeGen/LLVMIRGen.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Support/Debug.h"
#include "glow/Backends/NMP/CommandLine.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Host.h"

#include <numeric>

using llvm::dyn_cast;
using llvm::isa;
using namespace glow;

NMPBackend::NMPBackend() {
  /// If target is not explicitly given we use the riscv32 target attribute.
  auto &opts = getOptions();
  if (opts.getTarget().empty()) {
    opts.setTarget("riscv32-unknown-elf");
  }
}

/// We compile the standard library (libjit) to LLVM bitcode, and then convert
/// that binary data to an include file using an external utility (include-bin).
/// The resulting file is included here to compile the bitcode image into our
/// library.
static const unsigned char libjit_bc[] = {
#include "glow/libjit/libjit_nmp.inc"
};
static const size_t libjit_bc_size = sizeof(libjit_bc);

// We only support Int8 & Int16 quantized types
static bool isQuantizedType(ElemKind kind) {
  return kind == ElemKind::Int8QTy || kind == ElemKind::Int16QTy;
}

// returns whether the provided NI is supported by the NMP backend.
bool NMPBackend::isOpSupported(const NodeInfo &NI) const {
  if (NI.getKind() == Kinded::Kind::SaveNodeKind) {
    return true;
  }

  if (isQuantizedType(NI.getOutElemTy(0))) {
    switch (NI.getKind()) {
    case Kinded::Kind::AddNodeKind:
    case Kinded::Kind::AvgPoolNodeKind:
    case Kinded::Kind::ClipNodeKind:
    case Kinded::Kind::ConcatNodeKind:
    case Kinded::Kind::ConvolutionNodeKind:
    case Kinded::Kind::DivNodeKind:
    case Kinded::Kind::FullyConnectedNodeKind:
    case Kinded::Kind::NMPFullyConnectedNodeKind:
    case Kinded::Kind::MatMulNodeKind:
    case Kinded::Kind::MaxNodeKind:
    case Kinded::Kind::MaxPoolNodeKind:
    case Kinded::Kind::MinNodeKind:
    case Kinded::Kind::MulNodeKind:
    case Kinded::Kind::QuantizeNodeKind:
    case Kinded::Kind::ReluNodeKind:
    case Kinded::Kind::ReshapeNodeKind:
    case Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind:
    case Kinded::Kind::SliceNodeKind:
    case Kinded::Kind::SoftMaxNodeKind:
    case Kinded::Kind::SplatNodeKind:
    case Kinded::Kind::SubNodeKind:
    case Kinded::Kind::TileNodeKind:
    case Kinded::Kind::TransposeNodeKind:
      return true;
    case Kinded::Kind::RescaleQuantizedNodeKind:
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::Int8QTy, ElemKind::Int16QTy});
    default:
      return false;
    }
  }

  switch (NI.getKind()) {
  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::AvgPoolNodeKind:
  case Kinded::Kind::BatchMatMulNodeKind:
  case Kinded::Kind::BatchedAddNodeKind:
  case Kinded::Kind::BatchedReduceAddNodeKind:
  case Kinded::Kind::ClipNodeKind:
  case Kinded::Kind::ConcatNodeKind:
  case Kinded::Kind::DequantizeNodeKind:
  case Kinded::Kind::DivNodeKind:
  case Kinded::Kind::FullyConnectedNodeKind:
  case Kinded::Kind::NMPFullyConnectedNodeKind:
  case Kinded::Kind::MatMulNodeKind:
  case Kinded::Kind::MaxNodeKind:
  case Kinded::Kind::MaxPoolNodeKind:
  case Kinded::Kind::MinNodeKind:
  case Kinded::Kind::MulNodeKind:
  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::ReshapeNodeKind:
  case Kinded::Kind::SaveNodeKind:
  case Kinded::Kind::SliceNodeKind:
  case Kinded::Kind::SoftMaxNodeKind:
  case Kinded::Kind::SplatNodeKind:
  case Kinded::Kind::SubNodeKind:
  case Kinded::Kind::TransposeNodeKind:
  case Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind:
  case Kinded::Kind::LocalResponseNormalizationNodeKind:
    return true;
  default:
    return false; // ? Can delegate to LLVMBackend::isOpSupported(NI);
  }
}

// Allow the NMP backend to prevent lowering for some Node *N
bool NMPBackend::shouldLower(const Node *N) const {
  switch (N->getKind()) {
  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::ClipNodeKind:
  case Kinded::Kind::LeakyReluNodeKind:
  case Kinded::Kind::FullyConnectedNodeKind:
  case Kinded::Kind::NMPFullyConnectedNodeKind:
  case Kinded::Kind::ConvolutionNodeKind:
    return false;
  default:
    return true;
  }
}

bool NMPBackend::supportsFusedActivation(Node *parent, Node *activation) const {
  // NMP backend only supports fusing activations into Convolution,
  // and NMPFullyConnected.
  if (!llvm::isa<ConvolutionNode>(parent) &&
      !llvm::isa<NMPFullyConnectedNode>(parent)) {
    return false;
  }

  // Only the following activations can be fused.
  switch (activation->getKind()) {
  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::LeakyReluNodeKind:
    return !disableFuseActvOpt;
  case Kinded::Kind::ClipNodeKind:
    return false; // In the future NMP will support fusing Clip activation?
  default:
    return false;
  }
}

// Method that creates a CompiledFunction.
// Requires the JIT & bundle to be used for compiling the function.
std::unique_ptr<CompiledFunction> NMPBackend::createCompiledFunction(
    std::unique_ptr<GlowJIT> JIT,
    runtime::RuntimeBundle &&runtimeBundle) const {
  return std::make_unique<NMPFunction>(std::move(JIT),
                                       std::move(runtimeBundle));
}

// Method that creates the NMP LLVM IR generator. This creates a backend
// that inherits from the CPU backend, while providing a specific version
// of the NMP LLVM IR generator derived from LLVMIRGen.
std::unique_ptr<LLVMIRGen>
NMPBackend::createIRGen(const IRFunction *IR,
                        AllocationsInfo &allocationsInfo) const {
  NMPLLVMIRGen *irgen = new NMPLLVMIRGen(
      IR, allocationsInfo, "", getLibjitBitcode(), getObjectRegistry());
  return std::unique_ptr<LLVMIRGen>(irgen);
}

// returns libjit bitcode for the NMP backend.
llvm::StringRef NMPBackend::getLibjitBitcode() const {
  return llvm::StringRef(reinterpret_cast<const char *>(libjit_bc),
                         libjit_bc_size);
}

// Rescale Arithmetic operations input params to ensure both
// LHS and RHS are at the same quantized scale
template <typename T> static void RescaleInputParams(Node &node, Function *F) {

  // Rescale the result of the NodeValue that is input to the arithmetic node.
  auto rescaleNodeValueType = [](Function *F, NodeValue &dest, NodeValue &src) {
    const TypeRef srcTy = src.getType();
    const TypeRef desTy = dest.getType();
    const TypeRef newTy =
        F->getParent()->uniqueType(desTy->getElementType(), desTy->dims(),
                                   srcTy->getScale(), srcTy->getOffset());
    dest.setType(newTy);
  };

  auto LHS = dyn_cast<T>(&node)->getLHS();
  auto RHS = dyn_cast<T>(&node)->getRHS();
  if (LHS.getScale() == RHS.getScale()) {
    return;
  }

  if (!LHS.hasOneUse() && RHS.hasOneUse())
    rescaleNodeValueType(F, RHS, LHS);
  else if (!RHS.hasOneUse() && LHS.hasOneUse())
    rescaleNodeValueType(F, LHS, RHS);
  else if (LHS.getScale() < RHS.getScale()) {
    rescaleNodeValueType(F, LHS, RHS);
  } else {
    rescaleNodeValueType(F, RHS, LHS);
  }
}

// Rescale Bias constants to ensure both Convolution/NMPFullyConnected
// results and Bias have the same quantized scale
template <typename T>
static void RescaleBias(Node &node, Function *F) {
  auto *N = dyn_cast<T>(&node);
  auto *bias = dyn_cast<Constant>(N->getBias());
  Tensor floatBias =
      quantization::dequantizeTensor(bias->getPayload(), ElemKind::FloatTy);
  auto resTy = N->getResult().getType();
  auto scale = resTy->getScale();
  auto offset = resTy->getOffset();
  auto elemTy = resTy->getElementType();
  Tensor newBias =
      quantization::quantizeTensor(floatBias, {scale, offset}, elemTy);
  Constant *newConst = F->getParent()->createConstant(
      bias->getName().str() + "_rescaled", std::move(newBias));
  const TypeRef biasTy = N->getBias().getType();
  const TypeRef newTy =
      F->getParent()->uniqueType(elemTy, biasTy->dims(), scale, offset);
  newConst->setType(0, newTy);
  bias->getOutput().typeUnsafeReplaceAllUsesOfWith(newConst->getOutput());
  bias->setType(0, newTy);
}

Expected<bool>
NMPBackend::transformPostOptPipeline(Function *F,
                                     CompilationContext &cctx) const {
  // Rescale Arithmetic operations input params to ensure both
  // LHS and RHS are at the same quantized scale
  // Also, rescale Bias to match Conv/FullyConnected Result scale
  for (auto &node : F->getNodes()) {
    switch (node.getKind()) {
    case Kinded::Kind::AddNodeKind:
      RescaleInputParams<AddNode>(node, F);
      break;
    case Kinded::Kind::SubNodeKind:
      RescaleInputParams<SubNode>(node, F);
      break;
    case Kinded::Kind::MulNodeKind:
      RescaleInputParams<MulNode>(node, F);
      break;
    case Kinded::Kind::DivNodeKind:
      RescaleInputParams<DivNode>(node, F);
      break;
    case Kinded::Kind::MaxNodeKind:
      RescaleInputParams<MaxNode>(node, F);
      break;
    case Kinded::Kind::MinNodeKind:
      RescaleInputParams<MinNode>(node, F);
      break;
    default:;
    }
    continue;
  }

  // Rescale Bias to match the scale of Conv/FullyConnected result.
  // This is done in two steps because Arithmetic operations can
  // change the scale of Conv/FullyConnected result operations.
  // Also, Try to remove RescaleQuantizedNodes, rescaling the
  // inputs to the same scale of the outputs
  for (auto &node : F->getNodes()) {
    switch (node.getKind()) {

    case Kinded::Kind::ConvolutionNodeKind: {
      auto *CN = dyn_cast<ConvolutionNode>(&node);
      if (rescaleBiasOpt)
        RescaleBias<ConvolutionNode>(node, F);
      break;
    }

    case Kinded::Kind::NMPFullyConnectedNodeKind: {
      auto *FC = dyn_cast<NMPFullyConnectedNode>(&node);
      if (rescaleBiasOpt)
        RescaleBias<NMPFullyConnectedNode>(node, F);
      break;
    }

    case Kinded::Kind::ConcatNodeKind: {
      // The RescaleQuantizedNode was inserted before ConcatNode
      // to ensure that the scales of inputs are same. I removed it
      // on transformPostLowering, so we need to fix then here, 
      // rescaling Inputs with same scale of Output.
      auto *CN = dyn_cast<ConcatNode>(&node);
      TypeRef outTy = CN->getResult().getType();
      for (unsigned i=0; i < CN->getNumInputs(); i++) {
        auto NV = CN->getNthInput(i);
        auto inTy = NV.getType();
        if (inTy->getScale() != outTy->getScale()) {
          auto newTy =
              F->getParent()->uniqueType(inTy->getElementType(), inTy->dims(),
                                         outTy->getScale(), outTy->getOffset());
          // Set the new type
          NV.setType(newTy);
          /* auto *N = NV.getNode(); */
          /* N->setType(0, newTy); */
          /* N->dump(); */
        }
      }
      break;
    }

    default:;
    }
    continue;
  }

  return true;
}

bool NMPBackend::verify(const Function &F, bool verbose) const { return true; }

glow::TensorLayoutCommon &NMPBackend::getTensorLayoutRequirements() const {
  return NMPTensorLayout::getInstance();
}

unsigned NMPBackend::numDevices() {
  return std::thread::hardware_concurrency();
}

std::vector<unsigned> NMPBackend::scanDeviceIDs() {
  std::vector<unsigned> deviceIDs(NMPBackend::numDevices());
  std::iota(std::begin(deviceIDs), std::end(deviceIDs), 0);
  return deviceIDs;
}

#include "nmpObjectRegistry.h"
llvm::ArrayRef<llvm::MemoryBufferRef> NMPBackend::getObjectRegistry() const {
  return nmpObjectRegistry;
}

void NMPBackend::save(Function *F, llvm::StringRef outputDir,
                      llvm::StringRef bundleName,
                      llvm::StringRef mainEntryName) const {
  llvm::SmallVector<std::string, 8> targetFeatures(llvmTargetFeatures.begin(),
                                                   llvmTargetFeatures.end());
  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());
  auto bundleSaver = createNMPBundleSaver(*this, outputDir, bundleName);
  bundleSaver->save(mainEntryName, IR.get());
  bundleSaver->produceBundle();
}

void NMPBackend::saveFunctions(llvm::ArrayRef<BundleEntry> entries,
                               llvm::StringRef outputDir,
                               llvm::StringRef bundleName) const {
  auto bundleSaver = createNMPBundleSaver(*this, outputDir, bundleName);
  std::vector<std::unique_ptr<glow::IRFunction>> irFunctions;
  for (auto &entry : entries) {
    auto IR = generateAndOptimizeIR(entry.func, *this, shouldShareBuffers());
    bundleSaver->save(entry.name, IR.get());
    irFunctions.emplace_back(std::move(IR));
  }
  bundleSaver->produceBundle();
}

std::unique_ptr<NMPBundleSaver>
NMPBackend::createNMPBundleSaver(const NMPBackend &nmpBackend,
                                 llvm::StringRef outputDir,
                                 llvm::StringRef bundleName) const {
  return glow::make_unique<NMPBundleSaver>(nmpBackend, outputDir, bundleName);
}
