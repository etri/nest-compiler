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
 * @file: Transforms.cpp
 * @brief description: Do graph transformantions specific to NMP.
 * @date: 11 18, 2021
 */

#include "NMPBackend.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Backends/LayoutConverter.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "llvm/Support/Casting.h"

using llvm::dyn_cast;
using namespace glow;

/// Try to optimize the regular FullyConnectedNode into a target-specific
/// FullyConnectedNode with a different weight memory layout (row-wise).
/// This optimization adds a new kind of NMP-specific FullyConnectedNode
/// that operates on weight data in a non-standard format. The default
/// format is WH. This optimization changes the data layout to HW to make
/// the access pattern be more efficient.
template <typename T>
void WeightTranspose(Constant *wSrc, Constant *wDest,
                     std::vector<dim_t> shape) {
  auto dims = wSrc->getPayload().dims();
  assert(dims.size() == 2 && "Invalid weight dimmensions");

  auto hSrc = wSrc->getPayload().getHandle<T>();
  auto hDest = wDest->getPayload().getHandle<T>();
  
  // If shape.size > 2 then we need to reordering the tensor
  if (shape.size() > 2) {
    assert(dims[0] == shape[0] * shape[1] * shape[2] &&
           "Invalid weight dimmensions");
    // copy the tensor reordering to the original layout
    Tensor orgTensor = wSrc->getPayload().clone();
    auto hAux = orgTensor.getHandle<T>();
    unsigned step = shape[2];
    unsigned row = 0;
    for (dim_t c0 = 0; c0 < step; c0++) {
      for (dim_t c1 = c0; c1 <= dims[0] - step + c0; c1 += step) {
        for (dim_t c2 = 0; c2 < dims[1]; c2++) {
          hAux.at({row, c2}) = hSrc.at({c1, c2});
        }
        row++;
      }
    }
    // copy the transpose of tensor to dest
    for (dim_t c0 = 0; c0 < dims[0]; c0++)
      for (dim_t c1 = 0; c1 < dims[1]; c1++)
        hDest.at({c1, c0}) = hAux.at({c0, c1});
  }
  else {
    // copy the transpose of tensor to dest
    for (dim_t c0 = 0; c0 < dims[0]; c0++)
      for (dim_t c1 = 0; c1 < dims[1]; c1++)
        hDest.at({c1, c0}) = hSrc.at({c0, c1});

  }
}

static Node *transformFCNtoNMPversion(FullyConnectedNode *FCN, Function *F) {
  // get original shape of the tensor (without first one)
  // The product must be same of first dimmension of the weight
  std::vector<dim_t> shape;
  if (auto *RS = dyn_cast<ReshapeNode>(FCN->getInput())) {
    auto inpd = RS->getInput().dims();
    for (unsigned i = 1; i < inpd.size(); ++i) {
      shape.emplace_back(inpd[i]);
    }
  }
  Constant *weights = dyn_cast<Constant>(FCN->getWeights());
  // Check if we can transpose the weights.
  if (!weights || weights->getNumUsers() != 1) {
    return nullptr;
  }

  // We only support quantized types.
  auto ElemTy = weights->getElementType();
  if (!(ElemTy == ElemKind::Int8QTy || ElemTy == ElemKind::Int16QTy)) {
    return nullptr;
  }

  // Create a new constant weights with the transposed layout:
  TypeRef wType = weights->getType();
  auto WScale = wType->getScale();
  auto WOffset = wType->getOffset();
  auto dims = wType->dims();
  assert(dims.size() == 2 && "Invalid weights dimmensions");

  // Create a new constant
  auto *M = F->getParent();
  auto *weightsT =
      M->createConstant(wType->getElementType(), {dims[1], dims[0]}, WScale,
                        WOffset, weights->getName().str() + "_transposed");

  // Transpose the weights into the new layout
  if (ElemTy == ElemKind::Int8QTy) {
    WeightTranspose<int8_t>(weights, weightsT, shape);
  } else {
    WeightTranspose<int16_t>(weights, weightsT, shape);
  }

  return F->addNode(new NMPFullyConnectedNode(
      FCN->getName(), FCN->getResult().getType(), FCN->getInput(), weightsT,
      FCN->getBias(), glow::FusedActivation::NONE, {}));
}

// Perform NMP specific post-lowering graph transformation.
Expected<bool>
NMPBackend::transformPostLowering(Function *F, CompilationContext &cctx,
                                  const glow::runtime::DeviceInfo *devInfo) const {
  LOG_SCOPE(F->getLogContext(), "NMPBackend::transformPostLowering")

  bool changed = false;
  for (auto &node : F->getNodes()) {
    if (auto *CN = dyn_cast<ConvolutionNode>(&node)) {
      if (cctx.compMode == CompilationMode::Train) {
        continue;
      }

      if (CN->getLayout() == NCHW) {
        continue;
      }

      auto *NR = convertConvToNCHWConv(CN, F);
      CN->getResult().replaceAllUsesOfWith(NR);
      changed = true;
      continue;
    }

    if (auto *PMN = dyn_cast<MaxPoolNode>(&node)) {
      if (cctx.compMode == CompilationMode::Train) {
        continue;
      }

      if (PMN->getLayout() == NCHW) {
        continue;
      }

      // We need to replace both of the MaxPool results with their NCHW
      // counterpart in order to get rid of the old node.
      if (PMN->getArgmax().getNumUsers() > 0) {
        continue;
      }

      auto results = convertMaxPoolToNCHWPool(PMN, F);
      PMN->getResult().replaceAllUsesOfWith(results.first);
      changed = true;
      continue;
    }

    if (auto *PAN = dyn_cast<AvgPoolNode>(&node)) {
      if (cctx.compMode == CompilationMode::Train) {
        continue;
      }

      if (PAN->getLayout() == NCHW) {
        continue;
      }

      auto *NR = convertAvgPoolToNCHWPool(PAN, F);
      PAN->getResult().replaceAllUsesOfWith(NR);
      changed = true;
      continue;
    }

    if (auto *FCN = dyn_cast<FullyConnectedNode>(&node)) {
      if (cctx.compMode == CompilationMode::Train) {
        continue;
      }

      // Try to replace generic FullyConnectedNode with NMP-version.
      if (Node *NFC = transformFCNtoNMPversion(FCN, F)) {
        FCN->getResult().typeUnsafeReplaceAllUsesOfWith(NFC);
        changed = true;
        continue;
      }
    }
  }

  if (cctx.compMode != CompilationMode::Train) {
    for (auto &node : F->getNodes()) {
      if (auto *RN = dyn_cast<ReluNode>(&node)) {
        // Try to fuse Relu up to NMP FC op
        if (auto *FC = dyn_cast<NMPFullyConnectedNode>(RN->getInput())) {
          if (FC->getNumUsers() != 1) {
            continue;
          }
          FC->setFusedActivation(FusedActivation::RELU);
          FC->getResult().typeUnsafeReplaceAllUsesOfWith(RN->getResult(), F,
                                                         RN);
          // We can get rid of the relu.
          RN->getResult().typeUnsafeReplaceAllUsesOfWith(FC);
          changed = true;
          continue;
        }
      }
      if (auto *RS = dyn_cast<ReshapeNode>(&node)) {
        // Try to remove Transpose up to RS op
        if (auto *TN = dyn_cast<TransposeNode>(RS->getInput())) {
          if (TN->getNumUsers() != 1) {
            continue;
          }
          RS->getInput().typeUnsafeReplaceAllUsesOfWith(TN->getInput());
          changed = true;
          continue;
        }
      }
      if (auto *RQ = dyn_cast<RescaleQuantizedNode>(&node)) {
        // Just remove the Rescale operation. The scale of
        // input after glow remove all transpose operations
        // will be modified in transformPostOptPipeline function.
        TypeRef outTy = RQ->getResult().getType();
        auto newTy =
          F->getParent()->uniqueType(outTy->getElementType(), outTy->dims(),
                                     outTy->getScale(), outTy->getOffset());
        // Get the previous node value
        auto *PN = RQ->getInput().getNode();
        // Set the result type to newTy
        PN->setType(0, newTy);
        PN->getNthResult(0).setType(newTy);
        // set the user of RQ result with PN result type
        RQ->getResult().typeUnsafeReplaceAllUsesOfWith(PN->getNthResult(0));
        changed = true;
        continue;
      }
      continue;
    }
  }

  return changed;
}
