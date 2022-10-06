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
 * @file: NMPTensorLayout.cpp
 * @brief description: Define the default layout for tensors ion the NMP model.
 * @date: 11 18, 2021
 */

#include "NMPTensorLayout.h"
#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"

using namespace glow;

/// Definitions of different tensor layouts.
static std::string nmpDimsNHWC[] = {
    {"N"},
    {"H"},
    {"W"},
    {"C"},
};
static std::string nmpDimsNCHW[] = {
    {"N"},
    {"C"},
    {"H"},
    {"W"},
};
static TensorLayoutDescription nmpLayoutNHWC(nmpDimsNHWC);
static TensorLayoutDescription nmpLayoutNCHW(nmpDimsNCHW);

static std::string returnBaseReqOrNHWC(TensorLayoutDescription &baseReq,
                                       const Node *node) {
  if (!baseReq.isSameLayout(
          CanonicalTensorLayout::getInstance().getLayoutsForDims()[4])) {
    return baseReq.getSerializedLayout();
  }
  if (CanonicalTensorLayout::getInstance().acceptsAnyLayout(node)) {
    // These nodes accept any 4-D layout.
    return baseReq.getSerializedLayout();
  }
  // For Placeholders and Constants that were loaded from another tool,
  // we don't have the layout information in during load time. Glow's
  // assume they are in Canonical NHWC format, which is not correct if
  // we are loading an image with NCHW layout as in resent loader. 
  if (llvm::dyn_cast<Storage>(node)) {
    return baseReq.getSerializedLayout();
  }

  return CanonicalTensorLayout::getInstance().getDefaultNDLayout(4);
}

/// returns NHWC or NCHW layout based on the instruction's layout enum.
template <typename N>
static const TensorLayoutDescription *getLayoutFromEnum(const N &node) {
  if (node->getLayout() == NCHW) {
    return &nmpLayoutNCHW;
  }
  return &nmpLayoutNHWC;
}

/// returns NHWC or NCHW layout based on the instruction's layout enum
static const TensorLayoutDescription *
getLayoutForTempEnumRep(size_t n, const Node *node) {
  if (const auto MP = llvm::dyn_cast<MaxPoolNode>(node)) {
    return getLayoutFromEnum(MP);
  }
  if (const auto AP = llvm::dyn_cast<AvgPoolNode>(node)) {
    return getLayoutFromEnum(AP);
  }
  if (const auto *CN = llvm::dyn_cast<ConvolutionNode>(node)) {
    switch (n) {
    case ConvolutionNode::InputIndices::BiasIdx:
      return &CanonicalTensorLayout::getInstance().getLayoutsForDims()[1];
    default:
      return getLayoutFromEnum(CN);
    }
  }
  return nullptr;
}

std::string NMPTensorLayout::getNthInputLayoutRequirements(const Node *node,
                                                              size_t n) {
  DCHECK_LT(n, node->getNumInputs()) << "Wrong input number";
  auto inputNode = node->getNthInput(n);
  auto dims = inputNode.getType()->dims();
  DCHECK_LE(dims.size(), max_tensor_dimensions) << "Too many dimensions";
  const auto *layout = getLayoutForTempEnumRep(n, node);
  if (layout) {
    return layout->getSerializedLayout();
  }
  auto baseReq = TensorLayoutCommon::getNthInputLayoutRequirements(node, n);
  auto baseReqHelper = TensorLayoutDescription(baseReq);
  return returnBaseReqOrNHWC(baseReqHelper, node);
}

std::string NMPTensorLayout::getNthResultLayoutRequirements(const Node *node,
                                                               size_t n) {
  DCHECK_LT(n, node->getNumResults()) << "Wrong output number";
  auto dims = node->getNthResult(n).getType()->dims();
  DCHECK_LE(dims.size(), max_tensor_dimensions) << "Too many dimensions";
  const auto *layout = getLayoutForTempEnumRep(n, node);
  if (layout) {
    return layout->getSerializedLayout();
  }
  auto baseReq = TensorLayoutCommon::getNthResultLayoutRequirements(node, n);
  auto baseReqHelper = TensorLayoutDescription(baseReq);
  return returnBaseReqOrNHWC(baseReqHelper, node);
}
