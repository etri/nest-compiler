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

#include "VTA.h"
#include "VTABundle.h"
#include "glow/IR/Instrs.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Quantization/Base/Profile.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cmath>
#include <glow/Base/TensorSerialization.h>

using namespace glow;

#define dispatchImpl(functionName, elemTy, ...)                                \
  switch (elemTy) {                                                            \
  case ElemKind::FloatTy:                                                      \
    functionName<float>(__VA_ARGS__);                                          \
    break;                                                                     \
  case ElemKind::Float16Ty:                                                    \
    functionName<float16_t>(__VA_ARGS__);                                      \
    break;                                                                     \
  case ElemKind::Int8QTy:                                                      \
    functionName<int8_t>(__VA_ARGS__);                                         \
    break;                                                                     \
  case ElemKind::Int16QTy:                                                     \
    functionName<int16_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int32QTy:                                                     \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int32ITy:                                                     \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int64ITy:                                                     \
    functionName<int64_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::BoolTy:                                                       \
    functionName<bool>(__VA_ARGS__);                                           \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchFloatingPointImpl(functionName, elemTy, ...)                   \
  switch (elemTy) {                                                            \
  case ElemKind::FloatTy:                                                      \
    functionName<float>(__VA_ARGS__);                                          \
    break;                                                                     \
  case ElemKind::Float16Ty:                                                    \
    functionName<float16_t>(__VA_ARGS__);                                      \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchFloatingPointAndIndexImpl(functionName, elemTy, elemTyIndex,   \
                                          ...)                                 \
  switch (elemTy) {                                                            \
  case ElemKind::FloatTy:                                                      \
    if (elemTyIndex == ElemKind::Int64ITy) {                                   \
      functionName<float, int64_t>(__VA_ARGS__);                               \
    } else if (elemTyIndex == ElemKind::Int32ITy) {                            \
      functionName<float, int32_t>(__VA_ARGS__);                               \
    }                                                                          \
    break;                                                                     \
  case ElemKind::Float16Ty:                                                    \
    if (elemTyIndex == ElemKind::Int64ITy) {                                   \
      functionName<float16, int64_t>(__VA_ARGS__);                             \
    } else if (elemTyIndex == ElemKind::Int32ITy) {                            \
      functionName<float16, int32_t>(__VA_ARGS__);                             \
    }                                                                          \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchIndexTypeImpl(functionName, elemTy, ...)                       \
  switch (elemTy) {                                                            \
  case ElemKind::Int32ITy:                                                     \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int64ITy:                                                     \
    functionName<int64_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchArithmeticImpl(functionName, elemTy, ...)                      \
  switch (elemTy) {                                                            \
  case ElemKind::FloatTy:                                                      \
    functionName<float>(__VA_ARGS__);                                          \
    break;                                                                     \
  case ElemKind::Float16Ty:                                                    \
    functionName<float16_t>(__VA_ARGS__);                                      \
    break;                                                                     \
  case ElemKind::Int32ITy:                                                     \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int64ITy:                                                     \
    functionName<int64_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchQuantizedImpl(functionName, elemTy, ...)                       \
  switch (elemTy) {                                                            \
  case ElemKind::Int8QTy:                                                      \
    functionName<int8_t>(__VA_ARGS__);                                         \
    break;                                                                     \
  case ElemKind::Int16QTy:                                                     \
    functionName<int16_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int32QTy:                                                     \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchQuantizedWithAccumulationImpl(functionName, elemTy, ...)       \
  switch (elemTy) {                                                            \
  case ElemKind::Int8QTy:                                                      \
    functionName<int8_t, int32_t>(__VA_ARGS__);                                \
    break;                                                                     \
  case ElemKind::Int16QTy:                                                     \
    functionName<int16_t, int64_t>(__VA_ARGS__);                               \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define dispatchQuantizedWithAccumulationAndBiasImpl(functionName, elemTy,     \
                                                     biasElemType, ...)        \
  if (elemTy == ElemKind::Int8QTy && biasElemType == ElemKind::Int8QTy) {      \
    functionName<int8_t, int32_t, int8_t>(__VA_ARGS__);                        \
  } else if (elemTy == ElemKind::Int8QTy &&                                    \
             biasElemType == ElemKind::Int32QTy) {                             \
    functionName<int8_t, int32_t, int32_t>(__VA_ARGS__);                       \
  } else if (elemTy == ElemKind::Int16QTy &&                                   \
             biasElemType == ElemKind::Int16QTy) {                             \
    functionName<int16_t, int64_t, int16_t>(__VA_ARGS__);                      \
  } else if (elemTy == ElemKind::Int16QTy &&                                   \
             biasElemType == ElemKind::Int32QTy) {                             \
    functionName<int16_t, int64_t, int32_t>(__VA_ARGS__);                      \
  } else {                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

#define staticAssertFloatingPointType(ElemTy)                                  \
  static_assert(                                                               \
      std::is_floating_point<ElemTy>::value ||                                 \
          std::is_same<float16_t,                                              \
                       typename std::remove_cv<ElemTy>::type>::value,          \
      "This implementation is for floating-point values only")

#define staticAssertArithmeticType(ElemTy)                                     \
  static_assert(                                                               \
      std::is_arithmetic<ElemTy>::value ||                                     \
          std::is_same<float16_t,                                              \
                       typename std::remove_cv<ElemTy>::type>::value,          \
      "This implementation is for arithmetic values only")



//===----------------------------------------------------------------------===//
//                       Arithmetic operations
//===----------------------------------------------------------------------===//


void BoundVTAFunction::fwdElementAddInstI8Impl(
    const ElementAddInst *I) {
  assert(getTensor(I->getLHS())->getType().isQuantizedType() &&
      "Wrong function");
  auto lhsTy = I->getLHS()->getType();
  auto rhsTy = I->getRHS()->getType();
  auto destTy = I->getDest()->getType();

  float lhsScale = lhsTy->getScale();
  float rhsScale = rhsTy->getScale();
  float destScale = destTy->getScale();

  int32_t lhsOffset = lhsTy->getOffset();
  int32_t rhsOffset = rhsTy->getOffset();
  int32_t destOffset = destTy->getOffset();

  auto outW = getWeightHandle<int8_t>(I->getDest());
  auto lhsW = getWeightHandle<int8_t>(I->getLHS());
  auto rhsW = getWeightHandle<int8_t>(I->getRHS());
  for (dim_t i = 0, e = outW.size(); i < e; i++) {
    int32_t L = lhsW.raw(i);
    int32_t R = rhsW.raw(i);

    // We increase the size of the integer up to 16 bits to prevent overflow.
    const float largeScale = float(1) / (1 << 15);
    // Scale both sides from 8-bit to 16-bits.
    int32_t L32 = std::round(float(L - lhsOffset) * (lhsScale / largeScale));
    int32_t R32 = std::round(float(R - rhsOffset) * (rhsScale / largeScale));
    int32_t sum32 = L32 + R32;
    sum32 = std::round(float(sum32) * (largeScale / destScale) + destOffset);
    outW.raw(i) = quantization::clip<int32_t, int8_t>(sum32);
  }
}

void BoundVTAFunction::fwdElementAddInst(const ElementAddInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    fwdElementAddInstI8Impl(I);
    return;
  }

  llvm_unreachable("Not supported for VTA");
}


template <typename ElemTy>
void BoundVTAFunction::fwdElementSubInstArithmeticImpl(
    const ElementSubInst *I) {
  staticAssertArithmeticType(ElemTy);

  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto lhsW = getWeightHandle<ElemTy>(I->getLHS());
  auto rhsW = getWeightHandle<ElemTy>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = lhsW.raw(i) - rhsW.raw(i);
  }
}

void BoundVTAFunction::fwdElementSubInst(const ElementSubInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    auto destTy = I->getDest()->getType();
    auto lhsTy = I->getLHS()->getType();
    auto rhsTy = I->getRHS()->getType();

    float destScale = destTy->getScale();
    float lhsScale = lhsTy->getScale();
    float rhsScale = rhsTy->getScale();

    int32_t destOffset = destTy->getOffset();
    int32_t lhsOffset = lhsTy->getOffset();
    int32_t rhsOffset = rhsTy->getOffset();

    auto outW = getWeightHandle<int8_t>(I->getDest());
    auto lhsW = getWeightHandle<int8_t>(I->getLHS());
    auto rhsW = getWeightHandle<int8_t>(I->getRHS());
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      //    s_d * (i_d - o_d) = s_l * (i_l - o_l) - s_r * (i_r - o_r)
      // => i_d = (s_l / s_d) * (i_l - o_l) - (s_r / s_d) * (i_r - o_r) + o_d
      float l = (lhsScale / destScale) * float(lhsW.raw(i) - lhsOffset);
      float r = (rhsScale / destScale) * float(rhsW.raw(i) - rhsOffset);
      int32_t q = std::round(l - r + destOffset);
      outW.raw(i) = quantization::clip<int32_t, int8_t>(q);
    }
    return;
  }

  dispatchArithmeticImpl(fwdElementSubInstArithmeticImpl,
                         I->getDest()->getElementType(), I);
}

void BoundVTAFunction::fwdElementDivInst(const ElementDivInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    auto destTy = I->getDest()->getType();
    auto lhsTy = I->getLHS()->getType();
    auto rhsTy = I->getRHS()->getType();

    float destScale = destTy->getScale();
    float lhsScale = lhsTy->getScale();
    float rhsScale = rhsTy->getScale();

    int32_t destOffset = destTy->getOffset();
    int32_t lhsOffset = lhsTy->getOffset();
    int32_t rhsOffset = rhsTy->getOffset();

    auto outW = getWeightHandle<int8_t>(I->getDest());
    auto lhsW = getWeightHandle<int8_t>(I->getLHS());
    auto rhsW = getWeightHandle<int8_t>(I->getRHS());
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      //    s_d * (i_d - o_d) = (s_l * (i_l - o_l)) / (s_r * (i_r - o_r))
      // => i_d = (s_l * (i_l - o_l)) / (s_d * s_r * (i_r - o_r)) + o_d
      float l = lhsScale * float(lhsW.raw(i) - lhsOffset);
      float r = rhsScale * destScale * float(rhsW.raw(i) - rhsOffset);
      int32_t q = std::round(l / r + destOffset);
      outW.raw(i) = quantization::clip<int32_t, int8_t>(q);
    }
    return;
  }

#define DIV_LOOP(TYPE_)                                                        \
  auto outW = getWeightHandle<TYPE_>(I->getDest());                            \
  auto lhsW = getWeightHandle<TYPE_>(I->getLHS());                             \
  auto rhsW = getWeightHandle<TYPE_>(I->getRHS());                             \
  for (size_t i = 0, e = outW.size(); i < e; i++) {                            \
    outW.raw(i) = lhsW.raw(i) / rhsW.raw(i);                                   \
  }

  auto *T = getTensor(I->getDest());
  switch (T->getElementType()) {
  case ElemKind::Int64ITy: {
    DIV_LOOP(int64_t);
    return;
  }
  case ElemKind::FloatTy: {
    DIV_LOOP(float);
    return;
  }
  case ElemKind::Float16Ty: {
    DIV_LOOP(float16_t);
    return;
  }
  case ElemKind::BFloat16Ty: {
    DIV_LOOP(bfloat16_t);
    return;
  }
  default:
    llvm_unreachable("Unsupported type for Div.");
  }
}


void BoundVTAFunction::fwdElementMaxInstI8Impl(
    const ElementMaxInst *I) {
  assert(getTensor(I->getLHS())->getType().isQuantizedType() &&
      "Wrong function");
  auto lhsTy = I->getLHS()->getType();
  auto rhsTy = I->getRHS()->getType();
  auto destTy = I->getDest()->getType();

  TensorQuantizationParams lhsQ{lhsTy->getScale(), lhsTy->getOffset()};
  TensorQuantizationParams rhsQ{rhsTy->getScale(), rhsTy->getOffset()};
  TensorQuantizationParams destQ{destTy->getScale(), destTy->getOffset()};

  auto outW = getWeightHandle<int8_t>(I->getDest());
  auto lhsW = getWeightHandle<int8_t>(I->getLHS());
  auto rhsW = getWeightHandle<int8_t>(I->getRHS());
  for (size_t i = 0, e = outW.size(); i < e; i++) {
    // Convert both sides to the destination scale and perform a regular
    // comparison.
    int8_t L = quantization::quantize(
        quantization::dequantize(lhsW.raw(i), lhsQ), destQ);
    int8_t R = quantization::quantize(
        quantization::dequantize(rhsW.raw(i), rhsQ), destQ);
    outW.raw(i) = std::max(L, R);
  }
}


void BoundVTAFunction::fwdElementMaxInst(const ElementMaxInst *I) {
  if (getTensor(I->getLHS())->getType().isQuantizedType()) {
    fwdElementMaxInstI8Impl(I);
    return;
  }

  llvm_unreachable("Not supported for VTA");
}



//===----------------------------------------------------------------------===//
//                       Convolution
//===----------------------------------------------------------------------===//


/// This is the quantized implementation of Convolution.
template <typename ElemTy, typename AccumulatorTy, typename BiasElemTy>
void BoundVTAFunction::fwdConvolutionInstQuantizedImpl(
    Value *inV, Value *outV, Value *filterV, Value *biasV,
    llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
    llvm::ArrayRef<unsigned_t> pads, size_t group, size_t dilation, bool doRelu) {
  auto inW = getWeightHandle<ElemTy>(inV);
  auto outW = getWeightHandle<ElemTy>(outV);
  auto filterW = getWeightHandle<ElemTy>(filterV);
  auto biasW = getWeightHandle<BiasElemTy>(biasV);

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;

  PaddingTLBR pdim(pads);
  auto outTy = outV->getType();
  auto inTy = inV->getType();
  auto filterTy = filterV->getType();
  auto biasTy = biasV->getType();

  int32_t outOffset = outTy->getOffset();
  int32_t inOffset = inTy->getOffset();
  int32_t filterOffset = filterTy->getOffset();
  int32_t biasOffset = biasTy->getOffset();

  float outScale = outTy->getScale();
  float inScale = inTy->getScale();
  float filterScale = filterTy->getScale();
  float biasScale = biasTy->getScale();

  // Calculate the scale of the values that come out of the matrix
  // multiplication part of the calculation.
  float matMulScale = inScale * filterScale;

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {
    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

            // For each element in the convolution-filter:
            AccumulatorTy sum = 0;
            for (dim_t fx = 0; fx < kdim.height; fx++) {
              for (dim_t fy = 0; fy < kdim.width; fy++) {
                sdim_t ox = x + fx * dilation;
                sdim_t oy = y + fy * dilation;

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= sdim_t(idim.w)) {
                  continue;
                }
                for (dim_t fd = 0; fd < inCperG; fd++) {

                  AccumulatorTy F = filterW.at({d, fx, fy, fd});
                  AccumulatorTy I =
                      inW.at({n, (dim_t)ox, (dim_t)oy, g * inCperG + fd});
                  // We represent the element multiplication with offset as
                  // (value - offset).
                  sum += (F - filterOffset) * (I - inOffset);
                }
              }
            }

            // Scale the bias to match the scale of the matrix multiplication.
            AccumulatorTy B = std::round(float(biasW.at({d}) - biasOffset) *
                (biasScale / matMulScale));

            // Add the bias.
            sum += B;
            if(doRelu && sum<0)
              sum = 0;
            // Scale the result back to the expected destination scale.
            outW.at({n, ax, ay, d}) = quantization::clip<AccumulatorTy, ElemTy>(
                std::round(float(sum) * (matMulScale / outScale) + outOffset));
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

/// This is the quantized implementation of Convolution.
void BoundVTAFunction::fwdVTAConvolutionInstQuantizedImpl(
    Value *inV, Value *outV, Value *filterV, Value *biasV,
    llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
    llvm::ArrayRef<unsigned_t> pads, size_t group, size_t dilation,
    const int32_t fuseRelu) {
  auto inW = getWeightHandle<int8_t>(inV);
  auto outW = getWeightHandle<int8_t>(outV);
  auto filterW = getWeightHandle<int8_t>(filterV);
  auto biasW = getWeightHandle<int32_t>(biasV);

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;

  PaddingTLBR pdim(pads);
  auto outTy = outV->getType();
  auto inTy = inV->getType();
  auto filterTy = filterV->getType();
  auto biasTy = biasV->getType();

  float outScale = outTy->getScale();
  float inScale = inTy->getScale();
  float filterScale = filterTy->getScale();
  float biasScale = biasTy->getScale();
  filterScale = 1 / filterScale;
  inScale = 1 / inScale;
  biasScale = 1 / biasScale;
  outScale = 1 / outScale;

  // Calculate the scale of the values that come out of the matrix
  // multiplication part of the calculation.
  float matMulScale = inScale * filterScale;
  float scale = matMulScale / outScale;
  float tempScale = 1.0;
  assert(scale > 1);
  uint32_t shift = 0;
  {
    while (tempScale < scale) {
      tempScale *= 2;
      shift++;
    }
  }

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {
    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

            // For each element in the convolution-filter:
            int32_t sum = 0;
            for (dim_t fx = 0; fx < kdim.height; fx++) {
              for (dim_t fy = 0; fy < kdim.width; fy++) {
                sdim_t ox = x + fx * dilation;
                sdim_t oy = y + fy * dilation;

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= sdim_t(idim.w)) {
                  continue;
                }
                for (dim_t fd = 0; fd < inCperG; fd++) {

                  int32_t F = filterW.at({d, fx, fy, fd});
                  int32_t I =
                      inW.at({n, (dim_t)ox, (dim_t)oy, g * inCperG + fd});
                  // We represent the element multiplication with offset as
                  // (value - offset).
                  sum += (F) * (I);
                }
              }
            }

            // Scale the bias to match the scale of the matrix multiplication.

            int32_t B = biasW.at({d});

            // Add the bias.
            sum += B;

            // Scale the result back to the expected destination scale.

            outW.at({n, ax, ay, d}) =
                quantization::clip<int32_t, int8_t>(sum >> shift);

            if (fuseRelu) {
              outW.at({n, ax, ay, d}) =
                  outW.at({n, ax, ay, d}) > 0 ? outW.at({n, ax, ay, d}) : 0;
            }

          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

/*
void BoundVTAFunction::fwdVTAConvolutionInst(const glow::Instruction *I) {
  llvm::outs() << "Found VTAConvolution but VTAConv is not yet supported on VTA\n";
  assert(I->getKind() == Kinded::Kind::VTAConvolutionInstKind);
  auto *CI = llvm::cast<VTAConvolutionInst>(I);
  assert(CI);
  auto kernelSizes = CI->getKernels();
  auto pads = CI->getPads();
  auto strides = CI->getStrides();
  size_t group = CI->getGroup();

  assert(CI->getSrc()->getType()->isQuantizedType());

  auto inW = getWeightHandle<int8_t>(CI->getSrc());
  ShapeVTAIO idim(inW.dims());
  auto outW = getWeightHandle<int8_t>(CI->getDest());
  ShapeVTAIO odim(outW.dims());

  assert(CI->getSrc()->getElementType() ==  ElemKind::Int8QTy &&
     CI->getBias()->getElementType() ==  ElemKind::Int32QTy);

  Value *inV = CI->getSrc();
  Value *outV = CI->getDest();
  Value *filterV = CI->getFilter();
  Value *biasV = CI->getBias();

  auto filterW = getWeightHandle<int8_t>(filterV);
  ShapeVTAKernel fdim(filterW.dims());

  auto biasW = getWeightHandle<int32_t>(biasV);

  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);
  assert(group==1);

  PaddingTLBR pdim(pads);
  auto outTy = outV->getType();
  auto inTy = inV->getType();
  auto filterTy = filterV->getType();
  auto biasTy = biasV->getType();

  int32_t outOffset = outTy->getOffset();
  int32_t inOffset = inTy->getOffset();
  int32_t filterOffset = filterTy->getOffset();
  int32_t biasOffset = biasTy->getOffset();


  float outScale = outTy->getScale();
  float inScale = inTy->getScale();
  float filterScale = filterTy->getScale();
  float biasScale = biasTy->getScale();

  assert(outOffset == 0);
  assert(inOffset == 0);
  assert(filterOffset ==0);
  assert(biasOffset == 0);


  assert(pdim.top==pdim.left);
  assert(pdim.top==pdim.right);
  assert(pdim.top==pdim.bottom);
  uint32_t pad_size = pdim.top;
  assert(strides[0] == strides[1]);
  uint32_t stride_size = strides[0];

  bool doRelu = false;

  bool doBias = false;

  filterScale = 1/filterScale;
  inScale = 1/inScale;
  biasScale = 1/biasScale;
  outScale = 1/outScale;

  float matMulScale = inScale * filterScale;
  float scale =  matMulScale / outScale;
  float tempScale = 1.0;
  assert(scale > 1);
  uint32_t shift = 0;
  {
    while(tempScale<scale)
    {
      tempScale *= 2;
      shift++;
    }
    assert(tempScale==scale);
  }



  int inm = idim.nm;
  int ins = idim.ns;
  int icm = idim.cm;
  int ih = idim.h;
  int iw = idim.w;
  int ics = idim.cs;

  int8_t *input = (int8_t *)malloc(inm*ins*icm*ih*iw*ics);
  for (dim_t i0 = 0; i0 < inm; i0++) {
    for (dim_t i1 = 0; i1 < icm; i1++) {
      for (dim_t i2 = 0; i2 < ih; i2++) {
        for (dim_t i3 = 0; i3 < iw; i3++) {
          for (dim_t i4 = 0; i4 < ins; i4++) {
            for (dim_t i5 = 0; i5 < ics; i5++) {
              *(input + i0*icm*ih*iw*ins*ics + i1*ih*iw*ins*ics +
                i2*iw*ins*ics + i3*ins*ics + i4*ics + i5) =
                  inW.at({i0, i1, i2, i3, i4, i5});
            }
          }
        }
      }
    }
  }

  int KN = odim.cm*16;
  int KH = kdim.height;
  int KW = kdim.width;



  int fnm = fdim.nm;
  int fcm = fdim.cm;
  int fh = fdim.h;
  int fw = fdim.w;
  int fns = fdim.ns;
  int fcs = fdim.cs;

  int8_t *kernel = (int8_t *)malloc(fnm*fcm*fh*fw*fns*fcs);
  for (dim_t i0 = 0; i0 < fnm; i0++) {
    for (dim_t i1 = 0; i1 < fcm; i1++) {
      for (dim_t i2 = 0; i2 < fh; i2++) {
        for (dim_t i3 = 0; i3 < fw; i3++) {
          for (dim_t i4 = 0; i4 < fns; i4++) {
            for (dim_t i5 = 0; i5 < fcs; i5++) {
              *(kernel + i0*fcm*fh*fw*fns*fcs + i1*fh*fw*fns*fcs +
                i2*fw*fns*fcs + i3*fns*fcs + i4*fcs + i5) =
                  filterW.at({i0, i1, i2, i3, i4, i5});
            }
          }
        }
      }
    }
  }

  int32_t* bias = (int32_t *)malloc(odim.cs*odim.ns*sizeof(int32_t));
  for (unsigned long i = 0 ; i < odim.cs*odim.ns ; i++)
  {
    bias[i] = biasW.at({i});
    if(bias[i]!=0){
      doBias = true;
    }
  }

  int onm = odim.nm;
  int ons = odim.ns;
  int ocm = odim.cm;
  int oh = odim.h;
  int ow = odim.w;
  int ocs = odim.cs;

  int8_t *output = (int8_t *)malloc(onm*ons*ocm*oh*ow*ocs);

  convolution_wo_tr_wo_ch(input, kernel, bias, output, inm*ins, ih, iw, icm*ics, fnm*fns, fh, fw, pad_size, stride_size, doRelu, doBias, shift, oh, ow);


  for (dim_t i0 = 0; i0 < onm; i0++) {
    for (dim_t i1 = 0; i1 < ocm; i1++) {
      for (dim_t i2 = 0; i2 < oh; i2++) {
        for (dim_t i3 = 0; i3 < ow; i3++) {
          for (dim_t i4 = 0; i4 < ons; i4++) {
            for (dim_t i5 = 0; i5 < ocs; i5++) {
              outW.at({i0, i1, i2, i3, i4, i5})=
                  *(output + i0*ocm*oh*ow*ons*ocs + i1*oh*ow*ons*ocs +
                    i2*ow*ons*ocs + i3*ons*ocs + i4*ocs + i5);
            }
          }
        }
      }
    }
  }
  return;
}
*/
void BoundVTAFunction::fwdConvolutionInst(const ConvolutionInst *I) {
  auto kernelSizes = I->getKernels();
  auto pads = I->getPads();
  auto strides = I->getStrides();
  size_t group = I->getGroup();

  if (I->getSrc()->getType()->isQuantizedType()) {
    //assert(I->getSrc()->getElementType()==ElemKind::Int8QTy);
    auto inW = getWeightHandle<int8_t>(I->getSrc());
    ShapeNHWC idim(inW.dims());
    auto outW = getWeightHandle<int8_t>(I->getDest());
    ShapeNHWC odim(outW.dims());

    if(idim.c%16 ==0 && odim.c%16 ==0 &&
            I->getSrc()->getElementType() ==  ElemKind::Int8QTy &&
            I->getBias()->getElementType() ==  ElemKind::Int32QTy &&
        group == 1) {

      Value *inV = I->getSrc();
      Value *outV = I->getDest();
      Value *filterV = I->getFilter();
      Value *biasV = I->getBias();
      auto inW = getWeightHandle<int8_t>(inV);
      auto outW = getWeightHandle<int8_t>(outV);
      auto filterW = getWeightHandle<int8_t>(filterV);
      auto biasW = getWeightHandle<int32_t>(biasV);

      ShapeNHWC odim(outW.dims());
      ShapeNHWC idim(inW.dims());
      ShapeHW kdim(kernelSizes);
      ShapeHW sdim(strides);
      assert(group==1);
      assert(idim.c % group == 0 && "Input channels must be divisible by group.");
      assert(odim.c % group == 0 && "Output channels must be divisible by group.");

      PaddingTLBR pdim(pads);
      auto outTy = outV->getType();
      auto inTy = inV->getType();
      auto filterTy = filterV->getType();
      auto biasTy = biasV->getType();

      int32_t outOffset = outTy->getOffset();
      int32_t inOffset = inTy->getOffset();
      int32_t filterOffset = filterTy->getOffset();
      int32_t biasOffset = biasTy->getOffset();


      float outScale = outTy->getScale();
      float inScale = inTy->getScale();
      float filterScale = filterTy->getScale();
      float biasScale = biasTy->getScale();

      assert(outOffset == 0);
      assert(inOffset == 0);
      assert(filterOffset ==0);
      assert(biasOffset == 0);


      assert(pdim.top==pdim.left);
      assert(pdim.top==pdim.right);
      assert(pdim.top==pdim.bottom);
      uint32_t pad_size = pdim.top;
      assert(strides[0] == strides[1]);
      uint32_t stride_size = strides[0];

      bool doRelu = false;
      if (I->getFusedActivation() == FusedActivation::RELU) {
        doRelu = true;
      }


      bool doBias = false;

      filterScale = 1/filterScale;
      inScale = 1/inScale;
      biasScale = 1/biasScale;
      outScale = 1/outScale;

      float matMulScale = inScale * filterScale;
      float scale =  matMulScale / outScale;
      float tempScale = 1.0;
      assert(scale > 1);
      uint32_t shift = 0;
      {
        while(tempScale<scale)
        {
          tempScale *= 2;
          shift++;
        }
        assert(tempScale==scale);
      }


      int N = idim.n;
      int H = idim.h;
      int W = idim.w;
      int C = idim.c;
      int KN = odim.c;
      int KH = kdim.height;
      int KW = kdim.width;

      assert(C%16 == 0);
      assert(KN%16 == 0);

      int8_t *input = (int8_t *)malloc(N*C*H*W);
      for (dim_t n = 0; n < N; n++) {
        for (dim_t h = 0; h < H; h++) {
          for (dim_t w = 0; w < W; w++) {
            for (dim_t c = 0; c < C; c++) {
              *(input + n*H*W*C + h*W*C + w*C + c) = inW.at({n,h,w,c});
            }
          }
        }
      }

      int8_t *kernel = (int8_t *)malloc(KN*KH*KW*C +KN* sizeof(int32_t));

      for (int n = 0; n < KN; n++) {
        for (int h = 0; h < KH; h++) {
          for (int w = 0; w < KW; w++) {
            for (int c = 0; c < C; c++) {
              *(kernel + n*KH*KW*C + h*KW*C + w*C + c) =
                  filterW.at({(dim_t)n,(dim_t)h,(dim_t)w,(dim_t)c});
            }
          }
        }
      }

      int32_t* bias = (int32_t *) (kernel+KN*KH*KW*C);
      for (unsigned long i = 0 ; i < KN ; i++)
      {
        bias[i] = biasW.at({i});
        if(bias[i]!=0){
          doBias = true;
        }
      }
      int out_h = odim.h;
      int out_w = odim.w;
      int8_t *output = (int8_t *)malloc(N * out_h * out_w * KN);
      convolution(input, kernel, output, N, H, W, C, KN, KH, KW, pad_size, stride_size, doRelu, doBias, shift, out_h, out_w);
      for (int n = 0; n < N; n++) {
        for (int h = 0; h < out_h; h++) {
          for (int w = 0; w < out_w; w++) {
            for (int c = 0; c < KN; c++) {
              outW.at({(dim_t) n, (dim_t) h, (dim_t) w, (dim_t) c}) = *(output + n * out_h * out_w * KN + h * out_w * KN + w * KN + c);
            }
          }
        }
      }


    }
    else if(group != 1) {
      bool doRelu = false;
      if (I->getFusedActivation() == FusedActivation::RELU) {
        doRelu = true;
      }
      fwdVTAConvolutionInstQuantizedImpl(I->getSrc(), I->getDest(),
          I->getFilter(), I->getBias(), kernelSizes, strides, pads, group,
          I->getDilation(), doRelu);
    }
    else{
      bool doRelu = false;
      if (I->getFusedActivation() == FusedActivation::RELU) {
        doRelu = true;
      }

      dispatchQuantizedWithAccumulationAndBiasImpl(
          fwdConvolutionInstQuantizedImpl, I->getSrc()->getElementType(),
          I->getBias()->getElementType(), I->getSrc(), I->getDest(),
          I->getFilter(), I->getBias(), kernelSizes, strides, pads, group,
          I->getDilation(), doRelu);
    }
    return;
  }

  llvm_unreachable("Not supported for VTA");
  return;
}

//===----------------------------------------------------------------------===//
//                  Tensor allocation operations
//===----------------------------------------------------------------------===//

void BoundVTAFunction::fwdAllocActivationInst(
    const AllocActivationInst *I) {
  getOrCreateTensor(I);
}

void BoundVTAFunction::fwdDeallocActivationInst(
    const DeallocActivationInst *I) {
  deleteTensor(I->getSrc());
}

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
void BoundVTAFunction::fwdVTAReluInstQuantizedImpl(Value *inV,
                                                              Value *outV,
                                                              float scale) {
  auto inW = getWeightHandle<int8_t>(inV);
  auto outW = getWeightHandle<int8_t>(outV);

  float tempScale = 1.0;
  int shift = 0;

  /*
  for (dim_t i = 0, e = outW.size(); i < e; i++) {
    float val = inW.raw(i);
    outW.raw(i) = MAX(val, 0);
  }
  */

  if (scale < 1){
    // Compute left bit shifting
    while (tempScale > scale) {
      tempScale /= 2;
      shift++;
    }
    for (dim_t i = 0, e = outW.size(); i < e; i++) {
      int32_t val = inW.raw(i);
      val = val << shift;
      outW.raw(i) = quantization::clip<int32_t, int8_t>(MAX(val, 0));
    }
  }else{
    // Compute right bit shifting
    while (tempScale < scale) {
      tempScale *= 2;
      shift++;
    }
    for (dim_t i = 0, e = outW.size(); i < e; i++) {
      int32_t val = inW.raw(i);
      val = val >> shift;
      outW.raw(i) = quantization::clip<int32_t, int8_t>(MAX(val, 0));
    }
  }
}

void BoundVTAFunction::fwdReluInst(const ReluInst *I) {
  // DCHECK(!"Found ReluInst but Relu is lowered on VTAInterpreter");
  if (I->getSrc()->getType()->isQuantizedType()) {
    assert(I->getSrc()->getElementType() == ElemKind::Int8QTy);
    auto inW = getWeightHandle<int8_t>(I->getSrc());
    ShapeNHWC idim(inW.dims());
    auto outW = getWeightHandle<int8_t>(I->getDest());
    ShapeNHWC odim(outW.dims());

    auto inTy = I->getSrc()->getType();
    float inScale = inTy->getScale();
    inScale = 1 / inScale;
    auto outTy = I->getDest()->getType();
    float outScale = outTy->getScale();
    outScale = 1 / outScale;
    float scale = inScale / outScale;
    float tempScale = 1.0;
    //assert(scale > 1); //in relu case, outScale could be less than inScale
    // because of negative value.
    /*uint32_t shift = 0;
    {
      while (tempScale < scale) {
        tempScale *= 2;
        shift++;
      }
    }
    */

    int32_t outOffset = outTy->getOffset();
    int32_t inOffset = inTy->getOffset();
    if (outOffset == 0 && inOffset == 0) {
      fwdVTAReluInstQuantizedImpl(I->getSrc(), I->getDest(), scale);
    } else {
      llvm::outs()<<"Unsupported Quantized Relu"<<"\n";
      return;
    }
  } else {
    llvm::outs()<<"Unsupported Relu"<<"\n";
  }
}


//===----------------------------------------------------------------------===//
//                       Pooling
//===----------------------------------------------------------------------===//
template <class T>
static void fwdMaxPool(Tensor *inW, Tensor *outW, Tensor *argmaxW,
                       llvm::ArrayRef<unsigned_t> kernelSizes,
                       llvm::ArrayRef<unsigned_t> strides,
                       llvm::ArrayRef<unsigned_t> pads) {
  ShapeNHWC odim(outW->dims());
  ShapeNHWC idim(inW->dims());
  Handle<T> inHandle = inW->getHandle<T>();
  Handle<T> outHandle = outW->getHandle<T>();
  PaddingTLBR pdim(pads);
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  llvm::Optional<Handle<int64_t>> argmaxH;
  if (argmaxW) {
    argmaxH = argmaxW->getHandle<int64_t>();
  }
  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (dim_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      sdim_t x = -sdim_t(pdim.top);
      for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
        sdim_t y = -sdim_t(pdim.left);
        for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

          bool first = true;
          T max_value = 0;
          dim_t argmaxNHWC = 0;

          for (dim_t fx = 0; fx < kdim.height; fx++) {
            for (dim_t fy = 0; fy < kdim.width; fy++) {
              sdim_t ox = x + fx;
              sdim_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }

              T val = inHandle.at({n, (dim_t)ox, (dim_t)oy, z});
              if (first || (val >= max_value)) {
                first = false;
                max_value = val;
                if (argmaxW) {
                  argmaxNHWC = &inHandle.at({n, (dim_t)ox, (dim_t)oy, z}) -
                      &inHandle.raw(0);
                }
              }
            }
          }

          outHandle.at({n, ax, ay, z}) = max_value;

          if (argmaxW) {
            (*argmaxH).at({n, ax, ay, z}) = argmaxNHWC;
          }
        } // W
      }   // H
    }     // C
  }       // N
}


void BoundVTAFunction::fwdMaxPoolInst(const MaxPoolInst *I) {
  auto inW = getTensor(I->getSrc());
  auto outW = getTensor(I->getDest());

  if (inW->getType().isQuantizedType()) {
    dispatchQuantizedImpl(fwdMaxPool, inW->getType().getElementType(), inW,
                          outW, nullptr, I->getKernels(), I->getStrides(),
                          I->getPads());
    return;
  }

  dispatchFloatingPointImpl(fwdMaxPool, inW->getType().getElementType(), inW,
                            outW, nullptr, I->getKernels(), I->getStrides(),
                            I->getPads());
}








void BoundVTAFunction::fwdAvgPoolInstI8Impl(const AvgPoolInst *I) {
  ShapeNHWC odim(I->getDest()->dims());
  ShapeNHWC idim(I->getSrc()->dims());

  PaddingTLBR pdim(I->getPads());
  ShapeHW kdim(I->getKernels());
  ShapeHW sdim(I->getStrides());
  // Implement the avg pooling operation as defined here:
  // https://arxiv.org/abs/1312.4400
  float filterArea = kdim.height * kdim.width;

  auto inW = getWeightHandle<int8_t>(I->getSrc());
  auto outW = getWeightHandle<int8_t>(I->getDest());
  TensorQuantizationParams inQP{I->getSrc()->getType()->getScale(),
                                I->getSrc()->getType()->getOffset()};
  TensorQuantizationParams outQP{I->getDest()->getType()->getScale(),
                                 I->getDest()->getType()->getOffset()};

  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {
    // For each layer in the output tensor:
    for (dim_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pdim.top);
      for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
        ssize_t y = -ssize_t(pdim.left);
        for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
          int32_t sum = 0;

          for (dim_t fx = 0; fx < kdim.height; fx++) {
            for (dim_t fy = 0; fy < kdim.width; fy++) {
              sdim_t ox = x + fx;
              sdim_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }

              sum += inW.at({n, (dim_t)ox, (dim_t)oy, z}) - inQP.offset;
            }
          }
          // Instead of dividing by filterArea, just change scale.
          outW.at({n, ax, ay, z}) = quantization::clip<int32_t, int8_t>(
              std::round(float(sum) * (inQP.scale / outQP.scale / filterArea) +
                  outQP.offset));
        } // W
      }   // H
    }     // C
  }       // N
}





void BoundVTAFunction::fwdAvgPoolInst(const AvgPoolInst *I) {
  bool isConv3D = is3DData(ConvolutionLayout(I->getLayout()));
  bool isQuantized = I->getSrc()->getType()->isQuantizedType();

  if (isConv3D) {
    if (isQuantized) {
      llvm_unreachable("DebugPrint format not supported!");

    } else {
      llvm_unreachable("DebugPrint format not supported!");

    }
  } else {
    if (isQuantized) {
      fwdAvgPoolInstI8Impl(I);
    } else {
      llvm_unreachable("DebugPrint format not supported!");

    }
  }
}





//===----------------------------------------------------------------------===//
//                Instructions used by Quantization
//===----------------------------------------------------------------------===//
/// Quantize floating point tensor. Scale and Offset are based on return type
/// of the instruction \p I.
void BoundVTAFunction::fwdQuantizeInst(const glow::QuantizeInst *I) {
  auto *srcTensor = getTensor(I->getSrc());
  auto *destTensor = getTensor(I->getDest());
  auto destTy = destTensor->getType();
  Tensor qTensor = quantization::quantizeTensor(
      *srcTensor, {destTy.getScale(), destTy.getOffset()},
      destTy.getElementType());
  destTensor->assign(&qTensor);
}

/// Dequantize integer tensor. Scale and Offset are based
/// on the source tensor type.
void BoundVTAFunction::fwdDequantizeInst(
    const glow::DequantizeInst *I) {
  auto *srcTensor = getTensor(I->getSrc());
  auto *destTensor = getTensor(I->getDest());
  auto destTy = destTensor->getType();
  Tensor fTensor =
      quantization::dequantizeTensor(*srcTensor, destTy.getElementType());
  destTensor->assign(&fTensor);
}


//===----------------------------------------------------------------------===//
//                       Debug instructions
//===----------------------------------------------------------------------===//
/// Prints a value of the instruction's operand.
/// In most cases it will be the name of the variable and the value of the
/// tensor.
void BoundVTAFunction::fwdDebugPrintInst(const DebugPrintInst *I) {
  auto *V = I->getSrc();
  auto *T = getTensor(V);
  std::string format = I->getFormat();
  std::string filename = I->getFileName();

  if (format == "console") {
    // Dump tensor in console.
    llvm::outs() << I->getName() << ": ";
    V->dump();
    llvm::outs() << "\n";
    dumpImpl(T);
    llvm::outs() << "\n";
  } else if (format == "bin") {
    TensorSerializationOptions opts;
    opts.withType = true;
    glow::dumpTensorToBinaryFile(*T, filename, opts);
  } else if (format == "txt") {
    TensorSerializationOptions opts;
    opts.withType = true;
    glow::dumpTensorToTextFile(*T, filename, opts);
  } else if (format == "rawbin") {
    TensorSerializationOptions opts;
    opts.withType = false;
    glow::dumpTensorToBinaryFile(*T, filename, opts);
  } else if (format == "rawtxt") {
    TensorSerializationOptions opts;
    opts.withType = false;
    glow::dumpTensorToTextFile(*T, filename, opts);
  } else {
    llvm_unreachable("DebugPrint format not supported!");
  }
}


//===----------------------------------------------------------------------===//
//                       Tensor shape (copy/transpose/concat/...)
//===----------------------------------------------------------------------===//


void BoundVTAFunction::fwdInsertTensorInst(
        const glow::InsertTensorInst *I) {
  Tensor *outT = getTensor(I->getDest());
  Tensor *inT = getTensor(I->getSrc());
  ElemKind k = outT->getElementType();
#define TYPED_INSERT(TY, TYPEKIND)                                             \
  if (k == TYPEKIND) {                                                         \
    auto OH = outT->getHandle<TY>();                                           \
    auto IH = inT->getHandle<TY>();                                            \
    return OH.insertTensors(IH, I->getOffsets(), I->getCount(), I->getAxis()); \
  }

  TYPED_INSERT(int64_t, ElemKind::Int64ITy);
  TYPED_INSERT(int32_t, ElemKind::Int32ITy);
  TYPED_INSERT(float, ElemKind::FloatTy);
  TYPED_INSERT(float16_t, ElemKind::Float16Ty);
  TYPED_INSERT(bfloat16_t, ElemKind::BFloat16Ty);
  TYPED_INSERT(int8_t, ElemKind::Int8QTy);
  TYPED_INSERT(bool, ElemKind::BoolTy);
#undef TYPED_INSERT

  llvm_unreachable("Unsupported tensor type");
}

void BoundVTAFunction::fwdTensorViewInst(const TensorViewInst *I) {
  getOrCreateUnownedTensor(I, I->getSrc(), I->getOffsets());
}



void BoundVTAFunction::fwdTransposeInst(const TransposeInst *I) {
  auto inT = getTensor(I->getSrc());
  (void)inT;
  auto outT = getTensor(I->getDest());

  assert(outT->size() == inT->size() && "Invalid tensor dimensions");

  if (I->getSrc()->getType()->isQuantizedType()) {
    inT->transpose(outT, I->getShuffle());
  } else {
    inT->transpose(outT, I->getShuffle());
  }
}


void BoundVTAFunction::fwdSplatInst(const glow::SplatInst *I) {
  auto *T = getTensor(I->getDest());
  ElemKind k = T->getElementType();

  if (k == ElemKind::Int32ITy) {
    return T->getHandle<int32_t>().clear(I->getValue());
  }

  if (k == ElemKind::Int64ITy) {
    return T->getHandle<int64_t>().clear(I->getValue());
  }

  if (k == ElemKind::Int32ITy) {
    return T->getHandle<int32_t>().clear(I->getValue());
  }

  if (k == ElemKind::FloatTy) {
    return T->getHandle<float>().clear(I->getValue());
  }

  if (k == ElemKind::Float16Ty) {
    return T->getHandle<float16_t>().clear(I->getValue());
  }

  if (k == ElemKind::BFloat16Ty) {
    return T->getHandle<bfloat16_t>().clear(I->getValue());
  }

  if (k == ElemKind::BoolTy) {
    return T->getHandle<bool>().clear(static_cast<bool>(I->getValue()));
  }

  if (k == ElemKind::Int8QTy) {
    // Quantize the requested floating point splat value into the correct
    // integer representation.
    auto destTy = I->getDest()->getType();
    TensorQuantizationParams destQ{destTy->getScale(), destTy->getOffset()};
    float val = I->getValue();
    return T->getHandle<int8_t>().clear(quantization::quantize(val, destQ));
  }

  if (k == ElemKind::BoolTy) {
    return T->getHandle<bool>().clear(static_cast<bool>(I->getValue()));
  }

  llvm_unreachable("Unsupported tensor type");
}


//===----------------------------------------------------------------------===//
//                                 FC
//===----------------------------------------------------------------------===//
template <typename ElemTy, typename AccumulatorTy, typename BiasElemTy>
void BoundVTAFunction::fwdFullyConnectedInstQuantizedImpl(
    const glow::FullyConnectedInst *I) {
  assert(getTensor(I->getSrc())->getType().isQuantizedType());

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto weightsW = getWeightHandle<ElemTy>(I->getWeights());
  auto biasW = getWeightHandle<BiasElemTy>(I->getBias());
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  auto inTy = inW.getType();
  auto weightsTy = weightsW.getType();
  auto biasTy = biasW.getType();
  auto outTy = outW.getType();

  int32_t inOffset = inTy.getOffset();
  int32_t weightsOffset = weightsTy.getOffset();
  int32_t biasOffset = biasTy.getOffset();
  int32_t outOffset = outTy.getOffset();

  float outScale = outTy.getScale();
  float weightsScale = weightsTy.getScale();
  float biasScale = biasTy.getScale();
  float inScale = inTy.getScale();

  ShapeHW idim(inW.dims());
  ShapeHW odim(outW.dims());

  // Calculate the scale of the values that come out of the matrix
  // multiplication part of the calculation.
  float matMulScale = weightsScale * inScale;

  outW.clear(0);

  for (dim_t i = 0; i < idim.height; i++) {
    for (dim_t j = 0; j < odim.width; j++) {
      AccumulatorTy sum = 0;
      for (dim_t k = 0; k < idim.width; k++) {
        AccumulatorTy W = weightsW.at({k, j});
        AccumulatorTy A = inW.at({i, k});
        sum += (W - weightsOffset) * (A - inOffset);
      }

      // Scale the bias to match the scale of the matrix multiplication.
      AccumulatorTy B = std::round(float(biasW.at({j}) - biasOffset) *
          (biasScale / matMulScale));

      // Add the bias.
      sum += B;

      // Scale the result back to the expected destination scale.
      outW.at({i, j}) = quantization::clip<AccumulatorTy, ElemTy>(
          std::round(float(sum) * (matMulScale / outScale)) + outOffset);
    }
  }
}

void BoundVTAFunction::fwdFullyConnectedInst(
    const glow::FullyConnectedInst *I) {

  if (getTensor(I->getSrc())->getType().isQuantizedType()) {
    dispatchQuantizedWithAccumulationAndBiasImpl(
        fwdFullyConnectedInstQuantizedImpl, I->getSrc()->getElementType(),
        I->getBias()->getElementType(), I);
    return;
  } else {
    llvm_unreachable("Type is not supported");                                 \
  }
}



//===----------------------------------------------------------------------===//
//                        Loss Functions (Softmax/regression/...)
//===----------------------------------------------------------------------===//


template <typename ElemTy>
void BoundVTAFunction::fwdSoftMaxInstImpl(const SoftMaxInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto outW = getWeightHandle<ElemTy>(I->getDest());
  auto idim = inW.dims();

  for (dim_t n = 0; n < idim[0]; n++) {
    // Find Max.
    float max = float(inW.at({n, 0}));
    for (dim_t i = 1; i < idim[1]; i++) {
      max = std::max(max, float(inW.at({n, i})));
    }

    // Compute exp.
    float sum = 0;
    for (dim_t i = 0; i < idim[1]; i++) {
      float e = std::exp(float(inW.at({n, i})) - max);
      sum += e;
      outW.at({n, i}) = ElemTy(e);
    }

    // Normalize the output.
    for (dim_t i = 0; i < idim[1]; i++) {
      outW.at({n, i}) = ElemTy(float(outW.at({n, i})) / sum);
    }
  } // N
}

void BoundVTAFunction::fwdSoftMaxInst(const SoftMaxInst *I) {
  dispatchFloatingPointImpl(fwdSoftMaxInstImpl, I->getSrc()->getElementType(),
                            I);
}

