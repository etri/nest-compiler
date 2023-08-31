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
 * Modifications copyright (C) 2023 <ETRI/Yongin Kwon>
 */

#include "Enlight.h"
#include "glow/IR/Instrs.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Quantization/Base/Profile.h"

#include "llvm/ADT/SmallVector.h"

using namespace glow;

#define staticAssertFloatingPointType(ElemTy)                                  \
  static_assert(                                                               \
      std::is_floating_point<ElemTy>::value ||                                 \
          std::is_same<float16_t,                                              \
                       typename std::remove_cv<ElemTy>::type>::value,          \
      "This implementation is for floating-point values only")


//===----------------------------------------------------------------------===//
//                  Tensor allocation operations
//===----------------------------------------------------------------------===//

void BoundEnlightFunction::fwdAllocActivationInst(
    const AllocActivationInst *I) {
  getOrCreateTensor(I);
}

void BoundEnlightFunction::fwdDeallocActivationInst(
    const DeallocActivationInst *I) {
  deleteTensor(I->getSrc());
}

//===----------------------------------------------------------------------===//
//                                 FC
//===----------------------------------------------------------------------===//

void gemm_nt(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc) {
  int i, j, k;
#pragma omp parallel for
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float sum = 0;
      for (k = 0; k < K; ++k) {
        sum += ALPHA*A[i*lda+k]*B[j*ldb+k];
      }
      C[i*ldc+j] += sum;
    }
  }
}


template <typename ElemTy>
void BoundEnlightFunction::fwdFullyConnectedInstFloatImpl(
    const FullyConnectedInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto weightsW = getWeightHandle<ElemTy>(I->getWeights());
  auto biasW = getWeightHandle<ElemTy>(I->getBias());
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  ShapeHW idim(inW.dims());
  ShapeHW odim(outW.dims());

  outW.clear(0);

  for (dim_t i = 0; i < idim.height; i++) {
    for (dim_t j = 0; j < odim.width; j++) {
      float sum = 0;
      for (dim_t k = 0; k < idim.width; k++) {
        sum += float(inW.at({i, k})) * float(weightsW.at({k, j}));
      }

      outW.at({i, j}) = sum + float(biasW.at({j}));
    }
  }
}



void BoundEnlightFunction::fwdFullyConnectedInst(
    const glow::FullyConnectedInst *I) {

  if (getTensor(I->getSrc())->getType().isQuantizedType()) {
    llvm_unreachable("Type is not supported");
    return;
  } else {
    fwdFullyConnectedInstFloatImpl<float>(I);
  }
}
//===----------------------------------------------------------------------===//
//                       Convolution
//===----------------------------------------------------------------------===//

void BoundEnlightFunction::fwdConvolutionInst(const ConvolutionInst *I) {
  auto kernelSizes = I->getKernels();
  auto pads = I->getPads();
  auto strides = I->getStrides();
  size_t group = I->getGroup();

  return;
}