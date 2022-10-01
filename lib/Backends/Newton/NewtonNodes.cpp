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
#include "glow/IR/Instrs.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Quantization/Base/Profile.h"

#include "llvm/ADT/SmallVector.h"


#include <chrono>
#include <cmath>

#include <aim.h>
#include "src/activation/include/activation.h"
#include "src/datatype/include/aim_mac.h"
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

void BoundNewtonFunction::fwdAllocActivationInst(
    const AllocActivationInst *I) {
  getOrCreateTensor(I);
}

void BoundNewtonFunction::fwdDeallocActivationInst(
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
void BoundNewtonFunction::fwdFullyConnectedInstFloatImpl(
    const FullyConnectedInst *I) {
  staticAssertFloatingPointType(ElemTy);

  auto inW = getWeightHandle<ElemTy>(I->getSrc());
  auto weightsW = getWeightHandle<ElemTy>(I->getWeights());
  auto biasW = getWeightHandle<ElemTy>(I->getBias());
  auto outW = getWeightHandle<ElemTy>(I->getDest());

  ShapeHW idim(inW.dims());
  ShapeHW wdim(weightsW.dims());
  ShapeHW odim(outW.dims());
  outW.clear(0);

  int iH = idim.height;
  int iW = idim.width;
  float *input = (float *) malloc(iH * iW * 4);
  for (dim_t h = 0; h < iH; h++) {
    for (dim_t w = 0; w < iW; w++) {
      *(input + h * iW + w) = inW.at({h, w});
    }
  }

  int wH = wdim.height;
  int wW = wdim.width;
  float *weight = (float *) malloc(wH * wW * 4);
  for (dim_t h = 0; h < wH; h++) {
    for (dim_t w = 0; w < wW; w++) {
      *(weight + w * wH + h) = weightsW.at({h, w});

    }
  }

  float *bias = (float *) malloc(odim.width * 4);

  for(dim_t w = 0; w < odim.width; w++){
    *(bias + w) = biasW.at({w});
  }


  int oH = odim.height;
  int oW = odim.width;
  float *output = (float *)malloc(oH * oW * 4);

  struct act_info act_info {static_cast<int>(ACTIVATION::BYPASS_ACT), 0.0f, 0};
  int op_id = CreateAiMLinearOp(-1, weight, bias,
                                iW, oW, act_info, false, false);
  // perform matrix multiplication - bias addition - activation
  struct analytic_result_ profile_ptr;
  RunAiMLinearOp(op_id, input, output, iH, (void*) &profile_ptr);

  for (dim_t h = 0; h < oH; h++) {
    for (dim_t w = 0; w < oW; w++) {
      outW.at({h, w}) = *(output + h * oW + w);
    }
  }

  //debug

//
//  std::cout<<"YIKWON INPUT::\n";
//  for(int i = 0; i < iH * iW; i++)
//  {
//    std::cout<<input[i]<<std::endl;
//  }
//  std::cout<<"YIKWON WEIGHT::\n";
//  for(int i = 0; i < wH * wW; i++)
//  {
//    std::cout<<weight[i]<<std::endl;
//  }
//  std::cout<<"YIKWON BIAS::\n"<<bias[0]<<std::endl;
//  std::cout<<"YIKWON OUTPUT\n::"<<output[0]<<std::endl;




//  float *output_cpu = (float *)malloc(oH * oW * 4);
//  for(dim_t w = 0; w < odim.width; w++){
//    *(output_cpu + w) = *(bias + w);
//  }
//  gemm_nt(iH, oW, iW, 1,
//          input, iW,
//          weight, iW,
//          output_cpu, oW);
//
//  std::vector<float> vInput {input, input+iH*iW};
//  std::vector<float> vWeight {weight, weight+wH*wW};
//  std::vector<float> vBias {bias, bias+oW};
//  AiMMAC mac;
//  auto output_aim_model = mac(vInput, vWeight, vBias);
//  auto output_aim_model_data = output_aim_model.data();

//  for (dim_t h = 0; h < oH; h++) {
//    for (dim_t w = 0; w < oW; w++) {
//      float temp1 = *(output + h * oW + w);
//      float temp2 = *(output_cpu + h * oW + w);
//      float temp3 = *(output_aim_model_data + h * oW + w);
//      std::cout << temp2 << ", "<<temp1<<", "<<temp3<<std::endl;
//    }
//  }
//  free(output_cpu);

  free(input);
  free(weight);
  free(output);
  free(bias);

}



void BoundNewtonFunction::fwdFullyConnectedInst(
    const glow::FullyConnectedInst *I) {

  if (getTensor(I->getSrc())->getType().isQuantizedType()) {
    llvm_unreachable("Type is not supported");
    return;
  } else {
    fwdFullyConnectedInstFloatImpl<float>(I);
  }
}

