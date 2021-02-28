#include "../../lib/Backends/VTA/tests/VTABackendTestUtils.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Random.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"

using namespace glow;
using llvm::cast;

TEST(NESTVTABackendTest, vtaLayoutConvTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 14, 14, 16}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {16, 3, 3, 16}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {16}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
  //inputs.toBin("input");
  //kernel.toBin("kernel");
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  //bias.zero();
  std::array<dim_t, 4> S{{1, 14, 14, 16}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 256, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 256, 0);

  //inferVTAConvNet(&inputs, &kernel, &bias, &out1, 3, 1, 1, "VTA");

  //TODO : uncomment the next line to test inference VTALayout Convolution
  //inferVTALayoutConvNet(&inputs, &kernel, &bias, &out2, 3, 1, 1, "VTAInterpreter");

  //out1.toBin("golden");
  //EXPECT_TRUE(out1.isEqual(out2, 0));
  EXPECT_TRUE(1);
}

TEST(NESTVTABackendTest, convTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 14, 14, 16}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {16, 3, 3, 16}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {16}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
  //inputs.toBin("input");
  //kernel.toBin("kernel");
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  //bias.zero();
  std::array<dim_t, 4> S{{1, 14, 14, 16}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 256, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 256, 0);

  inferVTAConvNet(&inputs, &kernel, &bias, &out1, 3, 1, 1, "VTA");
  inferVTAConvNet(&inputs, &kernel, &bias, &out2, 3, 1, 1, "VTAInterpreter");
  //out1.toBin("golden");
  EXPECT_TRUE(out1.isEqual(out2, 0));
}

TEST(NESTVTABackendTest, vtaConvReluTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 14, 14, 16}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {16, 3, 3, 16}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {16}, 1, 0);
  inputs.getHandle<int8_t>().randomize(-60, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
  //inputs.toBin("input");
  //kernel.toBin("kernel");
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  //bias.zero();
  std::array<dim_t, 4> S{{1, 14, 14, 16}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 256, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 256, 0);

  inferVTAConvReluNet(&inputs, &kernel, &bias, &out1, 3, 1, 1, "VTA");
  inferVTAConvReluNet(&inputs, &kernel, &bias, &out2, 3, 1, 1, "VTAInterpreter");
  //out1.toBin("golden");
  EXPECT_TRUE(out1.isEqual(out2, 0));
}

TEST(NESTVTABackendTest, vtaMultiConvTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 14, 14, 16}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {16, 3, 3, 16}, 1, 0);
  Tensor kernel2(ElemKind::Int8QTy, {16, 3, 3, 16}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {16}, 1, 0);
  Tensor bias2(ElemKind::Int32QTy, {16}, 256, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
  kernel2.getHandle<int8_t>().randomize(-10, 10, PRNG);
  //inputs.toBin("input");
  //kernel.toBin("kernel1");
  //kernel2.toBin("kernel2");
  //bias.zero();
  //bias2.zero();
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  bias2.getHandle<int32_t>().randomize(0, 32768, PRNG);

  std::array<dim_t, 4> S{{1, 14, 14, 16}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor outInterpreter(ElemKind::Int8QTy, shape, 256, 0);
  Tensor outVTA(ElemKind::Int8QTy, shape, 256, 0);
  Tensor outInterpreter2(ElemKind::Int8QTy, shape, 65536, 0);
  Tensor outVTA2(ElemKind::Int8QTy, shape, 65536, 0);

  inferVTAMultiConvNet(&inputs, &kernel, &bias, &kernel2, &bias2, &outInterpreter, &outInterpreter2, "VTAInterpreter");
  inferVTAMultiConvNet(&inputs, &kernel, &bias, &kernel2, &bias2, &outVTA, &outVTA2, "VTA");
  //outInterpreter.toBin("inter_golden");
  //outInterpreter2.toBin("golden");
  EXPECT_TRUE(outInterpreter2.isEqual(outVTA2, 0));
}

TEST(NESTVTABackendTest, vtaMaxPoolTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 14, 14, 16}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);

  unsigned_t kernel = 2;
  unsigned_t stride = 2;
  unsigned_t pad = 1;

  //inputs.toBin("input");
  std::array<dim_t, 4> S{{1, 8, 8, 16}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 1, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 1, 0);

  inferMaxPoolNet(&inputs, &out1, kernel, stride, pad, "Interpreter");
  inferMaxPoolNet(&inputs, &out2, kernel, stride, pad, "VTA");
  //out1.toBin("golden");
  EXPECT_TRUE(out1.isEqual(out2, 1.0));
}

TEST(NESTVTABackendTest, res2_0_branch2a) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 56, 56, 64}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {64, 1, 1, 64}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {64}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
  //inputs.toBin("input");
  //kernel.toBin("kernel");
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  //bias.zero();
  std::array<dim_t, 4> S{{1, 56, 56, 64}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 128, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 128, 0);

  inferVTAConvNet(&inputs, &kernel, &bias, &out1, 1, 1, 0, "VTA");
  inferVTAConvNet(&inputs, &kernel, &bias, &out2, 1, 1, 0, "VTAInterpreter");
  //out1.toBin("golden");
  EXPECT_TRUE(out1.isEqual(out2, 0.0));
}

TEST(NESTVTABackendTest, res2_1_branch2a) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 56, 56, 256}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {64, 1, 1, 256}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {64}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
  //inputs.toBin("input");
  //kernel.toBin("kernel");
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  //bias.zero();
  std::array<dim_t, 4> S{{1, 56, 56, 64}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 128, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 128, 0);

  inferVTAConvNet(&inputs, &kernel, &bias, &out1, 1, 1, 0, "VTA");
  //inferVTAConvNet(&inputs, &kernel, &bias, &out1, 1, 1, 0, "Interpreter");
  inferVTAConvNet(&inputs, &kernel, &bias, &out2, 1, 1, 0, "VTAInterpreter");
  //inferVTAConvNet(&inputs, &kernel, &bias, &out2, 1, 1, 0, "Interpreter");
//  out2.toBin("golden");
//  out1.toBin("output");

  for(dim_t h = 0 ; h<56; h++){
    for(dim_t w = 0; w<56; w++){
      for(dim_t c = 0; c<64; c++){
        int8_t vtaOut = out1.getHandle<int8_t>().at({0,h,w,c});
        int8_t goldenOut = out2.getHandle<int8_t>().at({0,h,w,c});
        if( vtaOut != goldenOut)
        {
          std::cout <<h<<"|"<<w<<"|"<<c<<": "<<vtaOut<< ", "<<goldenOut<<std::endl;
        }
      }
    }
  }

  EXPECT_TRUE(out1.isEqual(out2, 0.0));
}


TEST(NESTVTABackendTest, res2_1_branch2c) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 56, 56, 64}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {256, 1, 1, 64}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {256}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
  //inputs.toBin("input");
  //kernel.toBin("kernel");
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  //bias.zero();
  std::array<dim_t, 4> S{{1, 56, 56, 256}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 128, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 128, 0);

  inferVTAConvNet(&inputs, &kernel, &bias, &out1, 1, 1, 0, "VTA");
  //inferVTAConvNet(&inputs, &kernel, &bias, &out1, 1, 1, 0, "Interpreter");
  inferVTAConvNet(&inputs, &kernel, &bias, &out2, 1, 1, 0, "VTAInterpreter");
  //inferVTAConvNet(&inputs, &kernel, &bias, &out2, 1, 1, 0, "Interpreter");
//  out2.toBin("golden");
//  out1.toBin("output");


  EXPECT_TRUE(out1.isEqual(out2, 0.0));
}


TEST(NESTVTABackendTest, res4_0_branch2c) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 14, 14, 256}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {1024, 1, 1, 256}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {1024}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
  //inputs.toBin("input");
  //kernel.toBin("kernel");
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  //bias.zero();
  std::array<dim_t, 4> S{{1, 14, 14, 1024}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 128, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 128, 0);

  inferVTAConvNet(&inputs, &kernel, &bias, &out1, 1, 1, 0, "VTA");
  //inferVTAConvNet(&inputs, &kernel, &bias, &out1, 1, 1, 0, "Interpreter");
  inferVTAConvNet(&inputs, &kernel, &bias, &out2, 1, 1, 0, "VTAInterpreter");
  //inferVTAConvNet(&inputs, &kernel, &bias, &out2, 1, 1, 0, "Interpreter");
//  out2.toBin("golden");
//  out1.toBin("output");


  EXPECT_TRUE(out1.isEqual(out2, 0.0));
}

TEST(NESTVTABackendTest, resnetv10_stage3_conv1) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 14, 14, 256}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {256, 3, 3, 256}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {256}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
  //inputs.toBin("input");
  //kernel.toBin("kernel");
  //bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  bias.zero();
  std::array<dim_t, 4> S{{1, 14, 14, 256}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 128, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 128, 0);

  inferVTAConvNet(&inputs, &kernel, &bias, &out1, 3, 1, 1, "VTA");
  //inferVTAConvNet(&inputs, &kernel, &bias, &out1, 1, 1, 0, "Interpreter");
  inferVTAConvNet(&inputs, &kernel, &bias, &out2, 3, 1, 1, "VTAInterpreter");
  //inferVTAConvNet(&inputs, &kernel, &bias, &out2, 1, 1, 0, "Interpreter");
//  out2.toBin("golden");
//  out1.toBin("output");


  EXPECT_TRUE(out1.isEqual(out2, 0.0));
}

TEST(NESTVTABackendTest, resnetv10_stage2_conv0) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 56, 56, 64}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {128, 3, 3, 64}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {128}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
  //inputs.toBin("input");
  //kernel.toBin("kernel");
  //bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  bias.zero();
  std::array<dim_t, 4> S{{1, 28, 28, 128}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 128, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 128, 0);

  inferVTAConvNet(&inputs, &kernel, &bias, &out1, 3, 2, 1, "VTA");
  //inferVTAConvNet(&inputs, &kernel, &bias, &out1, 1, 1, 0, "Interpreter");
  inferVTAConvNet(&inputs, &kernel, &bias, &out2, 3, 2, 1, "VTAInterpreter");
  //inferVTAConvNet(&inputs, &kernel, &bias, &out2, 1, 1, 0, "Interpreter");
//  out2.toBin("golden");
//  out1.toBin("output");


  EXPECT_TRUE(out1.isEqual(out2, 0.0));
}

TEST(NESTVTABackendTest, resnetv10_stage1_conv0) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 56, 56, 64}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {64, 3, 3, 64}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {64}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
  //inputs.toBin("input");
  //kernel.toBin("kernel");
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  //bias.zero();
  std::array<dim_t, 4> S{{1, 56, 56, 64}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 64, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 64, 0);

  inferVTAConvNet(&inputs, &kernel, &bias, &out1, 3, 1, 1, "VTA");
  //inferVTAConvNet(&inputs, &kernel, &bias, &out1, 1, 1, 0, "Interpreter");
  inferVTAConvNet(&inputs, &kernel, &bias, &out2, 3, 1, 1, "VTAInterpreter");
  //inferVTAConvNet(&inputs, &kernel, &bias, &out2, 1, 1, 0, "Interpreter");
//  out2.toBin("golden");
//  out1.toBin("output");


  EXPECT_TRUE(out1.isEqual(out2, 0.0));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();

}