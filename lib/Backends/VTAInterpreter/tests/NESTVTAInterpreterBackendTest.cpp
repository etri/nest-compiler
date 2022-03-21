/*****************************************************************************
	*
	* Copyright Next-Generation System Software Research Group, All rights reserved.
	* Future Computing Research Division, Artificial Intelligence Reserch Laboratory
	* Electronics and Telecommunications Research Institute (ETRI)
	*
	* THESE DOCUMENTS CONTAIN CONFIDENTIAL INFORMATION AND KNOWLEDGE
	* WHICH IS THE PROPERTY OF ETRI. NO PART OF THIS PUBLICATION IS
	* TO BE USED FOR ANY OTHER PURPOSE, AND THESE ARE NOT TO BE"
	* REPRODUCED, COPIED, DISCLOSED, TRANSMITTED, STORED IN A RETRIEVAL
	"* SYSTEM OR TRANSLATED INTO ANY OTHER HUMAN OR COMPUTER LANGUAGE,
	* IN ANY FORM, BY ANY MEANS, IN WHOLE OR IN PART, WITHOUT THE
	* COMPLETE PRIOR WRITTEN PERMISSION OF ETRI.
	*
	* LICENSE file : LICENSE_ETRI located in the top directory
	*
*****************************************************************************/

#include "VTAInterpreterBackendTestUtils.h"

#include "glow/Base/Tensor.h"
#include "glow/Base/TensorSerialization.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Random.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"

using namespace glow;
using llvm::cast;

// Test Conv ReLU Fusion
TEST(inferVTAConvReluNet, conv_relu_fusion_test) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 6, 6, 16}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {1, 3, 3, 16}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {1}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-1, 1, PRNG);
  // inputs.toBin("input");
  // kernel.toBin("kernel");
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  // bias.zero();
  std::array<dim_t, 4> S{{1, 6, 6, 1}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 256, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 256, 0);


  inferVTAConvReluNet(&inputs, &kernel, &bias, &out1, 3, 1, 1, "Interpreter");
  inferVTAConvReluNet(&inputs, &kernel, &bias, &out2, 3, 1, 1,
                      "VTAInterpreter");
  TensorSerializationOptions opts;
  opts.withType = true;
  // dumpTensorToTextFile(out1, "./CPU.txt", opts);
  dumpTensorToTextFile(out2, "./VTAInterpreter.txt", opts);

  //llvm::outs()<<out1.getHandle().raw(0)<<"\n";
  //llvm::outs()<<(double)out2.getHandle<int8_t>().raw(0)<<"\n";
  //EXPECT_TRUE(out1.getHandle().raw(0) == (double)out2.getHandle<int8_t>().raw(0));
  EXPECT_TRUE(out2.isEqual(out1,1));
}

// Test VTAConv (6dim) and VTAConvInterpreter(6dim)
TEST(inferVTALayoutConvReluNet, VTAResnetTest1){
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 56, 56, 64}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {64, 3, 3, 64}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {64}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
  inputs.toBin("./input_origin.bin");
  kernel.toBin("./kernel_origin.bin");
  //bias.getHandle<int32_t>().randomize(0, 1, PRNG);
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  //bias.getHandle<int32_t>().randomize(0, 0, PRNG);
  //bias.zero();
  bias.toBin("./bias_origin.bin");

  std::array<dim_t, 4> S{{1, 56, 56, 64}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 64, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 64, 0);

  inferVTAConvNet3(&inputs, &kernel, &bias, &out1, "VTA");
  out1.toBin("golden_vta");
  //inferVTAConvNettemp2(&inputs, &kernel, &bias, &out2, "VTAInterpreter"); // original code
  //out2.toBin("golden_vtainterpreter");
  inferVTAConvNet4(&inputs, &kernel, &bias, &out2, "VTAInterpreter");
  //inferVTAConvNet4(&inputs, &kernel, &bias, &out2, "VTAInterpreter");
  out2.toBin("golden_vtainterpreter");
  //inferVTAConvNettemp2(&inputs, &kernel, &bias, &out1, "VTA");
  //inferVTAConvNettemp2(&inputs, &kernel, &bias, &out2, "VTAInterpreter"); // original code
  TensorSerializationOptions opts;
  opts.withType = true;
  //dumpTensorToTextFile(out1, "./vtaconvtest.txt", opts);
  //dumpTensorToTextFile(out2, "./VTAInterpreter4vtaconvtest.txt", opts);
  //out1.toBin("golden");
  EXPECT_TRUE(out1.isEqual(out2));
}

// Test VTAConvFusion and VTAConvInterpreterFusion
TEST(inferVTALayoutConvReluNet, VTAConvReluFusionTest){
    PseudoRNG PRNG;
    Tensor inputs(ElemKind::Int8QTy, {1, 56, 56, 64}, 1, 0);
    Tensor kernel(ElemKind::Int8QTy, {64, 3, 3, 64}, 1, 0);
    Tensor bias(ElemKind::Int32QTy, {64}, 1, 0);
    inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
    kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
    inputs.toBin("./input_origin.bin");
    kernel.toBin("./kernel_origin.bin");
    bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
    bias.toBin("./bias_origin.bin");

    std::array<dim_t, 4> S{{1, 56, 56, 64}};
    llvm::ArrayRef<dim_t> shape(S);
    Tensor out1(ElemKind::Int8QTy, shape, 64, 0);
    Tensor out2(ElemKind::Int8QTy, shape, 64, 0);

    inferVTAConvReluFusionNet(&inputs, &kernel, &bias, &out1, "VTA");
    out1.toBin("golden_vta");
    inferVTAConvReluFusionNet2(&inputs, &kernel, &bias, &out2, "VTAInterpreter");
    out2.toBin("golden_vtainterpreter");
    TensorSerializationOptions opts;
    opts.withType = true;
    EXPECT_TRUE(out1.isEqual(out2));
}

// transformPostLoweirng.optimizeVTAConv: 6dim VTAConv graph optimization testing
TEST(inferVTALayoutConvReluNet, VTAConvGraphOptzTest){
    PseudoRNG PRNG;
    Tensor inputs(ElemKind::Int8QTy, {1, 56, 56, 64}, 1, 0);
    Tensor kernel(ElemKind::Int8QTy, {64, 3, 3, 64}, 1, 0);
    Tensor bias(ElemKind::Int32QTy, {64}, 1, 0);
    inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
    kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
    inputs.toBin("./input_origin.bin");
    kernel.toBin("./kernel_origin.bin");
    bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
    bias.toBin("./bias_origin.bin");
    // bias.zero();
    std::array<dim_t, 4> S{{1, 56, 56, 64}};
    llvm::ArrayRef<dim_t> shape(S);
    Tensor out1(ElemKind::Int8QTy, shape, 64, 0);
    Tensor out2(ElemKind::Int8QTy, shape, 64, 0);

    inferVTAConvReluNet2(&inputs, &kernel, &bias, &out1, 3, 1, 1, "VTA");
    out1.toBin("golden_vta_graphoptz");
    inferVTAConvReluNet2(&inputs, &kernel, &bias, &out2, 3, 1, 1, "VTAInterpreter");
    out2.toBin("golden_vtainterpreter_graphoptz");
    EXPECT_TRUE(out2.isEqual(out1));
}