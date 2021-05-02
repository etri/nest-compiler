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


TEST(inferVTALayoutConvReluNet, vta_layout_conv_test) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 4, 56, 56, 1, 16}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {4, 4, 3, 3, 16, 16}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {4, 16}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-1, 1, PRNG);
  // inputs.toBin("input");
  // kernel.toBin("kernel");
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
  // bias.zero();
  std::array<dim_t, 4> S{{1, 6, 6, 1}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 256, 0);
  Tensor out2(ElemKind::Int8QTy, {1, 4, 56, 56, 1, 16}, 256, 0);


  //inferVTAConvReluNet(&inputs, &kernel, &bias, &out1, 3, 1, 1, "Interpreter");
  inferVTALayoutConvNet(&inputs, &kernel, &bias, &out2, 3, 1, 1,
                      "VTA");

  TensorSerializationOptions opts;
  opts.withType = true;
  // dumpTensorToTextFile(out1, "./CPU.txt", opts);
  //dumpTensorToTextFile(out2, "./VTAInterpreter.txt", opts);

  //llvm::outs()<<out1.getHandle().raw(0)<<"\n";
  //llvm::outs()<<(double)out2.getHandle<int8_t>().raw(0)<<"\n";
  //EXPECT_TRUE(out1.getHandle().raw(0) == (double)out2.getHandle<int8_t>().raw(0));
  //EXPECT_TRUE(out2.isEqual(out1,1));
}


TEST(inferVTALayoutConvReluNet, VTAResnetTest1) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 64, 56, 56}, 1, 0);
  Tensor kernel(ElemKind::Int8QTy, {64, 64,3, 3}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {64}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-10, 10, PRNG);
  //inputs.toBin("input");
  //kernel.toBin("kernel");
  //bias.getHandle<int32_t>().randomize(0, 1, PRNG);
  bias.zero();
  std::array<dim_t, 4> S{{1, 56, 56, 64}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 256, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 256, 0);

  //inferVTAConvNet2(&inputs, &kernel, &bias, &out1, "VTA");
  //inferVTAConvNettemp(&inputs, &kernel, &bias, &out2, "Interpreter");
  //out1.toBin("golden");
  //EXPECT_TRUE(out1.isEqual(out2, 1.0));
}