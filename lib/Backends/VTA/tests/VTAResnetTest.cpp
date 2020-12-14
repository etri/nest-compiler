#include "gtest/gtest.h"
#include "VTABackendTestUtils.h"
#include "glow/Graph/Graph.h"
#include "glow/Support/Random.h"
#include "llvm/ADT/STLExtras.h"

using namespace glow;
TEST(VTAResnetTest, VTAResnetTest1) {
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

  inferVTAConvNet2(&inputs, &kernel, &bias, &out1, "VTA");
  inferVTAConvNettemp(&inputs, &kernel, &bias, &out2, "Interpreter");
  //out1.toBin("golden");
  EXPECT_TRUE(out1.isEqual(out2, 1.0));
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();

}