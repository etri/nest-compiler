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

#include "gtest/gtest.h"
#include "EnlightBackendTestUtils.h"
#include "glow/Graph/Graph.h"
#include "glow/Support/Random.h"
#include "llvm/ADT/STLExtras.h"

using namespace glow;
TEST(EnlightOpTest, fcTest1) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {1, 2048});
  Tensor filter(ElemKind::FloatTy, {2048, 1});
  Tensor bias(ElemKind::FloatTy, {1});

  inputs.getHandle<float>().randomize(-1, 1, PRNG);
  filter.getHandle<float>().randomize(-1, 1, PRNG);
  inputs.toBin("newtonFcTestInput");
  bias.getHandle<float>().randomize(-1, 1, PRNG);

  std::array<dim_t, 2> S{{1, 1}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);
  inferFCNet(&inputs, &filter, &bias, &out1, "Interpreter");
  inferFCNet(&inputs, &filter, &bias, &out2, "Enlight");
  //out1.toBin("golden");
  EXPECT_TRUE(out1.isEqual(out2, 1));
}

TEST(EnlightOpTest, fcTest2) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {1, 2048});
  Tensor filter(ElemKind::FloatTy, {2048, 2048});
  Tensor bias(ElemKind::FloatTy, {2048});

  inputs.getHandle<float>().randomize(-1, 1, PRNG);
  filter.getHandle<float>().randomize(-1, 1, PRNG);
  inputs.toBin("newtonFcTestInput");
  bias.getHandle<float>().randomize(-1, 1, PRNG);

  std::array<dim_t, 2> S{{1, 2048}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);
  inferFCNet(&inputs, &filter, &bias, &out1, "Interpreter");
  inferFCNet(&inputs, &filter, &bias, &out2, "Enlight");
  //out1.toBin("golden");
  EXPECT_TRUE(out1.isEqual(out2, 2));
}

TEST(EnlightOpTest, fcTest2_2) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {1, 24});
  Tensor filter(ElemKind::FloatTy, {24, 12});
  Tensor bias(ElemKind::FloatTy, {12});

  inputs.getHandle<float>().randomize(-1, 1, PRNG);
  filter.getHandle<float>().randomize(-1, 1, PRNG);
  inputs.toBin("newtonFcTestInput");
  bias.getHandle<float>().randomize(-1, 1, PRNG);

  std::array<dim_t, 2> S{{1, 12}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);
  inferFCNet(&inputs, &filter, &bias, &out1, "Interpreter");
  inferFCNet(&inputs, &filter, &bias, &out2, "Enlight");
  //out1.toBin("golden");
  EXPECT_TRUE(out1.isEqual(out2, 1));
}

TEST(EnlightOpTest, fcTest2_3) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {1, 512});
  Tensor filter(ElemKind::FloatTy, {512, 1000});
  Tensor bias(ElemKind::FloatTy, {1000});

  inputs.getHandle<float>().randomize(-1, 1, PRNG);
  filter.getHandle<float>().randomize(-1, 1, PRNG);
  inputs.toBin("newtonFcTestInput");
  bias.getHandle<float>().randomize(-1, 1, PRNG);

  std::array<dim_t, 2> S{{1, 1000}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);
  inferFCNet(&inputs, &filter, &bias, &out1, "Interpreter");
  inferFCNet(&inputs, &filter, &bias, &out2, "Enlight");
  //out1.toBin("golden");
  EXPECT_TRUE(out1.isEqual(out2, 2));
}


TEST(EnlightOpTest, fcTest3) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {2048, 2048});
  Tensor filter(ElemKind::FloatTy, {2048, 1});
  Tensor bias(ElemKind::FloatTy, {1});

  inputs.getHandle<float>().randomize(-1, 1, PRNG);
  filter.getHandle<float>().randomize(-1, 1, PRNG);
  inputs.toBin("newtonFcTestInput");
  bias.getHandle<float>().randomize(-1, 1, PRNG);

  std::array<dim_t, 2> S{{2048, 1}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);
  inferFCNet(&inputs, &filter, &bias, &out1, "Interpreter");
  inferFCNet(&inputs, &filter, &bias, &out2, "Enlight");
  //out1.toBin("golden");
  EXPECT_TRUE(out1.isEqual(out2, 2));
}

TEST(EnlightOpTest, fcTest4) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {1, 1024});
  Tensor filter0(ElemKind::FloatTy, {1024, 2048});
  Tensor bias0(ElemKind::FloatTy, {2048});
  Tensor filter1(ElemKind::FloatTy, {2048, 1024});
  Tensor bias1(ElemKind::FloatTy, {1024});


  inputs.getHandle<float>().randomize(-0.001, 0.001, PRNG);
  inputs.toBin("newtonFcTestInput");
  filter0.getHandle<float>().randomize(-1, 1, PRNG);
  bias0.getHandle<float>().randomize(-1, 1, PRNG);
  filter1.getHandle<float>().randomize(-1, 1, PRNG);
  bias1.getHandle<float>().randomize(-1, 1, PRNG);

  std::array<dim_t, 2> S0{{1, 2048}};
  llvm::ArrayRef<dim_t> shape0(S0);
  std::array<dim_t, 2> S1{{1, 1024}};
  llvm::ArrayRef<dim_t> shape1(S1);
  Tensor outInter0(ElemKind::FloatTy, shape0);
  Tensor outInter1(ElemKind::FloatTy, shape1);
  Tensor outEnlight0(ElemKind::FloatTy, shape0);
  Tensor outEnlight1(ElemKind::FloatTy, shape1);

  inferFCFCNet(&inputs, &filter0, &bias0, &filter1, &bias1, &outInter0, &outInter1, "Interpreter");
  inferFCFCNet(&inputs, &filter0, &bias0, &filter1, &bias1, &outEnlight0, &outEnlight1, "Enlight");

  //out1.toBin("golden");
  EXPECT_TRUE(outEnlight1.isEqual(outInter1, 2));
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();

}