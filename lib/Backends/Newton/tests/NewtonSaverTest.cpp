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

#include "Tensor.h"
#include <glow/Optimizer/GraphOptimizer/GraphOptimizer.h>
#include "glow/Backend/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"

#include "gtest/gtest.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "NewtonBackendTestUtils.h"

#include "llvm/ADT/StringMap.h"

#include "Newton.h"

using namespace glow;
static Placeholder *createPlaceholder(Module &mod,
                                      PlaceholderBindings &bindings,
                                      Tensor *tensor, llvm::StringRef name,
                                      const std::string layout = ANY_LAYOUT) {
  auto *P = mod.createPlaceholder(tensor->getElementType(), tensor->dims(),
                                  name, false, layout);
  auto *PTensor = bindings.allocate(P);
  PTensor->assign(tensor);

  return P;
}

static Constant *createConstant(Module &mod,
                                         PlaceholderBindings &bindings,
                                         Tensor *tensor,
                                         llvm::StringRef name) {
  auto *P = mod.createConstant(tensor->getElementType(), tensor->dims(), name);
  P->assign(tensor);

  return P;
}

TEST(NewtonSaverTest, fcTest1) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {1, 2048});
  Tensor filter(ElemKind::FloatTy, {2048, 1});
  Tensor bias(ElemKind::FloatTy, {1});

  inputs.getHandle<float>().randomize(-1, 1, PRNG);
  filter.getHandle<float>().randomize(-1, 1, PRNG);
  inputs.toBin("newtonFCTest1Input");
  bias.getHandle<float>().randomize(-1, 1, PRNG);

  std::array<dim_t, 2> S{{1, 1}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Constant *filterP;
  Constant *biasP;
  Placeholder *outP;
  TypeRef OT1;

  auto &outType1 = out1.getType();
  auto &inType = inputs.getType();
  auto &filterType = filter.getType();
  auto &biasType = bias.getType();

  inputP = createPlaceholder(
      M, bindings, &inputs, "inputP");

  filterP = createConstant(M, bindings, &filter, "filterP");
  biasP = createConstant(M, bindings, &bias, "biasP");
  outP = createPlaceholder(M, bindings, &out2, "outP");
  OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims());
//  auto *conv = F->createConv("conv", inputP, filterP, biasP, OT1, 3, 1, 1, 1);
  auto *fc = F->createFullyConnected("fc", inputP, filterP, biasP, OT1);

  F->createSave("ret", fc, outP);

  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("Newton"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "newtonFCTest1Bundle"; //file name
  llvm::StringRef mainEntryName = "newtonFCTest1MainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);
  inferFCNet(&inputs, &filter, &bias, &out1, "Interpreter");
  inferFCNet(&inputs, &filter, &bias, &out2, "Newton");
  out2.toBin("newtonFCTest1Golden");
  EXPECT_TRUE(out1.isEqual(out2, 1));
}

TEST(NewtonSaverTest, fcTest2) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {1, 2048});
  Tensor filter(ElemKind::FloatTy, {2048, 2048});
  Tensor bias(ElemKind::FloatTy, {2048});

  inputs.getHandle<float>().randomize(-1, 1, PRNG);
  filter.getHandle<float>().randomize(-1, 1, PRNG);
  inputs.toBin("newtonFCTest2Input");
  bias.getHandle<float>().randomize(-1, 1, PRNG);

  std::array<dim_t, 2> S{{1, 2048}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Constant *filterP;
  Constant *biasP;
  Placeholder *outP;
  TypeRef OT1;

  auto &outType1 = out1.getType();
  auto &inType = inputs.getType();
  auto &filterType = filter.getType();
  auto &biasType = bias.getType();

  inputP = createPlaceholder(
      M, bindings, &inputs, "inputP");

  filterP = createConstant(M, bindings, &filter, "filterP");
  biasP = createConstant(M, bindings, &bias, "biasP");
  outP = createPlaceholder(M, bindings, &out2, "outP");
  OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims());
//  auto *conv = F->createConv("conv", inputP, filterP, biasP, OT1, 3, 1, 1, 1);
  auto *fc = F->createFullyConnected("fc", inputP, filterP, biasP, OT1);

  F->createSave("ret", fc, outP);

  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("Newton"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "newtonFCTest2Bundle"; //file name
  llvm::StringRef mainEntryName = "newtonFCTest2MainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);
  inferFCNet(&inputs, &filter, &bias, &out1, "Interpreter");
  inferFCNet(&inputs, &filter, &bias, &out2, "Newton");
  out2.toBin("newtonFCTest2Golden");
  EXPECT_TRUE(out1.isEqual(out2, 2));
}


TEST(NewtonSaverTest, fcTest2_3) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {1, 512});
  Tensor filter(ElemKind::FloatTy, {512, 1000});
  Tensor bias(ElemKind::FloatTy, {1000});

  inputs.getHandle<float>().randomize(-1, 1, PRNG);
  filter.getHandle<float>().randomize(-1, 1, PRNG);
  inputs.toBin("newtonFCTest2_3Input");
  bias.getHandle<float>().randomize(-1, 1, PRNG);

  std::array<dim_t, 2> S{{1, 1000}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Constant *filterP;
  Constant *biasP;
  Placeholder *outP;
  TypeRef OT1;

  auto &outType1 = out1.getType();
  auto &inType = inputs.getType();
  auto &filterType = filter.getType();
  auto &biasType = bias.getType();

  inputP = createPlaceholder(
      M, bindings, &inputs, "inputP");

  filterP = createConstant(M, bindings, &filter, "filterP");
  biasP = createConstant(M, bindings, &bias, "biasP");
  outP = createPlaceholder(M, bindings, &out2, "outP");
  OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims());
//  auto *conv = F->createConv("conv", inputP, filterP, biasP, OT1, 3, 1, 1, 1);
  auto *fc = F->createFullyConnected("fc", inputP, filterP, biasP, OT1);

  F->createSave("ret", fc, outP);

  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("Newton"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "newtonFCTest2_3Bundle"; //file name
  llvm::StringRef mainEntryName = "newtonFCTest2_3MainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);
  inferFCNet(&inputs, &filter, &bias, &out1, "Interpreter");
  inferFCNet(&inputs, &filter, &bias, &out2, "Newton");
  out2.toBin("newtonFCTest2_3Golden");
  EXPECT_TRUE(out1.isEqual(out2, 2));
}



TEST(NewtonSaverTest, fcTest4) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {1, 1024});
  Tensor filter0(ElemKind::FloatTy, {1024, 2048});
  Tensor bias0(ElemKind::FloatTy, {2048});
  Tensor filter1(ElemKind::FloatTy, {2048, 1024});
  Tensor bias1(ElemKind::FloatTy, {1024});


  inputs.getHandle<float>().randomize(-0.001, 0.001, PRNG);
  inputs.toBin("newtonFCTest4Input");
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
  Tensor outNewton0(ElemKind::FloatTy, shape0);
  Tensor outNewton1(ElemKind::FloatTy, shape1);

  inferFCFCNet(&inputs, &filter0, &bias0, &filter1, &bias1, &outInter0, &outInter1, "Interpreter");
  inferFCFCNet(&inputs, &filter0, &bias0, &filter1, &bias1, &outNewton0, &outNewton1, "Newton");

  PlaceholderBindings bindings;
  Module mod;
  Function *F = mod.createFunction("main");
  Placeholder *inputP;
  Constant *filter0P;
  Constant *bias0P;
  Constant *filter1P;
  Constant *bias1P;
  Placeholder *outP;
  TypeRef OT0, OT1;
  inputP = createPlaceholder(mod, bindings, &inputs, "inputP");
  filter0P = createConstant(mod, bindings, &filter0, "filter0P");
  bias0P = createConstant(mod, bindings, &bias0, "bias0P");
  filter1P = createConstant(mod, bindings, &filter1, "filter1P");
  bias1P = createConstant(mod, bindings, &bias1, "bias1P");
  outP = createPlaceholder(mod, bindings, &outNewton1, "outP");
  OT0 = F->getParent()->uniqueType(outNewton0.getElementType(), outNewton0.dims());
  OT1 = F->getParent()->uniqueType(outNewton1.getElementType(), outNewton1.dims());

  auto *fc0 = F->createFullyConnected("fc0", inputP, filter0P, bias0P, OT0, 1);
  auto *fc1 = F->createFullyConnected("fc1", fc0, filter1P, bias1P, OT1, 1);
  F->createSave("ret", fc1, outP);

  F->dumpDAG("fcfc.dot");
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("Newton"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "newtonFCTest4Bundle"; //file name
  llvm::StringRef mainEntryName = "newtonFCTest4MainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);
  outNewton1.toBin("newtonFCTest4Golden");
  EXPECT_TRUE(outNewton1.isEqual(outInter1, 2));
}