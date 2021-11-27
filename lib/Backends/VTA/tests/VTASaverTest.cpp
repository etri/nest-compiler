#include "Tensor.h"
#include <glow/Optimizer/GraphOptimizer/GraphOptimizer.h>
#include "glow/Backend/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"

#include "gtest/gtest.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "VTABackendTestUtils.h"

#include "llvm/ADT/StringMap.h"

#include "VTA.h"

using namespace glow;
static Placeholder *createQuantizedPlaceholder(Module &mod,
                                               PlaceholderBindings &bindings,
                                               Tensor *tensor, float scale,
                                               int32_t offset,
                                               llvm::StringRef name) {
    auto *P = mod.createPlaceholder(tensor->getElementType(), tensor->dims(),
                                    scale, offset, name, false);
    auto *PTensor = bindings.allocate(P);
    PTensor->assign(tensor);

    return P;
}

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


static Constant *createQuantizedConstant(Module &mod,
                                               PlaceholderBindings &bindings,
                                               Tensor *tensor, float scale,
                                               int32_t offset,
                                               llvm::StringRef name) {
  auto *P = mod.createConstant(tensor->getElementType(), tensor->dims(),
                                  scale, offset, name);
  P->assign(tensor);

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
TEST(VTASaverTest, convTest) {
    PseudoRNG PRNG;
    Tensor inputs(ElemKind::Int8QTy, {1, 14, 14, 16}, 1, 0);
    Tensor filter(ElemKind::Int8QTy, {16, 3, 3, 16}, 1, 0);
    Tensor filter2(ElemKind::Int8QTy, {16, 3, 3, 16}, 1, 0);
    Tensor bias(ElemKind::Int32QTy, {16}, 1, 0);
    Tensor bias2(ElemKind::Int32QTy, {16}, 256, 0);
    inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
    filter.getHandle<int8_t>().randomize(-10, 10, PRNG);
    filter2.getHandle<int8_t>().randomize(-10, 10, PRNG);
    //filter.toBin("filter1");
    //filter2.toBin("filter2");
    inputs.toBin("vtaConvTestInput");
    bias.getHandle<int32_t>().randomize(0, 32768, PRNG);
    bias2.getHandle<int32_t>().randomize(0, 32768, PRNG);
    //bias.toBin("bias1");
    //bias2.toBin("bias2");

  std::array<dim_t, 4> S{{1, 14, 14, 16}};
    llvm::ArrayRef<dim_t> shape(S);
    Tensor out1(ElemKind::Int8QTy, shape, 256, 0);
    Tensor out2(ElemKind::Int8QTy, shape, 65536, 0);

    //inferVTAConvNet(&inputs, &kernel, &bias, &out1, backendName_);


    PlaceholderBindings bindings;
    Module M;
    Function *F = M.createFunction("main");
    Placeholder *inputP;
    Constant *filterP;
    Constant *biasP;
    Constant *filter2P;
    Constant *bias2P;
    Placeholder *outP;
    TypeRef OT1, OT2;

    auto &outType1 = out1.getType();
    auto &outType2 = out2.getType();
    auto &inType = inputs.getType();
    auto &filterType = filter.getType();
    auto &biasType = bias.getType();
    auto &filter2Type = filter2.getType();
    auto &bias2Type = bias2.getType();

    inputP = createQuantizedPlaceholder(
            M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");

    filterP = createQuantizedConstant(M, bindings, &filter, filterType.getScale(),
                                       filterType.getOffset(), "filterP");
    biasP = createQuantizedConstant(M, bindings, &bias, biasType.getScale(),
                                       biasType.getOffset(), "biasP");
    filter2P =
        createQuantizedConstant(M, bindings, &filter2, filter2Type.getScale(),
                                       filter2Type.getOffset(), "filter2P");
    bias2P = createQuantizedConstant(M, bindings, &bias2, bias2Type.getScale(),
                                        bias2Type.getOffset(), "bias2P");
    outP = createQuantizedPlaceholder(M, bindings, &out2, outType2.getScale(),
                                      outType2.getOffset(), "outP");
    OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                     outType1.getScale(), outType1.getOffset());
    OT2 = F->getParent()->uniqueType(out2.getElementType(), out2.dims(),
                                     outType2.getScale(), outType2.getOffset());

    auto *conv = F->createConv("conv", inputP, filterP, biasP, OT1, 3, 1, 1, 1);
    auto *conv2 = F->createConv("conv2", conv, filter2P, bias2P, OT2, 3, 1, 1, 1);

    F->createSave("ret", conv2, outP);

    std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
    F->dump();
    std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
    //Optimize & Lowering
    std::unique_ptr<Backend> backend(createBackend("VTA"));
    CompilationContext cctx;
    auto error = ::glow::optimizeFunction(F, *backend, cctx);

    EXIT_ON_ERR(std::move(error));
    // Save bundle.
    llvm::StringRef outputDir = ".";
    llvm::StringRef bundleName = "vtaConvTestBundle"; //file name
    llvm::StringRef mainEntryName = "vtaConvTestMainEntry"; //main function name

    backend->save(F, outputDir, bundleName, mainEntryName);
    Tensor outVTA(ElemKind::Int8QTy, shape, 256, 0);
    Tensor outVTA2(ElemKind::Int8QTy, shape, 65536, 0);
    inferVTAMultiConvNet(&inputs, &filter, &bias, &filter2, &bias2, &outVTA, &outVTA2, "VTA");
    outVTA2.toBin("vtaConvTestGolden");
}

TEST(VTASaverTest, resnetv10S1C0Test) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 56, 56, 64}, 1, 0);
  Tensor filter(ElemKind::Int8QTy, {64, 3, 3, 64}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {64}, 1, 0);

  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  filter.getHandle<int8_t>().randomize(-10, 10, PRNG);
  inputs.toBin("vtaResnetv10S1C0TestInput");
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);

  std::array<dim_t, 4> S{{1, 56, 56, 64}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 64, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 64, 0);


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

  inputP = createQuantizedPlaceholder(
          M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");

  filterP = createQuantizedConstant(M, bindings, &filter, filterType.getScale(),
                                    filterType.getOffset(), "filterP");
  biasP = createQuantizedConstant(M, bindings, &bias, biasType.getScale(),
                                  biasType.getOffset(), "biasP");
  outP = createQuantizedPlaceholder(M, bindings, &out2, outType1.getScale(),
                                    outType1.getOffset(), "outP");
  OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                   outType1.getScale(), outType1.getOffset());
  auto *conv = F->createConv("conv", inputP, filterP, biasP, OT1, 3, 1, 1, 1);

  F->createSave("ret", conv, outP);

  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaResnetv10S1C0TestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaResnetv10S1C0TestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);
  inferVTAConvNet(&inputs, &filter, &bias, &out1, 3, 1, 1, "VTA");
  inferVTAConvNet(&inputs, &filter, &bias, &out2, 3, 1, 1, "VTAInterpreter");
  out1.toBin("vtaResnetv10S1C0TestGolden");
  EXPECT_TRUE(out1.isEqual(out2, 0.0));
}

TEST(VTASaverTest, multievtaConvTest) {
  for(int i = 0; i < 4; i++) {
    PseudoRNG PRNG;
    std::string inputname = "multievtaConvTestInput"+std::to_string(i);
    std::string outputname = "multievtaConvTestGolden"+std::to_string(i);
    std::string bundlename = "multievtaConvTestBundle"+std::to_string(i);
    std::string mainentryname = "multievtaConvTestMainEntry"+std::to_string(i);

    Tensor inputs(ElemKind::Int8QTy, {1, 14, 14, 32}, 1, 0);
    Tensor filter(ElemKind::Int8QTy, {32, 3, 3, 32}, 1, 0);
    Tensor bias(ElemKind::Int32QTy, {32}, 1, 0);

    inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
    filter.getHandle<int8_t>().randomize(-10, 10, PRNG);
    inputs.toBin(inputname.c_str());
    bias.getHandle<int32_t>().randomize(0, 32768, PRNG);

    std::array<dim_t, 4> S{{1, 14, 14, 32}};
    llvm::ArrayRef<dim_t> shape(S);
    Tensor out1(ElemKind::Int8QTy, shape, 64, 0);
    Tensor out2(ElemKind::Int8QTy, shape, 64, 0);

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

    std::string inputPname= "inputP" +std::to_string(i);
    std::string filterPname= "filterP" +std::to_string(i);
    std::string biasPname= "biasP" +std::to_string(i);
    std::string outPname= "outP" +std::to_string(i);
    inputP = createQuantizedPlaceholder(
        M, bindings, &inputs, inType.getScale(), inType.getOffset(), inputPname.c_str());

    filterP = createQuantizedConstant(M, bindings, &filter, filterType.getScale(),
                                      filterType.getOffset(), filterPname.c_str());
    biasP = createQuantizedConstant(M, bindings, &bias, biasType.getScale(),
                                    biasType.getOffset(), biasPname.c_str());
    outP = createQuantizedPlaceholder(M, bindings, &out2, outType1.getScale(),
                                      outType1.getOffset(), outPname.c_str());
    OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                     outType1.getScale(), outType1.getOffset());
    auto *conv = F->createConv("conv", inputP, filterP, biasP, OT1, 3, 1, 1, 1);

    F->createSave("ret", conv, outP);

    //Optimize & Lowering
    std::unique_ptr<Backend> backend(createBackend("VTA"));
    static_cast<VTA*>(backend.get())->setIdxMultiEVTA(1<<i);
    CompilationContext cctx;
    auto error = ::glow::optimizeFunction(F, *backend, cctx);

    EXIT_ON_ERR(std::move(error));
    // Save bundle.
    llvm::StringRef outputDir = ".";
    llvm::StringRef bundleName = bundlename.c_str(); //file name
    llvm::StringRef mainEntryName = mainentryname.c_str(); //main function name

    backend->save(F, outputDir, bundleName, mainEntryName);
    inferVTAConvNet(&inputs, &filter, &bias, &out1, 3, 1, 1, "VTA");
    inferVTAConvNet(&inputs, &filter, &bias, &out2, 3, 1, 1, "VTAInterpreter");
    out1.toBin(outputname.c_str());
  }
}


TEST(VTASaverTest, convTest2) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 14, 14, 32}, 1, 0);
  Tensor filter(ElemKind::Int8QTy, {32, 3, 3, 32}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {32}, 1, 0);

  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  filter.getHandle<int8_t>().randomize(-10, 10, PRNG);
  inputs.toBin("vtaConv2TestInput");
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);

  std::array<dim_t, 4> S{{1, 14, 14, 32}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 64, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 64, 0);


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

  inputP = createQuantizedPlaceholder(
          M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");

  filterP = createQuantizedConstant(M, bindings, &filter, filterType.getScale(),
                                    filterType.getOffset(), "filterP");
  biasP = createQuantizedConstant(M, bindings, &bias, biasType.getScale(),
                                  biasType.getOffset(), "biasP");
  outP = createQuantizedPlaceholder(M, bindings, &out2, outType1.getScale(),
                                    outType1.getOffset(), "outP");
  OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                   outType1.getScale(), outType1.getOffset());
  auto *conv = F->createConv("conv", inputP, filterP, biasP, OT1, 3, 1, 1, 1);

  F->createSave("ret", conv, outP);

  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaConv2TestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaConv2TestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);
  inferVTAConvNet(&inputs, &filter, &bias, &out1, 3, 1, 1, "VTA");
  inferVTAConvNet(&inputs, &filter, &bias, &out2, 3, 1, 1, "VTAInterpreter");
  out1.toBin("vtaConv2TestGolden");
  out2.toBin("vtainterConv2TestGolden");
  EXPECT_TRUE(out1.isEqual(out2, 0.0));
}

TEST(VTASaverTest, convTest3) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 14, 14, 16}, 1, 0);
  Tensor filter(ElemKind::Int8QTy, {32, 3, 3, 16}, 1, 0);
  Tensor bias(ElemKind::Int32QTy, {32}, 1, 0);

  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  filter.getHandle<int8_t>().randomize(-10, 10, PRNG);
  inputs.toBin("vtaConv3TestInput");
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);

  std::array<dim_t, 4> S{{1, 14, 14, 32}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 64, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 64, 0);


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

  inputP = createQuantizedPlaceholder(
          M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");

  filterP = createQuantizedConstant(M, bindings, &filter, filterType.getScale(),
                                    filterType.getOffset(), "filterP");
  biasP = createQuantizedConstant(M, bindings, &bias, biasType.getScale(),
                                  biasType.getOffset(), "biasP");
  outP = createQuantizedPlaceholder(M, bindings, &out2, outType1.getScale(),
                                    outType1.getOffset(), "outP");
  OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                   outType1.getScale(), outType1.getOffset());
  auto *conv = F->createConv("conv", inputP, filterP, biasP, OT1, 3, 1, 1, 1);

  F->createSave("ret", conv, outP);

  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaConv3TestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaConv3TestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);
  inferVTAConvNet(&inputs, &filter, &bias, &out1, 3, 1, 1, "VTA");
  inferVTAConvNet(&inputs, &filter, &bias, &out2, 3, 1, 1, "VTAInterpreter");
  out1.toBin("vtaConv3TestGolden");
  EXPECT_TRUE(out1.isEqual(out2, 0.0));
}

TEST(VTASaverTest, convTest4) {
    PseudoRNG PRNG;
    Tensor inputs(ElemKind::Int8QTy, {1, 56, 56, 64}, 1, 0);
    Tensor filter(ElemKind::Int8QTy, {64, 3, 3, 64}, 1, 0);
    Tensor bias(ElemKind::Int32QTy, {64}, 1, 0);

    inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
    filter.getHandle<int8_t>().randomize(-10, 10, PRNG);
    inputs.toBin("vtaConv4TestInput");
    bias.getHandle<int32_t>().randomize(0, 32768, PRNG);

    std::array<dim_t, 4> S{{1, 56, 56, 64}};
    llvm::ArrayRef<dim_t> shape(S);
    Tensor out1(ElemKind::Int8QTy, shape, 64, 0);
    Tensor out2(ElemKind::Int8QTy, shape, 64, 0);


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

    inputP = createQuantizedPlaceholder(
            M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");

    filterP = createQuantizedConstant(M, bindings, &filter, filterType.getScale(),
                                      filterType.getOffset(), "filterP");
    biasP = createQuantizedConstant(M, bindings, &bias, biasType.getScale(),
                                    biasType.getOffset(), "biasP");
    outP = createQuantizedPlaceholder(M, bindings, &out2, outType1.getScale(),
                                      outType1.getOffset(), "outP");
    OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                     outType1.getScale(), outType1.getOffset());
    auto *conv = F->createConv("conv", inputP, filterP, biasP, OT1, 3, 1, 1, 1);

    F->createSave("ret", conv, outP);

    //Optimize & Lowering
    std::unique_ptr<Backend> backend(createBackend("VTA"));
    CompilationContext cctx;
    auto error = ::glow::optimizeFunction(F, *backend, cctx);

    EXIT_ON_ERR(std::move(error));
    // Save bundle.
    llvm::StringRef outputDir = ".";
    llvm::StringRef bundleName = "vtaConv4TestBundle"; //file name
    llvm::StringRef mainEntryName = "vtaConv4TestMainEntry"; //main function name

    backend->save(F, outputDir, bundleName, mainEntryName);
    inferVTAConvNet(&inputs, &filter, &bias, &out1, 3, 1, 1, "VTA");
    inferVTAConvNet(&inputs, &filter, &bias, &out2, 3, 1, 1, "VTAInterpreter");
    out1.toBin("vtaConv4TestGolden");
    EXPECT_TRUE(out1.isEqual(out2, 0.0));
}

TEST(VTASaverTest, convTest5) {
    PseudoRNG PRNG;
    Tensor inputs(ElemKind::Int8QTy, {1, 56, 56, 64}, 1, 0);
    Tensor filter(ElemKind::Int8QTy, {128, 1, 1, 64}, 1, 0);
    Tensor bias(ElemKind::Int32QTy, {128}, 1, 0);

    inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
    filter.getHandle<int8_t>().randomize(-10, 10, PRNG);
    inputs.toBin("vtaConv5TestInput");
    bias.getHandle<int32_t>().randomize(0, 32768, PRNG);

    std::array<dim_t, 4> S{{1, 28, 28, 128}};
    llvm::ArrayRef<dim_t> shape(S);
    Tensor out1(ElemKind::Int8QTy, shape, 64, 0);
    Tensor out2(ElemKind::Int8QTy, shape, 64, 0);


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

    inputP = createQuantizedPlaceholder(
            M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");

    filterP = createQuantizedConstant(M, bindings, &filter, filterType.getScale(),
                                      filterType.getOffset(), "filterP");
    biasP = createQuantizedConstant(M, bindings, &bias, biasType.getScale(),
                                    biasType.getOffset(), "biasP");
    outP = createQuantizedPlaceholder(M, bindings, &out2, outType1.getScale(),
                                      outType1.getOffset(), "outP");
    OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                     outType1.getScale(), outType1.getOffset());
    auto *conv = F->createConv("conv", inputP, filterP, biasP, OT1, 1, 2, 0, 1);

    F->createSave("ret", conv, outP);

    //Optimize & Lowering
    std::unique_ptr<Backend> backend(createBackend("VTA"));
    CompilationContext cctx;
    auto error = ::glow::optimizeFunction(F, *backend, cctx);

    EXIT_ON_ERR(std::move(error));
    // Save bundle.
    llvm::StringRef outputDir = ".";
    llvm::StringRef bundleName = "vtaConv5TestBundle"; //file name
    llvm::StringRef mainEntryName = "vtaConv5TestMainEntry"; //main function name

    backend->save(F, outputDir, bundleName, mainEntryName);
    inferVTAConvNet(&inputs, &filter, &bias, &out1, 1, 2, 0, "VTA");
    inferVTAConvNet(&inputs, &filter, &bias, &out2, 1, 2, 0, "VTAInterpreter");
    out1.toBin("vtaConv5TestGolden");
    EXPECT_TRUE(out1.isEqual(out2, 0.0));
}

void conv_test_template(dim_t N, dim_t H, dim_t W, dim_t C, float input_scale,
                        dim_t KN, dim_t KH, dim_t KW, dim_t KC, float filter_scale,
                        //dim_t ON, dim_t OH, dim_t OW, dim_t OC, float output_scale,
                        float bias_scale, int stride, int pad, int group,
                        int input_low, int input_high, int filter_low, int filter_high,
                        int bias_low, int bias_high,
                        std::string test_name, Tensor& out1, Tensor& out2){
  assert(KH==KW);
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {N, H, W, C}, input_scale, 0);
  Tensor filter(ElemKind::Int8QTy, {KN, KH, KW, KC}, filter_scale, 0);
  Tensor bias(ElemKind::Int32QTy, {KN}, bias_scale, 0);

  inputs.getHandle<int8_t>().randomize(input_low, input_high, PRNG);
  filter.getHandle<int8_t>().randomize(filter_low, filter_high, PRNG);
  std::string input_bin_str = test_name + "Input";
  inputs.toBin(input_bin_str.c_str());
  bias.getHandle<int32_t>().randomize(bias_low, bias_high, PRNG);

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

  inputP = createQuantizedPlaceholder(
      M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");

  filterP = createQuantizedConstant(M, bindings, &filter, filterType.getScale(),
                                    filterType.getOffset(), "filterP");
  biasP = createQuantizedConstant(M, bindings, &bias, biasType.getScale(),
                                  biasType.getOffset(), "biasP");
  outP = createQuantizedPlaceholder(M, bindings, &out2, outType1.getScale(),
                                    outType1.getOffset(), "outP");
  OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                   outType1.getScale(), outType1.getOffset());
  auto *conv = F->createConv("conv", inputP, filterP, biasP, OT1, KH, stride, pad, group);

  F->createSave("ret", conv, outP);

  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  std::string sbundleName = test_name + "Bundle"; //file name
  llvm::StringRef bundleName = sbundleName; //file name
  std::string smainEntryName = test_name + "MainEntry";
  llvm::StringRef mainEntryName = smainEntryName;

  backend->save(F, outputDir, bundleName, mainEntryName);
  inferVTAConvNet(&inputs, &filter, &bias, &out1, KH, stride, pad, "VTA");
  inferVTAConvNet(&inputs, &filter, &bias, &out2, KH, stride, pad, "VTAInterpreter");
  std::string output_bin_str = test_name + "Golden";
  out1.toBin(output_bin_str.c_str());
}

TEST(VTASaverTest, Conv_inception_5b_5x5_1__1) {

  std::array<dim_t, 4> S{{1, 7, 7, 128}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 512, 0);
  Tensor out2(ElemKind::Int8QTy, shape, 512, 0);

  conv_test_template(1, 7, 7, 48, 1,
                     128, 5, 5, 48, 1,
                     1,
                     1, 2, 1, 0, 10, -10, 10, 0, 32768, "Conv_inception_5b_5x5_1__1", out1, out2
  );

  EXPECT_TRUE(out1.isEqual(out2, 0.0));
}



TEST(VTASaverTest, conv_gpu_0_conv1) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {1, 224, 224, 3});
  Tensor filter(ElemKind::FloatTy, {64, 7, 7, 3});
  Tensor bias(ElemKind::FloatTy, {64});
  inputs.getHandle().randomize(-1, 1, PRNG);
  filter.getHandle().randomize(-1, 1, PRNG);
  //filter.toBin("vtaFloatConvTestBundle.weights0");
  inputs.toBin("vtaFloatConvTestInput");
  bias.getHandle().randomize(-1, 1, PRNG);
  //bias.zero();
  //bias.toBin("vtaFloatConvTestBundle.weights1");

  Tensor out(ElemKind::FloatTy, {1, 112, 112, 64});


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Constant *filterP;
  Constant *biasP;
  Placeholder *outP;
  TypeRef OT;

  auto &outType = out.getType();
  auto &inType = inputs.getType();
  auto &filterType = filter.getType();
  auto &biasType = bias.getType();

  inputP = createPlaceholder(
      M, bindings, &inputs, "inputP");

  filterP = createConstant(M, bindings, &filter, "filterP");
  biasP = createConstant(M, bindings, &bias, "biasP");
  outP = createPlaceholder(M, bindings, &out, "outP");
  OT = F->getParent()->uniqueType(out.getElementType(), out.dims());

  auto *conv = F->createConv("conv", inputP, filterP, biasP, OT, 7, 2, 3, 1);

  F->createSave("ret", conv, outP);

  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaFloatConvTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaFloatConvTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);
  Tensor outVTA(ElemKind::FloatTy, {1, 112, 112, 64});
  inferFloatConvNet(&inputs, &filter, &bias, &outVTA, "VTAInterpreter");
  outVTA.toBin("vtaFloatConvTestGolden");
}

TEST(VTASaverTest, quantized_conv_gpu_0_conv1) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 224, 224, 3}, 1.0/32, 0);
  Tensor filter(ElemKind::Int8QTy, {64, 7, 7, 3}, 1.0/64, 0);
  Tensor bias(ElemKind::Int32QTy, {64}, 1.0/2048, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  filter.getHandle<int8_t>().randomize(0, 60, PRNG);
  filter.toBin("vtaNonVTAConvTestBundle.weights0");
  inputs.toBin("vtaNonVTAConvTestInput");
  bias.getHandle<int32_t>().randomize(0, 30, PRNG);
  //bias.zero();
  bias.toBin("vtaNonVTAConvTestBundle.weights1");

  Tensor out(ElemKind::Int8QTy, {1, 112, 112, 64}, 1.0/8, 0);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Constant *filterP;
  Constant *biasP;
  Placeholder *outP;
  TypeRef OT;

  auto &outType = out.getType();
  auto &inType = inputs.getType();
  auto &filterType = filter.getType();
  auto &biasType = bias.getType();

  inputP = createQuantizedPlaceholder(
      M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");

  filterP = createQuantizedConstant(M, bindings, &filter, filterType.getScale(), filterType.getOffset(),"filterP");
  biasP = createQuantizedConstant(M, bindings, &bias, biasType.getScale(), biasType.getOffset(),"biasP");
  outP = createQuantizedPlaceholder(M, bindings, &out, outType.getScale(),
                                    outType.getOffset(), "outP");
  OT = F->getParent()->uniqueType(out.getElementType(), out.dims(), outType.getScale(), outType.getOffset());

  auto *conv = F->createConv("conv", inputP, filterP, biasP, OT, 7, 2, 3, 1);

  F->createSave("ret", conv, outP);

  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaNonVTAConvTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaNonVTAConvTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);
  Tensor outVTA(ElemKind::Int8QTy, {1, 112, 112, 64}, 1.0/8, 0);
  inferQuantizedConvNet(&inputs, &filter, &bias, &outVTA, "VTAInterpreter");
  outVTA.toBin("vtaNonVTAConvTestGolden");
}

TEST(VTASaverTest, fullyconnected) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 2048}, 1.0/8, 0);
  Tensor filter(ElemKind::Int8QTy, {2048, 1000}, 1.0/128, 0);
  Tensor bias(ElemKind::Int32QTy, {1000}, 1.0/1024, 0);
  inputs.getHandle<int8_t>().randomize(-10, 10, PRNG);
  filter.getHandle<int8_t>().randomize(-3, 3, PRNG);
  filter.toBin("vtaFCTestBundle.weights0");
  inputs.toBin("vtaFCTestInput");
  bias.getHandle<int32_t>().randomize(0, 30, PRNG);
  //bias.zero();
  bias.toBin("vtaFCTestBundle.weights1");

  Tensor out(ElemKind::Int8QTy, {1, 1000}, 1.0/4, 0);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Constant *filterP;
  Constant *biasP;
  Placeholder *outP;
  TypeRef OT;

  auto &outType = out.getType();
  auto &inType = inputs.getType();
  auto &filterType = filter.getType();
  auto &biasType = bias.getType();

  inputP = createQuantizedPlaceholder(
      M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");

  filterP = createQuantizedConstant(M, bindings, &filter, filterType.getScale(), filterType.getOffset(),"filterP");
  biasP = createQuantizedConstant(M, bindings, &bias, biasType.getScale(), biasType.getOffset(),"biasP");
  outP = createQuantizedPlaceholder(M, bindings, &out, outType.getScale(),
                                    outType.getOffset(), "outP");
  OT = F->getParent()->uniqueType(out.getElementType(), out.dims(), outType.getScale(), outType.getOffset());

  auto *fc = F->createFullyConnected("fc", inputP, filterP, biasP, OT, 1);
  F->createSave("ret", fc, outP);


  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaFCTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaFCTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);
  Tensor outVTA(ElemKind::Int8QTy, {1, 1000}, 1.0/4, 0);
  inferFCNet(&inputs, &filter, &bias, &outVTA, "VTAInterpreter");
  outVTA.toBin("vtaFCTestGolden");
}


TEST(VTASaverTest, maxPoolTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 14, 14, 16}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  inputs.toBin("vtaMaxPoolTestInput");

  unsigned_t kernel = 2;
  unsigned_t stride = 2;
  unsigned_t pad = 1;

  //inputs.toBin("input");
  std::array<dim_t, 4> S{{1, 8, 8, 16}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 1, 0);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Placeholder *outP;
  auto &outType = out1.getType();
  auto &inType = inputs.getType();
  inputP = createQuantizedPlaceholder(
      M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");
  outP = createQuantizedPlaceholder(M, bindings, &out1, outType.getScale(),
                                    outType.getOffset(), "outP");
  auto *maxPool = F->createMaxPool("maxPool", inputP, kernel, stride, pad);
  F->createSave("ret", NodeValue(maxPool,0), outP);

  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaMaxPoolTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaMaxPoolTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);


  Tensor out2(ElemKind::Int8QTy, shape, 1, 0);
  inferMaxPoolNet(&inputs, &out2, kernel, stride, pad, "VTA");
  out2.toBin("vtaMaxPoolTestGolden");

}

TEST(VTASaverTest, avgPoolTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 7, 7, 2048}, 1.0, 0);
  inputs.getHandle<int8_t>().randomize(0, 10, PRNG);
  inputs.toBin("vtaAvgPoolTestInput");

  unsigned_t kernel = 7;
  unsigned_t stride = 1;
  unsigned_t pad = 0;

  std::array<dim_t, 4> S{{1, 1, 1, 2048}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 1.0/8, 0);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Placeholder *outP;
  auto &outType = out1.getType();
  auto &inType = inputs.getType();
  inputP = createQuantizedPlaceholder(
      M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");
  outP = createQuantizedPlaceholder(M, bindings, &out1, outType.getScale(),
                                    outType.getOffset(), "outP");
  llvm::SmallVector<unsigned_t, 4> pads = {pad, pad, pad, pad};
  llvm::SmallVector<unsigned_t, 2> strides = {stride, stride};
  llvm::SmallVector<unsigned_t, 2> kernels = {kernel, kernel};
  TypeRef OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                           outType.getScale(), outType.getOffset());
  auto *avgPool = F->createAvgPool("avgPool", inputP, OT1, kernels, strides, pads);
  //F->createSave("ret", NodeValue(maxPool,0), outP);
  F->createSave("ret", avgPool, outP);
  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaAvgPoolTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaAvgPoolTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);


  Tensor out2(ElemKind::Int8QTy, shape, 1.0/8, 0);
  inferAvgPoolNet(&inputs, &out2, kernel, stride, pad, "VTA");
  out2.toBin("vtaAvgPoolTestGolden");

}


TEST(VTASaverTest, quantizeTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {1, 3, 224, 224});
  inputs.getHandle().randomize(-1.0, 1.0, PRNG);
  inputs.toBin("vtaQuantizeTestInput");
  std::array<dim_t, 4> S{{1, 3, 224, 224}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 1.0/32, 0);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Placeholder *outP;
  auto &outType = out1.getType();
  auto &inType = inputs.getType();
  inputP = createPlaceholder(M, bindings, &inputs, "inputP");

  outP = createQuantizedPlaceholder(M, bindings, &out1, outType.getScale(),
                                    outType.getOffset(), "outP");
  TypeRef OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                   outType.getScale(), outType.getOffset());
  auto *quantize = F->createQuantize("quantize", inputP, OT1);
  F->createSave("ret", quantize, outP);

  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaQuantizeTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaQuantizeTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);

  Tensor out2(ElemKind::Int8QTy, shape, 1.0/32, 0);
  inferQuantizeNet(&inputs, &out2, "VTA");
  out2.toBin("vtaQuantizeTestGolden");

}

TEST(VTASaverTest, dequantizeTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 1000}, 1.0/4, 0);
  inputs.getHandle<int8_t>().randomize(-128, 127, PRNG);
  inputs.toBin("vtaDequantizeTestInput");
  std::array<dim_t, 2> S{{1, 1000}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Placeholder *outP;
  auto &outType = out1.getType();
  auto &inType = inputs.getType();
  inputP = createQuantizedPlaceholder(M, bindings, &inputs, inType.getScale(),
      inType.getOffset(), "inputP");

  outP = createPlaceholder(M, bindings, &out1, "outP");

  TypeRef OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims());
  auto *dequantize = F->createDequantize("dequantize", inputP, OT1);
  F->createSave("ret", dequantize, outP);

  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaDequantizeTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaDequantizeTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);

  Tensor out2(ElemKind::FloatTy, shape);
  inferDequantizeNet(&inputs, &out2, "VTA");
  out2.toBin("vtaDequantizeTestGolden");

}


TEST(VTASaverTest, transposeTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 3, 224, 224}, 1, 0);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  inputs.toBin("vtaReluInput");

  std::array<dim_t, 4> S{{1, 224, 224, 3}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 1, 0);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Placeholder *outP;
  auto &outType = out1.getType();
  auto &inType = inputs.getType();
  inputP = createQuantizedPlaceholder(
      M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");
  outP = createQuantizedPlaceholder(M, bindings, &out1, outType.getScale(),
                                    outType.getOffset(), "outP");
  std::array<unsigned_t, 4> T{{0, 2, 3, 1}};
  llvm::ArrayRef<unsigned_t> shuffle(T);
  auto *transpose = F->createTranspose("relu", inputP, shuffle);
  F->createSave("ret", transpose, outP);

  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaTransposeTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaTransposeTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);


  Tensor out2(ElemKind::Int8QTy, shape, 1, 0);
  inferTransposeNet(&inputs, &out2, shuffle, "VTA");
  out2.toBin("vtaTransposeTestGolden");

}


TEST(VTASaverTest, reluTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 7, 7, 2048}, 1.0, 0);
  inputs.getHandle<int8_t>().randomize(-100, 100, PRNG);
  inputs.toBin("vtaReluTestInput");


  std::array<dim_t, 4> S{{1, 7, 7, 2048}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 1.0, 0);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Placeholder *outP;
  auto &outType = out1.getType();
  auto &inType = inputs.getType();
  inputP = createQuantizedPlaceholder(
      M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");
  outP = createQuantizedPlaceholder(M, bindings, &out1, outType.getScale(),
                                    outType.getOffset(), "outP");

  TypeRef OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                           outType.getScale(), outType.getOffset());
  auto *relu = F->createRELU("relu", inputP);

  F->createSave("ret", relu, outP);
  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaReluTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaReluTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);


  Tensor out2(ElemKind::Int8QTy, shape, 1.0, 0);
  inferVTAReluNet(&inputs, &out2, "VTA");
  out2.toBin("vtaReluTestGolden");

}



TEST(VTASaverTest, elementadd) {
  PseudoRNG PRNG;
  Tensor inputs0(ElemKind::Int8QTy, {1, 7, 7, 2048}, 1.0/4, 0);
  inputs0.getHandle<int8_t>().randomize(0, 60, PRNG);
  inputs0.toBin("vtaElemAddInput0");

  Tensor inputs1(ElemKind::Int8QTy, {1, 7, 7, 2048}, 1.0/2, 0);
  inputs1.getHandle<int8_t>().randomize(0, 60, PRNG);
  inputs1.toBin("vtaElemAddInput1");

  std::array<dim_t, 4> S{{1, 7, 7, 2048}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 1, 0);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP0;
  Placeholder *inputP1;
  Placeholder *outP;
  auto &outType = out1.getType();
  auto &inType0 = inputs0.getType();
  auto &inType1 = inputs1.getType();
  inputP0 = createQuantizedPlaceholder(
      M, bindings, &inputs0, inType0.getScale(), inType0.getOffset(), "inputP0");
  inputP1 = createQuantizedPlaceholder(
      M, bindings, &inputs1, inType1.getScale(), inType1.getOffset(), "inputP1");
  outP = createQuantizedPlaceholder(M, bindings, &out1, outType.getScale(),
                                    outType.getOffset(), "outP");
  TypeRef OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                           outType.getScale(), outType.getOffset());
  auto *elemadd = F->createAdd("elementadd", inputP0, inputP1);
  elemadd->setType(0, OT1);
  F->createSave("ret", elemadd, outP);

  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaElemAddTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaElemAddTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);


  Tensor out2(ElemKind::Int8QTy, shape, 1, 0);
  inferElemAddNet(&inputs0, &inputs1, &out2, "VTA");
  out2.toBin("vtaElemAddTestGolden");

}


TEST(VTASaverTest, elementsub) {
  PseudoRNG PRNG;
  Tensor inputs0(ElemKind::Int8QTy, {1, 7, 7, 2048}, 4.0, 0);
  inputs0.getHandle<int8_t>().randomize(0, 60, PRNG);
  inputs0.toBin("vtaElemSubInput0");

  Tensor inputs1(ElemKind::Int8QTy, {1, 7, 7, 2048}, 1.0, 0);
  inputs1.getHandle<int8_t>().randomize(0, 60, PRNG);
  inputs1.toBin("vtaElemSubInput1");

  std::array<dim_t, 4> S{{1, 7, 7, 2048}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 2.0, 0);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP0;
  Placeholder *inputP1;
  Placeholder *outP;
  auto &outType = out1.getType();
  auto &inType0 = inputs0.getType();
  auto &inType1 = inputs1.getType();
  inputP0 = createQuantizedPlaceholder(
      M, bindings, &inputs0, inType0.getScale(), inType0.getOffset(), "inputP0");
  inputP1 = createQuantizedPlaceholder(
      M, bindings, &inputs1, inType1.getScale(), inType1.getOffset(), "inputP1");
  outP = createQuantizedPlaceholder(M, bindings, &out1, outType.getScale(),
                                    outType.getOffset(), "outP");
  TypeRef OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                           outType.getScale(), outType.getOffset());
  auto *elemsub = F->createSub("elementsub", inputP0, inputP1);
  elemsub->setType(0, OT1);
  F->createSave("ret", elemsub, outP);

  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaElemSubTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaElemSubTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);


  Tensor out2(ElemKind::Int8QTy, shape, 2.0, 0);
  inferElemSubNet(&inputs0, &inputs1, &out2, "VTA");
  out2.toBin("vtaElemSubTestGolden");
}

TEST(VTASaverTest, elementdiv) {
  PseudoRNG PRNG;
  Tensor inputs0(ElemKind::Int8QTy, {1, 7, 7, 2048}, 4.0, 0);
  inputs0.getHandle<int8_t>().randomize(-60, 60, PRNG);
  inputs0.toBin("vtaElemDivInput0");

  Tensor inputs1(ElemKind::Int8QTy, {1, 7, 7, 2048}, 1.0, 0);
  inputs1.getHandle<int8_t>().randomize(1, 60, PRNG);
  inputs1.toBin("vtaElemDivInput1");

  std::array<dim_t, 4> S{{1, 7, 7, 2048}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 2.0, 0);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP0;
  Placeholder *inputP1;
  Placeholder *outP;
  auto &outType = out1.getType();
  auto &inType0 = inputs0.getType();
  auto &inType1 = inputs1.getType();
  inputP0 = createQuantizedPlaceholder(
      M, bindings, &inputs0, inType0.getScale(), inType0.getOffset(), "inputP0");
  inputP1 = createQuantizedPlaceholder(
      M, bindings, &inputs1, inType1.getScale(), inType1.getOffset(), "inputP1");
  outP = createQuantizedPlaceholder(M, bindings, &out1, outType.getScale(),
                                    outType.getOffset(), "outP");
  TypeRef OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims(),
                                           outType.getScale(), outType.getOffset());
  auto *elemdiv = F->createDiv("elementdiv", inputP0, inputP1);
  elemdiv->setType(0, OT1);
  F->createSave("ret", elemdiv, outP);

  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaElemDivTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaElemDivTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);


  Tensor out2(ElemKind::Int8QTy, shape, 2.0, 0);
  inferElemDivNet(&inputs0, &inputs1, &out2, "VTA");
  out2.toBin("vtaElemDivTestGolden");
}

TEST(VTASaverTest, splatMax) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 112, 112, 64}, 1.0/8, 0);
  inputs.getHandle<int8_t>().randomize(-60, 60, PRNG);
  inputs.toBin("vtaSplatMaxTestInput");

  std::array<dim_t, 4> S{{1, 112, 112, 64}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 1.0/8, 0);


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Placeholder *outP;
  auto &outType = out1.getType();
  auto &inType = inputs.getType();
  inputP = createQuantizedPlaceholder(
      M, bindings, &inputs, inType.getScale(), inType.getOffset(), "inputP");
  outP = createQuantizedPlaceholder(M, bindings, &out1, outType.getScale(),
                                    outType.getOffset(), "outP");
  TypeRef IT = F->getParent()->uniqueType(inputs.getElementType(), inputs.dims(),
                                           inType.getScale(), inType.getOffset());
  auto *splat = F->createSplat("splat", IT, 0.0);
  auto *elemmax = F->createMax("elementmax", inputP, splat);
  F->createSave("ret", elemmax, outP);

  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaSplatMaxTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaSplatMaxTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);


  Tensor out2(ElemKind::Int8QTy, shape, 1.0/8, 0);
  inferSplatMaxNet(&inputs, &out2, "VTA");
  out2.toBin("vtaSplatMaxTestGolden");


}

TEST(VTASaverTest, softMaxTest) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {1, 1000});
  inputs.getHandle().randomize(-1.0, 1.0, PRNG);
  inputs.toBin("vtaSoftMaxTestInput");

  Tensor out1(ElemKind::FloatTy, {1,1000});


  PlaceholderBindings bindings;
  Module M;
  Function *F = M.createFunction("main");
  Placeholder *inputP;
  Placeholder *outP;
  auto &outType = out1.getType();
  auto &inType = inputs.getType();
  inputP = createPlaceholder(M, bindings, &inputs, "inputP");

  outP = createPlaceholder(M, bindings, &out1, "outP");
  TypeRef OT1 = F->getParent()->uniqueType(out1.getElementType(), out1.dims());

  auto selected =
      M.createConstant(ElemKind::Int64ITy, {inputP->dims()[0], 1}, "selected");

  auto *softMax = F->createSoftMax("softmax", inputP, selected, OT1);
  F->createSave("ret", softMax, outP);

  std::cout<<"Function Dump Before optimizeFunction START" <<std::endl;
  F->dump();
  std::cout<<"Function Dump  Before optimizeFunction FINISH" <<std::endl;
  //Optimize & Lowering
  std::unique_ptr<Backend> backend(createBackend("VTA"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);

  EXIT_ON_ERR(std::move(error));
  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "vtaSoftMaxTestBundle"; //file name
  llvm::StringRef mainEntryName = "vtaSoftMaxTestMainEntry"; //main function name

  backend->save(F, outputDir, bundleName, mainEntryName);

  Tensor out2(ElemKind::FloatTy, {1, 1000});
  inferSoftMaxNet(&inputs, &out2, "VTA");
  out2.toBin("vtaSoftMaxTestGolden");
}
