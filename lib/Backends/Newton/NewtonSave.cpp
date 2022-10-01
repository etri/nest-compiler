/*****************************************************************************
*
* Copyright Next-Generation System Software Research Group, All rights reserved.
* Future Computing Research Division, Artificial Intelligence Reserch Laboratory
* Electronics and Telecommunications Research Institute (ETRI)
*
* THESE DOCUMENTS CONTAIN CONFIDENTIAL INFORMATION AND KNOWLEDGE
* WHICH IS THE PROPERTY OF ETRI. NO PART OF THIS PUBLICATION IS
* TO BE USED FOR ANY OTHER PURPOSE, AND THESE ARE NOT TO BE
* REPRODUCED, COPIED, DISCLOSED, TRANSMITTED, STORED IN A RETRIEVAL
* SYSTEM OR TRANSLATED INTO ANY OTHER HUMAN OR COMPUTER LANGUAGE,
* IN ANY FORM, BY ANY MEANS, IN WHOLE OR IN PART, WITHOUT THE
* COMPLETE PRIOR WRITTEN PERMISSION OF ETRI.
*
* LICENSE file : LICENSE_ETRI located in the top directory
*
*****************************************************************************/

#include "Newton.h"
#include "NewtonCodeGen/NewtonSaver.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"
#include <cstring>
#include <fstream>
#include <vector>

using namespace glow;

namespace newton_save{

class NewtonSaveContext{
 public:
  NewtonSaveContext(IRFunction::VariableMap *vmap)
  {
    variable_map = vmap;
    constantWeightVarsMemSize = 0;
    mutableWeightVarsMemSize = 0;
  }
  std::vector<struct SymbolTableEntry> *getSymbols(){
    return &syms;
  }
  std::vector<struct SymbolTableEntry> *getConstantSymbols(){
    return &csyms;
  }
  std::vector<struct SymbolTableEntry> *getTempSymbols(){
    return &tsyms;
  }
  std::vector<struct SymbolTableEntry> *getFCsSymbols(){
    return &fcs;
  }

  uint64_t getCMemSize(){
    return constantWeightVarsMemSize;
  }
  void setCMemSize(uint64_t size){
    constantWeightVarsMemSize = size;
  }
  uint64_t getMMemSize(){
    return mutableWeightVarsMemSize;
  }
  void setMMemSize(uint64_t size){
    mutableWeightVarsMemSize = size;
  }
  void setWeightFileStream(std::ofstream *fos)
  {
    wfos= fos;
  }
  std::ofstream* getWeightFileStream(){
    return wfos;
  }
  IRFunction::VariableMap *getVariableMap()
  {
    return variable_map;
  }

 private:

  IRFunction::VariableMap *variable_map;
  std::vector<struct SymbolTableEntry> syms;
  std::vector<struct SymbolTableEntry> csyms;
  std::vector<struct SymbolTableEntry> tsyms;
  std::vector<struct SymbolTableEntry> fcs;
  uint64_t constantWeightVarsMemSize;
  uint64_t mutableWeightVarsMemSize;
  std::ofstream* wfos;

};

struct SymbolTableEntry {
  // Name of a variable.
  const char *name;
  // Offset of the variable inside the memory area.
  uint64_t offset;
  // The number of elements inside this variable.
  uint64_t size;
  // Variable kind: 1 if it is a mutable variable, 0 otherwise.
  char kind;
  WeightVar* wgt = nullptr;
};

SymbolTableEntry addSymbolEntry(WeightVar* wgt, NewtonSaveContext *ctx){
  auto syms = ctx->getSymbols();
  struct SymbolTableEntry ste;
  for (auto s : *syms) {
    auto name = (const char *) (wgt->getName().data());

    if (strcmp(s.name, name) == 0) {
      ste = s;
      return ste;
    }
  }
  syms->push_back({wgt->getName().data(), ctx->getMMemSize(), wgt->size(), '1', wgt});
  ste = syms->back();
  ctx->setMMemSize(ctx->getMMemSize() + wgt->getSizeInBytes());
  return ste;

}

SymbolTableEntry addSymbolEntryGenBundle(WeightVar* wgt, std::string *bundle, NewtonSaveContext *ctx){
  auto syms = ctx->getSymbols();
  struct SymbolTableEntry ste;
  for (auto s : *syms) {
    auto name = (const char *) (wgt->getName().data());

    if (strcmp(s.name, name) == 0) {
      ste = s;
      return ste;
    }
  }

  auto csyms = ctx->getConstantSymbols();
  for (auto s : *csyms) {
    auto name = (const char *) (wgt->getName().data());

    if (strcmp(s.name, name) == 0) {
      ste = s;
      return ste;
    }
  }

  syms->push_back({wgt->getName().data(), ctx->getMMemSize(), wgt->size()*4, '1', wgt});
  ste = syms->back();
  ctx->setMMemSize(ctx->getMMemSize() + wgt->getSizeInBytes());

  bundle->append("  float* ");
  bundle->append(ste.name);
  bundle->append(" = (float*)mutableWeight + ");
  bundle->append(std::to_string(ste.offset/4));
  bundle->append(";\n");

  return ste;
}

SymbolTableEntry addConstantSymbolEntry(Value* val, NewtonSaveContext *ctx){
  auto csyms = ctx->getConstantSymbols();
  struct SymbolTableEntry ste;
  for (auto s : *csyms) {
    auto name = (const char *) (val->getName().data());
    if (strcmp(s.name, name) == 0) {
      ste = s;
      return ste;
    }
  }

  csyms->push_back({val->getName().data(), ctx->getCMemSize(), val->getSizeInBytes(), '0'});
  ctx->setCMemSize(ctx->getCMemSize() + val->getSizeInBytes());
  ste = csyms->back();
  return ste;
}


void saveFullyConnectedInst(const glow::FullyConnectedInst *Inst, std::string *bundle, std::string *createAiM, NewtonSaveContext *ctx) {
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto destDims = dest->dims();
  auto filter = Inst->getWeights();
  auto filterDims = filter->dims();
  auto bias = Inst->getBias();

  //save Weight file
  auto vMap = ctx->getVariableMap();
  const Tensor* tensor = NULL;

  for(auto it = vMap->begin(); it != vMap->end(); it++)
  {
    if(it->second == filter)
    {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensor = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensor);
  auto handle = tensor->getHandle<float>();
  {
    auto fos = ctx->getWeightFileStream();
//    for (size_t i = 0, e = handle.size(); i < e; i++) {
//      auto data = handle.raw(i);
//      fos->write((const char *)&data, 4);
//    }
    for (dim_t w = 0; w < filterDims[1]; w++) {
      for (dim_t h = 0; h < filterDims[0]; h++) {
        auto data = handle.raw(h*filterDims[1]+w);
        fos->write((const char *)&data,4);
      }
    }

  }

  auto filterSte = addConstantSymbolEntry(filter, ctx);

  //save Bias file
  const Tensor* tensorBias = NULL;

  for(auto it = vMap->begin(); it != vMap->end(); it++)
  {
    if(it->second == bias)
    {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensorBias = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensorBias);
  auto handleBias = tensorBias->getHandle<float>();
  {
    auto fos = ctx->getWeightFileStream();
    for (size_t i = 0, e = handleBias.size(); i < e; i++) {
      auto data = handleBias.raw(i);
      fos->write((const char *) &data, 4);
    }
  }
  auto biasSte = addConstantSymbolEntry(bias, ctx);

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }
  //  struct act_info act_info {static_cast<int>(ACTIVATION::BYPASS_ACT), 0.0f, 0};
  createAiM->append("  struct act_info ");
  createAiM->append(dest->getName());
  createAiM->append("_act_info {static_cast<int>(ACTIVATION::BYPASS_ACT), 0.0f, 0};\n");
  createAiM->append("  ");
  createAiM->append(dest->getName());
  createAiM->append("_id = CreateAiMLinearOp(-1, ");

  bundle->append("  struct analytic_result_ ");
  bundle->append(dest->getName());
  bundle->append("_profile_ptr;\n");
  bundle->append("  RunAiMLinearOp(");
  bundle->append(dest->getName());
  bundle->append("_id, ");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }

  createAiM->append(filterSte.name);
  createAiM->append(", ");
  createAiM->append(biasSte.name);
  createAiM->append(", ");
  createAiM->append(std::to_string(srcDims[1]));
  createAiM->append(", ");
  createAiM->append(std::to_string(destDims[1]));
  createAiM->append(", ");
  createAiM->append(dest->getName());
  createAiM->append("_act_info, false, false);\n");

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }
  bundle->append(std::to_string(srcDims[0]));
  bundle->append(", (void*) &");
  bundle->append(dest->getName());
  bundle->append("_profile_ptr);\n");
  auto fcs = ctx->getFCsSymbols();
  fcs->push_back({dest->getName().data(), 0, 0, '0'});
}


struct BundleSaveCtx{
  std::string bundle;
  std::string initConst;
  std::string createAiM;
  std::string destroyConst;
  std::string includeHeader;
  std::string symbolTable;
  std::string bundleConfig;
};


void initBundleSave(struct BundleSaveCtx &bundleCtx,
                    llvm::StringRef bundleName,
                    llvm::StringRef mainEntryName){
  auto& bundle = bundleCtx.bundle;
  //bundleCtx.bundle = new std::string();
  bundle = "";
  bundle.append("int ");
  bundle.append(mainEntryName);
  bundle.append("(float *constantWeight, float *mutableWeight, float *activations){\n");
#ifdef Newton_ALLINONE_FUNCTION
  bundle.append("  ");
  bundle.append(bundleName);
  bundle.append("_load_module(constantWeight);\n");
#endif
#ifdef Newton_RESET_CMDH
  bundle.append("  vtaCmdH = NewtonTLSCommandHandle();\n");
#endif

#ifdef Newton_PROFILE
  bundle.append("  std::ofstream prof_out(\"profileResult.txt\");\n");
#endif

  bundleCtx.initConst = "";
  bundleCtx.destroyConst = "";
  bundleCtx.createAiM = "";
  // Generate Include Header
  bundleCtx.includeHeader ="#include \"";
  bundleCtx.includeHeader.append(mainEntryName);
  bundleCtx.includeHeader.append(".h\"\n");
  bundleCtx.includeHeader.append("#include <aim.h>\n#include \"src/activation/include/activation.h\"\n");



}

void finalizeBundleEntrySave(struct BundleSaveCtx &bundleCtx,
                             llvm::StringRef bundleName,
                             llvm::StringRef mainEntryName,
                             NewtonSaveContext &ctx){
  auto& bundle = bundleCtx.bundle;

  bundle.append("  return 0;\n");
  bundle.append("}");



  // Generate SymbolTable struct
  auto& symbolTable = bundleCtx.symbolTable;
  symbolTable = "SymbolTableEntry symbolTableEntry_";
  symbolTable.append(bundleName);
  symbolTable.append("[");
  symbolTable.append(std::to_string(ctx.getSymbols()->size()));
  symbolTable.append("]={");
  for(auto sym : *(ctx.getSymbols())){
    symbolTable.append("{\"");
    symbolTable.append(sym.name);
    symbolTable.append("\",");
    symbolTable.append(std::to_string(sym.offset));
    symbolTable.append(",");
    symbolTable.append(std::to_string(sym.size));
    symbolTable.append(",'1'},");
  }
  symbolTable.erase(symbolTable.size()-1,1);
  symbolTable.append("};\n");


  // Generate BundleConfig struct
  auto& bundleConfig = bundleCtx.bundleConfig;
  bundleConfig = "BundleConfig ";
  bundleConfig.append(bundleName);
  bundleConfig.append("_config = {");
  bundleConfig.append(std::to_string(ctx.getCMemSize()));
  bundleConfig.append(", ");
  bundleConfig.append(std::to_string(ctx.getMMemSize()));
  bundleConfig.append(", 0, 64, ");
  bundleConfig.append(std::to_string(ctx.getSymbols()->size()));
  bundleConfig.append(", symbolTableEntry_");
  bundleConfig.append(bundleName);
  bundleConfig.append("};\n");



  auto& initConst = bundleCtx.initConst;
  initConst.append("namespace namespace_");
  initConst.append(bundleName);
  initConst.append(" {\n");
  // Generate Init Constant values
  for(auto ste : *ctx.getConstantSymbols()){
    if(ste.kind=='0') {
      initConst.append("float* ");
      initConst.append(ste.name);
      initConst.append(";\n");
    }
    else{
      initConst.append("float* ");
      initConst.append(ste.name);
      initConst.append(" = (float *)malloc(");
      initConst.append(std::to_string(ste.size));
      initConst.append(");\n");
    }
  }

  for(auto ste : *ctx.getFCsSymbols()){
    if(ste.kind=='0') {
      initConst.append("int ");
      initConst.append(ste.name);
      initConst.append("_id;\n");
    }
  }
  initConst.append("}\n");
  initConst.append("using namespace namespace_");
  initConst.append(bundleName);
  initConst.append(";\n");
//  initConst.append("extern NewtonCommandHandle vtaCmdH");
//  initConst.append(";\n");

  initConst.append("\nvoid ");
  initConst.append(mainEntryName);
  initConst.append("_load_module(float *constantWeight){\n");
  //initConst.append("  xlnk_reset();\n");
  //initConst.append("  vtaCmdH = NewtonTLSCommandHandle();\n");

//
//  auto weight_ste = ctx.getConstantSymbols()->at(0);
//  auto bias_ste = ctx.getConstantSymbols()->at(1);
//  initConst.append("  ");
//  initConst.append(bundleName);
//  initConst.append("_id = CreateAiMLinearOp(-1, constantWeight + ");
//  initConst.append(std::to_string(weight_ste.offset));
//  initConst.append(", constantWeight + ");
//  initConst.append(std::to_string(bias_ste.offset));
//
  for(auto ste : *ctx.getConstantSymbols()){
    if(ste.kind=='0') {
      initConst.append("  ");
      initConst.append(ste.name);
      initConst.append(" = (float *)(constantWeight + ");
      initConst.append(std::to_string(ste.offset/4));
      initConst.append(");\n");

//      initConst.append("  ");
//      initConst.append(ste.name);
//      initConst.append(" = (int8_t *)NewtonBufferAlloc(");
//      initConst.append(std::to_string(ste.size));
//      initConst.append(");\n");
//      initConst.append("  NewtonBufferCopy((int8_t *)(constantWeight + ");
//      initConst.append(std::to_string(ste.offset));
//      initConst.append("), 0, ");
//      initConst.append(ste.name);
//      initConst.append(", 0, ");
//      initConst.append(std::to_string(ste.size));
//      initConst.append(", 1);\n");
    }
  }
//  initConst.append("}\n");

  auto& destroyConst = bundleCtx.destroyConst;
  destroyConst.append("\nvoid ");
  destroyConst.append(mainEntryName);
  destroyConst.append("_destroy_module(){\n");
  for(auto ste : *ctx.getConstantSymbols()){
    if(ste.kind=='0') {
//      destroyConst.append("  ");
//      destroyConst.append("NewtonBufferFree(");
//      destroyConst.append(ste.name);
//      destroyConst.append(");\n");
    }
  }
  destroyConst.append("}\n");


}

void exportBundleEntry(llvm::StringRef outputDir,
                       llvm::StringRef bundleName,
                       struct BundleSaveCtx &bundleCtx){
  std::string outputFile = outputDir;
  if(outputFile[outputFile.size()-1]!='/')
    outputFile.append("/");
  outputFile.append(bundleName);
  outputFile.append(".cpp");

  std::ofstream fos(outputFile.c_str(), std::ios::out);
//  fos.write(Newton_SAVE_COMMON.c_str(), Newton_SAVE_COMMON.size());
  fos.write(bundleCtx.includeHeader.c_str(), bundleCtx.includeHeader.size());
  auto& symbolTable = bundleCtx.symbolTable;
  fos.write(symbolTable.c_str(), symbolTable.size());
  auto& bundleConfig = bundleCtx.bundleConfig;
  fos.write(bundleConfig.c_str(), bundleConfig.size());
  fos.write(bundleCtx.initConst.c_str(), bundleCtx.initConst.size());
  fos.write(bundleCtx.createAiM.c_str(), bundleCtx.createAiM.size());
  fos.write("}\n", 2);
  fos.write(bundleCtx.destroyConst.c_str(), bundleCtx.destroyConst.size());
  fos.write(bundleCtx.bundle.c_str(), bundleCtx.bundle.size());
  fos.close();
}

void exportNewtonRuntimeHeader(llvm::StringRef outputDir){
  std::string vtaRuntimeHeaderFile = outputDir;
  vtaRuntimeHeaderFile.append("/");
  vtaRuntimeHeaderFile.append("NewtonRuntime");
  vtaRuntimeHeaderFile.append(".h");

  std::ofstream ros(vtaRuntimeHeaderFile.c_str(), std::ios::out);
//  ros.write(Newton_RUNTIME_HEADER.c_str(), Newton_RUNTIME_HEADER.size());
  ros.close();
}

void exportBundleHeader(llvm::StringRef outputDir,
                        llvm::StringRef bundleName,
                        llvm::StringRef mainEntryName,
                        NewtonSaveContext &ctx){
  std::string vtaBundleHeaderFile = outputDir;
  vtaBundleHeaderFile.append("/");
  vtaBundleHeaderFile.append(mainEntryName);
  vtaBundleHeaderFile.append(".h");

  std::ofstream bhos(vtaBundleHeaderFile.c_str(), std::ios::out);
  std::string bundleHeaderDefineName = "Newton_BUNDLE_";
  bundleHeaderDefineName.append(bundleName);
  std::transform(bundleHeaderDefineName.begin(), bundleHeaderDefineName.end(), bundleHeaderDefineName.begin(), ::toupper);

  std::string bundleHeaderDefine = "#ifndef ";
  bundleHeaderDefine.append(bundleHeaderDefineName);
  bundleHeaderDefine.append("\n#define ");
  bundleHeaderDefine.append(bundleHeaderDefineName);
  bundleHeaderDefine.append("\n");

  bhos.write(bundleHeaderDefine.c_str(), bundleHeaderDefine.size());
  bhos.write(Newton_BUNDLE_HEADER_0.c_str(), Newton_BUNDLE_HEADER_0.size());
  std::string bundleHeader = "";
  bundleHeader.append("extern BundleConfig ");
  bundleHeader.append(bundleName);
  bundleHeader.append("_config;\n");
  bundleHeader.append("void ");
  bundleHeader.append(mainEntryName);
  bundleHeader.append("_load_module(float *constantWeight);\n");
  bundleHeader.append("void ");
  bundleHeader.append(mainEntryName);
  bundleHeader.append("_destroy_module();\n");
  bundleHeader.append("int ");
  bundleHeader.append(mainEntryName);
  bundleHeader.append("(float *constantWeight, float *mutableWeight, float *activations);\n");
  std::string bundleHeaderPHInfo = R"~(
// ---------------------------------------------------------------
//                          Bundle API
// ---------------------------------------------------------------
)~";
  bundleHeaderPHInfo.append("// Model name: \"");
  bundleHeaderPHInfo.append(bundleName);
  bundleHeaderPHInfo.append("\"\n");
  bundleHeaderPHInfo.append("// Total data size: ");
  bundleHeaderPHInfo.append(std::to_string(ctx.getMMemSize()));
  bundleHeaderPHInfo.append("(bytes)\n");
  bundleHeaderPHInfo.append("// Placeholders:\n");

  for(auto ste : *ctx.getSymbols()) {
    auto wgt = ste.wgt;
    assert(wgt);
    bundleHeaderPHInfo.append("//   Name: \"");
    bundleHeaderPHInfo.append(ste.name);
    bundleHeaderPHInfo.append("\"\n");
    bundleHeaderPHInfo.append("//   Size: ");
    bundleHeaderPHInfo.append(std::to_string(wgt->size()));
    bundleHeaderPHInfo.append(" (elements)\n");
    bundleHeaderPHInfo.append("//   Size: ");
    bundleHeaderPHInfo.append(std::to_string(wgt->getSizeInBytes()));
    bundleHeaderPHInfo.append(" (bytes)\n");
    bundleHeaderPHInfo.append("//   Offset: ");
    bundleHeaderPHInfo.append(std::to_string(ste.offset));
    bundleHeaderPHInfo.append(" (bytes)\n//\n");

  }
  bhos.write(bundleHeaderPHInfo.c_str(), bundleHeaderPHInfo.size());
  bhos.write(bundleHeader.c_str(), bundleHeader.size());
  bhos.write(Newton_BUNDLE_HEADER_1.c_str(), Newton_BUNDLE_HEADER_1.size());

  std::string bundleHeaderEndif = "#endif //";
  bundleHeaderEndif.append(bundleHeaderDefineName);
  bundleHeaderEndif.append("\n");
  bhos.write(bundleHeaderEndif.c_str(), bundleHeaderEndif.size());

  bhos.close();
}

}
using namespace newton_save;

void Newton::save(Function *F, llvm::StringRef outputDir,
               llvm::StringRef bundleName,
               llvm::StringRef mainEntryName) const {

  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());

  NewtonSaveContext ctx(&IR->getVariableMap());

  std::string weightFileName = outputDir;
  weightFileName.append("/");
  weightFileName.append(bundleName);
  weightFileName.append(".weights.bin");
  std::ofstream wfos(weightFileName.c_str(), std::ios::binary);
  ctx.setWeightFileStream(&wfos);

  struct BundleSaveCtx bundleCtx;
  auto& bundle = bundleCtx.bundle;
  auto& initConst = bundleCtx.initConst;
  auto& createAiM = bundleCtx.createAiM;
  auto& destroyConst = bundleCtx.destroyConst;


  initBundleSave(bundleCtx, bundleName, mainEntryName);


  for (const auto &I : IR->getInstrs()) {


    switch (I.getKind()) {
    case Kinded::Kind::FullyConnectedInstKind: {
      auto I2 = llvm::cast<FullyConnectedInst>(&I);
      saveFullyConnectedInst(I2, &bundle, &createAiM, &ctx);
      break;
    }
      case Kinded::Kind::AllocActivationInstKind: {
        auto I2 = llvm::cast<AllocActivationInst>(&I);
        // for not constant but represented as a constant value
        auto csyms = ctx.getConstantSymbols();
        csyms->push_back({I2->getName().data(), 0, I2->getSizeInBytes(), '1'});

        break;
      }
      case Kinded::Kind::DeallocActivationInstKind: {
        break;
      }

    default:
      std::string msg = I.getKindName();
      msg.append(" is an unhandled instruction");

      llvm_unreachable(msg.c_str());
    }

  }

  wfos.close();
  finalizeBundleEntrySave(bundleCtx, bundleName, mainEntryName, ctx);
  exportBundleEntry(outputDir, bundleName, bundleCtx);
  exportNewtonRuntimeHeader(outputDir);
  exportBundleHeader(outputDir, bundleName, mainEntryName, ctx);



  return;
}


