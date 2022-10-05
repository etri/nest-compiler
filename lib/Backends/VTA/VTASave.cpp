/*****************************************************************************
 *
 * Copyright Next-Generation System Software Research Group, All rights
 *reserved. Future Computing Research Division, Artificial Intelligence Reserch
 *Laboratory Electronics and Telecommunications Research Institute (ETRI)
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

#include "VTA.h"
#include "VTACodeGen/VTASaver.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"
#include <cstring>
#include <fstream>
#include <vector>

//#define VTA_PROFILE
//#define VTA_ALLINONE_FUNCTION
//#define VTA_RESET_CMDH
//#define VTA_MEMOPT_DISABLE

using namespace glow;

namespace vta_save{

class VTASaveContext {
public:
  VTASaveContext(IRFunction::VariableMap *vmap) {
    variable_map = vmap;
    constantWeightVarsMemSize = 0;
    mutableWeightVarsMemSize = 0;
  }
  std::vector<struct SymbolTableEntry> *getSymbols() { return &syms; }
  std::vector<struct SymbolTableEntry> *getConstantSymbols() { return &csyms; }
  std::vector<struct SymbolTableEntry> *getTempSymbols() { return &tsyms; }
  uint64_t getCMemSize() { return constantWeightVarsMemSize; }
  void setCMemSize(uint64_t size) { constantWeightVarsMemSize = size; }
  uint64_t getMMemSize() { return mutableWeightVarsMemSize; }
  void setMMemSize(uint64_t size) { mutableWeightVarsMemSize = size; }
  void setWeightFileStream(std::ofstream *fos) { wfos = fos; }
  std::ofstream *getWeightFileStream() { return wfos; }
  IRFunction::VariableMap *getVariableMap() { return variable_map; }
  void setIdxMultiEVTA(uint32_t idx) { idxMultiEVTA = idx; }
  uint32_t getIdxMultiEVTA() { return idxMultiEVTA; }

private:
  IRFunction::VariableMap *variable_map;
  std::vector<struct SymbolTableEntry> syms;
  std::vector<struct SymbolTableEntry> csyms;
  std::vector<struct SymbolTableEntry> tsyms;
  uint64_t constantWeightVarsMemSize;
  uint64_t mutableWeightVarsMemSize;
  std::ofstream *wfos;
  uint32_t idxMultiEVTA;
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
  WeightVar *wgt = nullptr;
};

SymbolTableEntry addSymbolEntry(WeightVar *wgt, VTASaveContext *ctx) {
  auto syms = ctx->getSymbols();
  struct SymbolTableEntry ste;
  for (auto s : *syms) {
    auto name = (const char *)(wgt->getName().data());

    if (strcmp(s.name, name) == 0) {
      ste = s;
      return ste;
    }
  }
  syms->push_back(
      {wgt->getName().data(), ctx->getMMemSize(), wgt->size(), '1', wgt});
  ste = syms->back();
  ctx->setMMemSize(ctx->getMMemSize() + wgt->getSizeInBytes());
  return ste;
}

SymbolTableEntry addSymbolEntryGenBundle(WeightVar *wgt, std::string *bundle,
                                         VTASaveContext *ctx) {
  auto syms = ctx->getSymbols();
  struct SymbolTableEntry ste;
  for (auto s : *syms) {
    auto name = (const char *)(wgt->getName().data());

    if (strcmp(s.name, name) == 0) {
      ste = s;
      return ste;
    }
  }

  auto csyms = ctx->getConstantSymbols();
  for (auto s : *csyms) {
    auto name = (const char *)(wgt->getName().data());

    if (strcmp(s.name, name) == 0) {
      ste = s;
      return ste;
    }
  }

  syms->push_back(
      {wgt->getName().data(), ctx->getMMemSize(), wgt->size(), '1', wgt});
  ste = syms->back();
  ctx->setMMemSize(ctx->getMMemSize() + wgt->getSizeInBytes());

  bundle->append("  int8_t* ");
  bundle->append(ste.name);
  bundle->append(" = (int8_t*)mutableWeight + ");
  bundle->append(std::to_string(ste.offset));
  bundle->append(";\n");

  return ste;
}

SymbolTableEntry addConstantSymbolEntry(Value *val, VTASaveContext *ctx) {
  auto csyms = ctx->getConstantSymbols();
  struct SymbolTableEntry ste;
  for (auto s : *csyms) {
    auto name = (const char *)(val->getName().data());
    if (strcmp(s.name, name) == 0) {
      ste = s;
      return ste;
    }
  }
#if defined(VTA_BNN)
  if (val->dims().size() == 4 && val->dims()[3] % 128 == 0) {
    csyms->push_back({val->getName().data(), ctx->getCMemSize(),
                      val->getSizeInBytes() / 8, '0'});
    ctx->setCMemSize(ctx->getCMemSize() + val->getSizeInBytes() / 8);
    ste = csyms->back();
  } else {
    csyms->push_back({val->getName().data(), ctx->getCMemSize(),
                      val->getSizeInBytes(), '0'});
    ctx->setCMemSize(ctx->getCMemSize() + val->getSizeInBytes());
    ste = csyms->back();
  }
#else
  csyms->push_back(
      {val->getName().data(), ctx->getCMemSize(), val->getSizeInBytes(), '0'});
  ctx->setCMemSize(ctx->getCMemSize() + val->getSizeInBytes());
  ste = csyms->back();
#endif
  return ste;
}

int getExpofPowerofTwo(int x) {
  int n = 0;
  // checks whether a number is zero or not
  if (x == -1)
    return 0;

  // true till x is not equal to 1
  while (x != 1) {
    // checks whether a number is divisible by 2
    if (x % 2 != 0)
      return -1;
    x /= 2;
    n++;
  }
  return n;
}

void saveFloatConvolutionInst(const glow::ConvolutionInst *Inst,
                              std::string *bundle, VTASaveContext *ctx) {
  auto pad = Inst->getPads();
  auto strides = Inst->getStrides();
  // TODO group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto filter = Inst->getFilter();
  auto filterDims = filter->dims();

  int N = srcDims[0];
  int H = srcDims[1];
  int W = srcDims[2];
  int C = srcDims[3];
  int KN = filterDims[0];
  int KH = filterDims[1];
  int KW = filterDims[2];

  auto bias = Inst->getBias();
  assert(bias->dims()[0] == filterDims[0]);

  int pad_size = pad[0];
  int stride_size = strides[0];

  // save Weight file
  auto vMap = ctx->getVariableMap();
  const Tensor *tensor = NULL;

  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == filter) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensor = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensor);

  auto handle = tensor->getHandle();
  {
    auto fos = ctx->getWeightFileStream();

    for (size_t i = 0, e = handle.size(); i < e; i++) {
      auto data = handle.raw(i);
      fos->write((const char *)(&data), sizeof(float));
    }
  }

  auto filterSte = addConstantSymbolEntry(filter, ctx);

  // save Bias file
  const Tensor *tensorBias = NULL;
  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == bias) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensorBias = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensorBias);

  auto handleBias = tensorBias->getHandle();
  {
    auto fos = ctx->getWeightFileStream();
    for (size_t i = 0, e = handleBias.size(); i < e; i++) {
      auto data = handleBias.raw(i);
      fos->write((const char *)&data, sizeof(float));
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

  bundle->append("  convolutionFloat(");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }

  bundle->append("(int8_t*)VTABufferGetVirtAddr(");
  bundle->append(filterSte.name);
  bundle->append("), (int8_t *)VTABufferGetVirtAddr(");
  bundle->append(biasSte.name);
  bundle->append("), ");

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }

  bundle->append(std::to_string(N));
  bundle->append(", ");
  bundle->append(std::to_string(H));
  bundle->append(", ");
  bundle->append(std::to_string(W));
  bundle->append(", ");
  bundle->append(std::to_string(C));
  bundle->append(", ");
  bundle->append(std::to_string(KN));
  bundle->append(", ");
  bundle->append(std::to_string(KH));
  bundle->append(", ");
  bundle->append(std::to_string(KW));
  bundle->append(", ");
  bundle->append(std::to_string(pad_size));
  bundle->append(", ");
  bundle->append(std::to_string(stride_size));
  bundle->append(", ");
  bundle->append(std::to_string(Inst->getGroup()));
  bundle->append(", ");
  bundle->append(std::to_string(Inst->getDilation()));
  bundle->append(", ");
  bundle->append(std::to_string(Inst->getFusedActivation() ==
                                RELU)); // Fused Activation Disabled
  bundle->append(", ");
  bundle->append(std::to_string(1));
  bundle->append(", ");
  bundle->append(std::to_string(dest->dims()[1]));
  bundle->append(", ");
  bundle->append(std::to_string(dest->dims()[2]));
  bundle->append(" );\n");
}

void saveVTADWConvolutionInst(const glow::ConvolutionInst *Inst,
                              std::string *bundle, std::string *initConst,
                              VTASaveContext *ctx) {
  bundle->append("  dwc(...);\n");
}

void saveNonVTAConvolutionInst(
    const glow::ConvolutionInst *Inst, std::string *bundle,
    std::string *initConst,
    VTASaveContext *ctx) { // NON VTA Quantized Convolution
  auto pad = Inst->getPads();
  auto strides = Inst->getStrides();
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto filter = Inst->getFilter();
  auto filterDims = filter->dims();

  int N = srcDims[0];
  int H = srcDims[1];
  int W = srcDims[2];
  int C = srcDims[3];
  int KN = filterDims[0];
  int KH = filterDims[1];
  int KW = filterDims[2];

  auto bias = Inst->getBias();

  int pad_size = pad[0];
  int stride_size = strides[0];

  bool doRelu = Inst->getFusedActivation() == RELU;
  float inScale = src->getType()->getScale();
  float filterScale = filter->getType()->getScale();
  float outScale = Inst->getDest()->getType()->getScale();
  float biasScale = bias->getType()->getScale();

  filterScale = 1 / filterScale;
  inScale = 1 / inScale;
  biasScale = 1 / biasScale;
  outScale = 1 / outScale;

  // save Weight file
  auto vMap = ctx->getVariableMap();
  const Tensor *tensor = NULL;

  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == filter) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensor = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensor);
  auto handle = tensor->getHandle<int8_t>();
  assert(handle.size() % 2 == 0);
  {
    auto fos = ctx->getWeightFileStream();
    int16_t data16 = 0;
    for (size_t i = 0, e = handle.size(); i < e; i++) {
      auto data = handle.raw(i);
      if (data > 127)
        data = 127.0;
      if (data < -128)
        data = -128.0;
      int8_t clip_data = std::floor(data);
      if (i % 2 == 0) {
        data16 = 0xff & clip_data;
      } else {
        data16 = data16 | clip_data << 8;
        fos->write((const char *)&data16, 2);
      }
    }
  }

  auto filterSte = addConstantSymbolEntry(filter, ctx);

  // save Bias file

  bool doBias = true;

  const Tensor *tensorBias = NULL;

  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == bias) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensorBias = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensorBias);
  auto handleBias = tensorBias->getHandle<int32_t>();
  assert(handleBias.size() % 2 == 0);
  {
    auto fos = ctx->getWeightFileStream();
    for (size_t i = 0, e = handleBias.size(); i < e; i++) {
      auto data = handleBias.raw(i);
      fos->write((const char *)&data, 4);
    }
  }

  assert(biasScale == inScale * filterScale);

  auto biasSte = addConstantSymbolEntry(bias, ctx);

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

  bundle->append("  nonvtaconvolution(");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }
  bundle->append("1.0/");
  bundle->append(std::to_string(inScale));
  bundle->append(", 0, ");

  bundle->append("(int8_t *)VTABufferGetVirtAddr(");
  bundle->append(filterSte.name);
  bundle->append(")");

  bundle->append(", ");
  bundle->append("1.0/");
  bundle->append(std::to_string(filterScale));
  bundle->append(", 0, ");

  bundle->append("(int8_t *)VTABufferGetVirtAddr(");
  bundle->append(biasSte.name);
  bundle->append(")");
  bundle->append(", ");
  bundle->append("1.0/");
  bundle->append(std::to_string(biasScale));
  bundle->append(", 0, ");

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }
  bundle->append("1.0/");
  bundle->append(std::to_string(outScale));
  bundle->append(", 0, ");

  bundle->append(std::to_string(N));
  bundle->append(", ");
  bundle->append(std::to_string(H));
  bundle->append(", ");
  bundle->append(std::to_string(W));
  bundle->append(", ");
  bundle->append(std::to_string(C));
  bundle->append(", ");
  bundle->append(std::to_string(KN));
  bundle->append(", ");
  bundle->append(std::to_string(KH));
  bundle->append(", ");
  bundle->append(std::to_string(KW));
  bundle->append(", ");
  bundle->append(std::to_string(pad_size));
  bundle->append(", ");
  bundle->append(std::to_string(stride_size));
  bundle->append(", ");
  bundle->append(std::to_string(Inst->getGroup()));
  bundle->append(", ");
  bundle->append(std::to_string(Inst->getDilation()));
  bundle->append(", ");
  bundle->append(std::to_string(doRelu));
  bundle->append(", ");
  bundle->append(std::to_string(doBias));
  bundle->append(", ");
  bundle->append(std::to_string(dest->dims()[1]));
  bundle->append(", ");
  bundle->append(std::to_string(dest->dims()[2]));
  bundle->append(" );\n");
}

void saveBNNNonVTAConvolutionInst(
    const glow::ConvolutionInst *Inst, std::string *bundle,
    std::string *initConst,
    VTASaveContext *ctx) { // NON VTA Quantized Convolution
  auto pad = Inst->getPads();
  auto strides = Inst->getStrides();
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto filter = Inst->getFilter();
  auto filterDims = filter->dims();

  int N = srcDims[0];
  int H = srcDims[1];
  int W = srcDims[2];
  int C = srcDims[3];

  int KN = filterDims[0];
  int KH = filterDims[1];
  int KW = filterDims[2];

  auto bias = Inst->getBias();

  int pad_size = pad[0];
  int stride_size = strides[0];

  bool doRelu = Inst->getFusedActivation() == RELU;
  float inScale = src->getType()->getScale();
  float filterScale = filter->getType()->getScale();
  float outScale = Inst->getDest()->getType()->getScale();
  float biasScale = bias->getType()->getScale();

  filterScale = 1 / filterScale;
  inScale = 1 / inScale;
  biasScale = 1 / biasScale;
  outScale = 1 / outScale;

  // save Weight file
  auto vMap = ctx->getVariableMap();
  const Tensor *tensor = NULL;

  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == filter) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensor = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensor);
  auto handle = tensor->getHandle<int8_t>();
  assert(handle.size() % 2 == 0);
  {
    auto fos = ctx->getWeightFileStream();
    int16_t data16 = 0;
    for (size_t i = 0, e = handle.size(); i < e; i++) {
      auto data = handle.raw(i);
      if (data > 127)
        data = 127.0;
      if (data < -128)
        data = -128.0;
      // int8_t clip_data = std::floor(data);
      int8_t sign_data = (int(data > 0) - int(data <= 0));
      if (i % 2 == 0) {
        data16 = 0xff & sign_data;
      } else {
        data16 = data16 | sign_data << 8;
        fos->write((const char *)&data16, 2);
      }
    }
  }

  auto filterSte = addConstantSymbolEntry(filter, ctx);

  // save Bias file

  bool doBias = true;

  const Tensor *tensorBias = NULL;

  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == bias) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensorBias = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensorBias);
  auto handleBias = tensorBias->getHandle<int32_t>();
  assert(handleBias.size() % 2 == 0);
  {
    auto fos = ctx->getWeightFileStream();
    for (size_t i = 0, e = handleBias.size(); i < e; i++) {
      auto data = handleBias.raw(i);
      fos->write((const char *)&data, 4);
    }
  }

  // assert(biasScale == inScale * filterScale);

  auto biasSte = addConstantSymbolEntry(bias, ctx);

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

  bundle->append("  bnn_nonvtaconvolution(");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }

  bundle->append("(int8_t *)VTABufferGetVirtAddr(");
  bundle->append(filterSte.name);
  bundle->append(")");
  bundle->append(", ");

  bundle->append("(int32_t *)VTABufferGetVirtAddr(");
  bundle->append(biasSte.name);
  bundle->append(")");
  bundle->append(", ");

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }

  bundle->append(std::to_string(N));
  bundle->append(", ");
  bundle->append(std::to_string(H));
  bundle->append(", ");
  bundle->append(std::to_string(W));
  bundle->append(", ");
  bundle->append(std::to_string(C));
  bundle->append(", ");
  bundle->append(std::to_string(KN));
  bundle->append(", ");
  bundle->append(std::to_string(KH));
  bundle->append(", ");
  bundle->append(std::to_string(KW));
  bundle->append(", ");
  bundle->append(std::to_string(pad_size));
  bundle->append(", ");
  bundle->append(std::to_string(stride_size));
  bundle->append(", ");
  bundle->append(std::to_string(Inst->getGroup()));
  bundle->append(", ");
  bundle->append(std::to_string(Inst->getDilation()));
  bundle->append(", ");
  bundle->append(std::to_string(doRelu));
  bundle->append(", ");
  bundle->append(std::to_string(doBias));
  bundle->append(", ");
  bundle->append(std::to_string(dest->dims()[1]));
  bundle->append(", ");
  bundle->append(std::to_string(dest->dims()[2]));
  bundle->append(" );\n");
}

void prepareVTAConvolutionInst(const glow::ConvolutionInst *Inst,
                               std::string *bundle, std::string *initConst,
                               VTASaveContext *ctx) {
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto dest = Inst->getDest();
  auto filter = Inst->getFilter();
  auto filterDims = filter->dims();

  auto bias = Inst->getBias();
  assert(bias->dims()[0] == filterDims[0]);
  assert(filterDims[3] == src->dims()[3]);

  float inScale = src->getType()->getScale();
  float filterScale = filter->getType()->getScale();
  float outScale = Inst->getDest()->getType()->getScale();

  assert(Inst->getDest()->getType()->getOffset() == 0);
  assert(src->getType()->getOffset() == 0);
  assert(filter->getType()->getOffset() == 0);
  assert(bias->getType()->getOffset() == 0);

  filterScale = 1 / filterScale;
  inScale = 1 / inScale;
  outScale = 1 / outScale;
  float matMulScale = inScale * filterScale;
  assert(matMulScale / outScale > 1);

  // save Weight file
  auto vMap = ctx->getVariableMap();
  const Tensor *tensor = NULL;
  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == filter) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensor = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensor);
  auto filterType = Type(tensor->getType());
  std::array<dim_t, 6> reshapeFilterS{{(dim_t)filterDims[0] / 16, 16,
                                       (dim_t)filterDims[1], filterDims[2],
                                       filterDims[3] / 16, 16}};
  llvm::ArrayRef<dim_t> reshapeFilterShape(reshapeFilterS);
  auto filterReshapeType = glow::Type::newShape(filterType, reshapeFilterShape);
  // auto filterReshapeDims = filterReshapeType.dims();
  Tensor filterReshapeTensor(filterReshapeType);
  filterReshapeTensor.zero();
  auto handle = tensor->getHandle<int8_t>();
  auto reshapeHandle = filterReshapeTensor.getHandle<int8_t>();

  for (int i = 0; i < handle.size(); i++) {
    auto value = handle.raw(i);
    reshapeHandle.raw(i) = value;
  }

  std::array<unsigned_t, 6> transposeFilterShuffle{{0, 4, 2, 3, 1, 5}};

  std::array<dim_t, 6> transposeFilterS{
      {(dim_t)reshapeFilterShape[0], (dim_t)reshapeFilterShape[2],
       (dim_t)reshapeFilterShape[4], (dim_t)reshapeFilterShape[5],
       (dim_t)reshapeFilterShape[1], (dim_t)reshapeFilterShape[3]}};
  llvm::ArrayRef<dim_t> transposeFilterShape(transposeFilterS);
  auto filterTransposeType =
      glow::Type::newShape(filterReshapeType, transposeFilterShape);
  Tensor filterTransposeTensor(filterTransposeType);

  filterReshapeTensor.transpose(&filterTransposeTensor, transposeFilterShuffle);

  auto transposeHandle = filterTransposeTensor.getHandle<int8_t>();

  assert(handle.size() % 2 == 0);
  {
    auto fos = ctx->getWeightFileStream();
    int16_t data16 = 0;
    for (size_t i = 0, e = transposeHandle.size(); i < e; i++) {
      auto data = transposeHandle.raw(i);
      if (data > 127)
        data = 127.0;
      if (data < -128)
        data = -128.0;
      int8_t clip_data = std::floor(data);
      if (i % 2 == 0) {
        data16 = 0xff & clip_data;
      } else {
        data16 = data16 | clip_data << 8;
        fos->write((const char *)&data16, 2);
      }
    }
  }

  addConstantSymbolEntry(filter, ctx);

  // save Bias file

  bool doBias = false;

  const Tensor *tensorBias = NULL;

  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == bias) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensorBias = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensorBias);
  auto handleBias = tensorBias->getHandle<int32_t>();
  {
    for (size_t i = 0, e = handleBias.size(); i < e; i++) {
      auto data = handleBias.raw(i);
      if (data != 0) {
        doBias = true;
        break;
      }
    }
    if (doBias) {
      auto fos = ctx->getWeightFileStream();
      for (size_t i = 0, e = handleBias.size(); i < e; i++) {
        auto data = handleBias.raw(i);
        fos->write((const char *)&data, 4);
      }
    }
  }
  if (doBias) {
    assert(!doBias || (1 / (bias->getType()->getScale()) == matMulScale));
  }

  addConstantSymbolEntry(bias, ctx);

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }
}

void prepareEVTAConvolutionInst(const glow::VTAConvolutionInst *Inst,
                                std::string *bundle, std::string *initConst,
                                VTASaveContext *ctx) {
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto dest = Inst->getDest();
  auto filter = Inst->getFilter();
  auto filterDims = filter->dims();

  auto bias = Inst->getBias();
  // assert(bias->dims()[0]==filterDims[0]);
  // assert(filterDims[3] == src->dims()[3]);

  float inScale = src->getType()->getScale();
  float filterScale = filter->getType()->getScale();
  float outScale = Inst->getDest()->getType()->getScale();

  // assert(Inst->getDest()->getType()->getOffset() == 0);
  // assert(src->getType()->getOffset() == 0);
  // assert(filter->getType()->getOffset() == 0);
  // assert(bias->getType()->getOffset() == 0);

  filterScale = 1 / filterScale;
  inScale = 1 / inScale;
  outScale = 1 / outScale;
  float matMulScale = inScale * filterScale;
  // assert(matMulScale / outScale > 1);

  // save Weight file
  auto vMap = ctx->getVariableMap();
  const Tensor *tensor = NULL;
  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == filter) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensor = &(constant->getPayload());
      }
      break;
    }
  }
  //    //assert(tensor);
  //    auto filterType = Type(tensor->getType());
  //    std::array<dim_t, 6> reshapeFilterS{{(dim_t)filterDims[0]/16, 16,
  //    (dim_t)filterDims[1], filterDims[2],filterDims[3]/16, 16}};
  //    llvm::ArrayRef<dim_t> reshapeFilterShape(reshapeFilterS);
  //    auto filterReshapeType = glow::Type::newShape(filterType,
  //    reshapeFilterShape);
  //    //auto filterReshapeDims = filterReshapeType.dims();
  //    Tensor filterReshapeTensor(filterReshapeType);
  //    filterReshapeTensor.zero();
  //    auto handle = tensor->getHandle<int8_t>();
  //    auto reshapeHandle = filterReshapeTensor.getHandle<int8_t>();
  //
  //    for(int i = 0; i < handle.size(); i++){
  //        auto value = handle.raw(i);
  //        reshapeHandle.raw(i) = value;
  //    }
  //
  //
  //    std::array<unsigned_t, 6> transposeFilterShuffle{{0, 4, 2, 3, 1, 5}};
  //
  //
  //    std::array<dim_t, 6> transposeFilterS{{(dim_t)reshapeFilterShape[0],
  //    (dim_t)reshapeFilterShape[2], (dim_t)reshapeFilterShape[4],
  //    (dim_t)reshapeFilterShape[5], (dim_t)reshapeFilterShape[1],
  //    (dim_t)reshapeFilterShape[3]}}; llvm::ArrayRef<dim_t>
  //    transposeFilterShape(transposeFilterS); auto filterTransposeType =
  //    glow::Type::newShape(filterReshapeType, transposeFilterShape); Tensor
  //    filterTransposeTensor(filterTransposeType);
  //
  //    filterReshapeTensor.transpose(&filterTransposeTensor,
  //    transposeFilterShuffle);
  //
  //    auto transposeHandle = filterTransposeTensor.getHandle<int8_t>();
  auto transposeHandle = tensor->getHandle<int8_t>();

  // assert(handle.size() %2 == 0);
  {
    auto fos = ctx->getWeightFileStream();
    int16_t data16 = 0;
    for (size_t i = 0, e = transposeHandle.size(); i < e; i++) {
      auto data = transposeHandle.raw(i);
      if (data > 127)
        data = 127.0;
      if (data < -128)
        data = -128.0;
      int8_t clip_data = std::floor(data);
      if (i % 2 == 0) {
        data16 = 0xff & clip_data;
      } else {
        data16 = data16 | clip_data << 8;
        fos->write((const char *)&data16, 2);
      }
    }
  }

  addConstantSymbolEntry(filter, ctx);

  // save Bias file

  bool doBias = false;

  const Tensor *tensorBias = NULL;

  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == bias) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensorBias = &(constant->getPayload());
      }
      break;
    }
  }
  // assert(tensorBias);
  auto handleBias = tensorBias->getHandle<int32_t>();
  {
    for (size_t i = 0, e = handleBias.size(); i < e; i++) {
      auto data = handleBias.raw(i);
      if (data != 0) {
        doBias = true;
        break;
      }
    }
    if (doBias) {
      auto fos = ctx->getWeightFileStream();
      for (size_t i = 0, e = handleBias.size(); i < e; i++) {
        auto data = handleBias.raw(i);
        fos->write((const char *)&data, 4);
      }
    }
  }
  //    if(doBias){
  //        assert(!doBias || (1/(bias->getType()->getScale()) == matMulScale));
  //    }

  addConstantSymbolEntry(bias, ctx);

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }
}

void prepareBNNVTAConvolutionInst(const glow::ConvolutionInst *Inst,
                                  std::string *bundle, std::string *initConst,
                                  VTASaveContext *ctx) {
  auto src = Inst->getSrc();
  auto dest = Inst->getDest();
  auto filter = Inst->getFilter();
  auto filterDims = filter->dims();

  assert(filterDims[3] % 128 == 0);

  auto bias = Inst->getBias();
  assert(bias->dims()[0] == filterDims[0]);
  assert(filterDims[3] == src->dims()[3]);

  float inScale = src->getType()->getScale();
  float filterScale = filter->getType()->getScale();
  float outScale = Inst->getDest()->getType()->getScale();
  filterScale = 1 / filterScale;
  inScale = 1 / inScale;
  outScale = 1 / outScale;
  float matMulScale = inScale * filterScale;

  // save Weight file
  auto vMap = ctx->getVariableMap();
  const Tensor *tensor = NULL;
  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == filter) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensor = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensor);

  auto filterType = Type(tensor->getType());
  std::array<dim_t, 4> signFilterS{{(dim_t)filterDims[0], (dim_t)filterDims[1],
                                    (dim_t)filterDims[2],
                                    (dim_t)((int)filterDims[3] / 8)}};
  llvm::ArrayRef<dim_t> signFilterShape(signFilterS);
  auto signFilterShapeType = glow::Type::newShape(filterType, signFilterShape);
  Tensor signFilterTensor(signFilterShapeType);
  signFilterTensor.zero();
  auto handle = tensor->getHandle<int8_t>();
  auto signHandle = signFilterTensor.getHandle<int8_t>();

  auto packN = signFilterS[0];
  auto packH = signFilterS[1];
  auto packW = signFilterS[2];
  auto packC = signFilterS[3];
  for (dim_t b = 0; b < packN; ++b) {
    for (dim_t h = 0; h < packH; ++h) {
      for (dim_t w = 0; w < packW; ++w) {
        for (dim_t c = 0; c < packC; ++c) {
          for (dim_t bw = 0; bw < 8; ++bw) {
            signHandle.at({b, h, w, c}) +=
                ((handle.at({b, h, w, (c * 8 + bw)}) > 0) << bw);
          }
        }
      }
    }
  }

  std::array<dim_t, 6> reshapeFilterS{{(dim_t)signFilterS[0] / 16, 16,
                                       (dim_t)signFilterS[1], signFilterS[2],
                                       signFilterS[3] / 16, 16}};
  llvm::ArrayRef<dim_t> reshapeFilterShape(reshapeFilterS);
  auto filterReshapeType = glow::Type::newShape(filterType, reshapeFilterShape);
  Tensor filterReshapeTensor(filterReshapeType);
  filterReshapeTensor.zero();
  auto reshapeHandle = filterReshapeTensor.getHandle<int8_t>();

  for (int i = 0; i < signHandle.size(); i++) {
    auto value = signHandle.raw(i);
    reshapeHandle.raw(i) = value;
  }

  std::array<unsigned_t, 6> transposeFilterShuffle{{0, 4, 2, 3, 1, 5}};
  std::array<dim_t, 6> transposeFilterS{
      {(dim_t)reshapeFilterShape[0], (dim_t)reshapeFilterShape[2],
       (dim_t)reshapeFilterShape[4], (dim_t)reshapeFilterShape[5],
       (dim_t)reshapeFilterShape[1], (dim_t)reshapeFilterShape[3]}};
  llvm::ArrayRef<dim_t> transposeFilterShape(transposeFilterS);
  auto filterTransposeType =
      glow::Type::newShape(filterReshapeType, transposeFilterShape);
  Tensor filterTransposeTensor(filterTransposeType);

  filterReshapeTensor.transpose(&filterTransposeTensor, transposeFilterShuffle);

  auto transposeHandle = filterTransposeTensor.getHandle<int8_t>();

  assert(handle.size() % 2 == 0);
  {
    auto fos = ctx->getWeightFileStream();
    int16_t data16 = 0;
    for (size_t i = 0, e = transposeHandle.size(); i < e; i++) {
      auto data = transposeHandle.raw(i);
      if (data > 127)
        data = 127.0;
      if (data < -128)
        data = -128.0;
      int8_t clip_data = std::floor(data);
      if (i % 2 == 0) {
        data16 = 0xff & clip_data;
      } else {
        data16 = data16 | clip_data << 8;
        fos->write((const char *)&data16, 2);
      }
    }
  }
  addConstantSymbolEntry(filter, ctx);

  // save Bias file
  bool doBias = false;

  const Tensor *tensorBias = NULL;

  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == bias) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensorBias = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensorBias);
  auto handleBias = tensorBias->getHandle<int32_t>();
  {
    for (size_t i = 0, e = handleBias.size(); i < e; i++) {
      auto data = handleBias.raw(i);
      if (data != 0) {
        doBias = true;
        break;
      }
    }
    if (doBias) {
      auto fos = ctx->getWeightFileStream();
      for (size_t i = 0, e = handleBias.size(); i < e; i++) {
        auto data = handleBias.raw(i);
        fos->write((const char *)&data, 4);
      }
    }
  }
  addConstantSymbolEntry(bias, ctx);

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }
}

void generateInputTranspose(const glow::ConvolutionInst *Inst,
                            std::string *bundle, std::string *initConst,
                            VTASaveContext *ctx) {
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto filter = Inst->getFilter();

  int N = srcDims[0];
  int H = srcDims[1];
  int W = srcDims[2];
  int C = srcDims[3];
#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose_start = clock();\n");
#endif

  bundle->append("  int8_t* ");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose = (int8_t *)VTABufferAlloc(");
  bundle->append(std::to_string(N * H * W * C));
  bundle->append(");\n");
  bundle->append("  transpose_nhwc2vtaio(");
  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }

  bundle->append("(int8_t* )VTABufferGetVirtAddr(");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose), ");

  bundle->append(std::to_string(N));
  bundle->append(", ");
  bundle->append(std::to_string(H));
  bundle->append(", ");
  bundle->append(std::to_string(W));
  bundle->append(", ");
  bundle->append(std::to_string(C));
  bundle->append(");\n");

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose_end = clock();\n");
  bundle->append("  prof_out<<\"");
  bundle->append(Inst->getKindName());
  bundle->append("_input_transpose:");
  bundle->append(Inst->getName());
  bundle->append(" : \"<<");
  bundle->append("(double)(");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose_end - ");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose_start)/CLOCKS_PER_SEC*1000 << std::endl;\n");
#endif
}

void generateEVTAInputTranspose(const glow::VTAConvolutionInst *Inst,
                                std::string *bundle, std::string *initConst,
                                VTASaveContext *ctx) {
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto filter = Inst->getFilter();

  int Nm = srcDims[0];
  int Cm = srcDims[1];
  int H = srcDims[2];
  int W = srcDims[3];
  int Ns = srcDims[4];
  int Cs = srcDims[5];

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose_start = clock();\n");
#endif

  //  VTABufferCopy(transposeInput_res, 0, conv__1_input_transpose, 0, 6272, 1);

  bundle->append("  int8_t* ");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose = (int8_t *)VTABufferAlloc(");
  bundle->append(std::to_string(Nm * Cm * H * W * Ns * Cs));
  bundle->append(");\n");
  bundle->append("  VTABufferCopy(");
  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }
  bundle->append(std::to_string(0));
  bundle->append(", ");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose, ");

  bundle->append(std::to_string(0));
  bundle->append(", ");
  bundle->append(std::to_string(Nm * Cm * H * W * Ns * Cs));
  bundle->append(", ");
  bundle->append(std::to_string(1));
  bundle->append(");\n");

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose_end = clock();\n");
  bundle->append("  prof_out<<\"");
  bundle->append(Inst->getKindName());
  bundle->append("_input_transpose:");
  bundle->append(Inst->getName());
  bundle->append(" : \"<<");
  bundle->append("(double)(");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose_end - ");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose_start)/CLOCKS_PER_SEC*1000 << std::endl;\n");
#endif
}

void generateBNNInputTranspose(const glow::ConvolutionInst *Inst,
                               std::string *bundle, std::string *initConst,
                               VTASaveContext *ctx) {
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto filter = Inst->getFilter();

  int N = srcDims[0];
  int H = srcDims[1];
  int W = srcDims[2];
  int C = srcDims[3];
  C = C / 8;
#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose_start = clock();\n");
#endif

  bundle->append("  int8_t* ");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose = (int8_t *)VTABufferAlloc(");
  bundle->append(std::to_string(N * H * W * C));
  bundle->append(");\n");
  bundle->append("  transpose_nhwc2vtaio_pack(");
  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }

  bundle->append("(int8_t* )VTABufferGetVirtAddr(");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose), ");

  bundle->append(std::to_string(N));
  bundle->append(", ");
  bundle->append(std::to_string(H));
  bundle->append(", ");
  bundle->append(std::to_string(W));
  bundle->append(", ");
  bundle->append(std::to_string(C));
  bundle->append(");\n");

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose_end = clock();\n");
  bundle->append("  prof_out<<\"");
  bundle->append(Inst->getKindName());
  bundle->append("_input_transpose:");
  bundle->append(Inst->getName());
  bundle->append(" : \"<<");
  bundle->append("(double)(");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose_end - ");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose_start)/CLOCKS_PER_SEC*1000 << std::endl;\n");
#endif
}

void declareOutputTranspose(const glow::ConvolutionInst *Inst,
                            std::string *bundle, std::string *initConst) {
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto dest = Inst->getDest();
  auto destDims = dest->dims();
  int outN = destDims[0];
  int outH = destDims[1];
  int outW = destDims[2];
  int outC = destDims[3];

  bundle->append("  int8_t* ");
  bundle->append(Inst->getName());
  bundle->append("_output_bef_transpose = (int8_t *)VTABufferAlloc(");
  bundle->append(std::to_string(outN * outH * outW * outC));
  bundle->append(");\n");
}

void declareEVTAOutputTranspose(const glow::VTAConvolutionInst *Inst,
                                std::string *bundle, std::string *initConst) {
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto dest = Inst->getDest();
  auto destDims = dest->dims();
  int Nm = destDims[0];
  int Cm = destDims[1];
  int H = destDims[2];
  int W = destDims[3];
  int Ns = destDims[4];
  int Cs = destDims[5];

  bundle->append("  int8_t* ");
  bundle->append(Inst->getName());
  bundle->append("_output_bef_transpose = (int8_t *)VTABufferAlloc(");
  bundle->append(std::to_string(Nm * Cm * H * W * Ns * Cs));
  bundle->append(");\n");
}

void generateVTAConvolutionCall(const glow::ConvolutionInst *Inst,
                                std::string *bundle, std::string *initConst,
                                VTASaveContext *ctx) {
  auto pad = Inst->getPads();
  auto strides = Inst->getStrides();
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto filter = Inst->getFilter();
  auto filterDims = filter->dims();
  auto bias = Inst->getBias();
  int N = srcDims[0];
  int H = srcDims[1];
  int W = srcDims[2];
  int C = srcDims[3];
  int KN = filterDims[0];
  int KH = filterDims[1];
  int KW = filterDims[2];
  int pad_size = pad[0];
  int stride_size = strides[0];
  bool doRelu = Inst->getFusedActivation() == RELU;

  bool doBias = false;

  const Tensor *tensorBias = NULL;
  auto vMap = ctx->getVariableMap();
  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == bias) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensorBias = &(constant->getPayload());
      }
      break;
    }
  }
  auto handleBias = tensorBias->getHandle<int32_t>();
  {
    for (size_t i = 0, e = handleBias.size(); i < e; i++) {
      auto data = handleBias.raw(i);
      if (data != 0) {
        doBias = true;
        break;
      }
    }
  }

  float inScale = src->getType()->getScale();
  float filterScale = filter->getType()->getScale();
  float outScale = Inst->getDest()->getType()->getScale();
  filterScale = 1 / filterScale;
  inScale = 1 / inScale;
  outScale = 1 / outScale;
  float matMulScale = inScale * filterScale;
  float scale = matMulScale / outScale;
  int shift = getExpofPowerofTwo(scale);

  auto filterSte = addConstantSymbolEntry(filter, ctx);
  bundle->append("  convolution_wo_tr(");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose, ");
  bundle->append(filterSte.name);
  bundle->append(", ");

  bundle->append("(int32_t *)");
  auto biasSte = addConstantSymbolEntry(bias, ctx);
  bundle->append(biasSte.name);
  bundle->append(", ");

  bundle->append(Inst->getName());
  bundle->append("_output_bef_transpose, ");

  bundle->append(std::to_string(N));
  bundle->append(", ");
  bundle->append(std::to_string(H));
  bundle->append(", ");
  bundle->append(std::to_string(W));
  bundle->append(", ");
  bundle->append(std::to_string(C));
  bundle->append(", ");
  bundle->append(std::to_string(KN));
  bundle->append(", ");
  bundle->append(std::to_string(KH));
  bundle->append(", ");
  bundle->append(std::to_string(KW));
  bundle->append(", ");
  bundle->append(std::to_string(pad_size));
  bundle->append(", ");
  bundle->append(std::to_string(stride_size));
  bundle->append(", ");
  bundle->append(std::to_string(doRelu));
  bundle->append(", ");
  bundle->append(std::to_string(doBias));
  bundle->append(", ");
  bundle->append(std::to_string(shift));
  bundle->append(", ");
  bundle->append(std::to_string(dest->dims()[1]));
  bundle->append(", ");
  bundle->append(std::to_string(dest->dims()[2]));
  bundle->append(", vtaCmdH");
  uint32_t idxMultiEVTA = ctx->getIdxMultiEVTA();
  if (idxMultiEVTA) {
    bundle->append(std::to_string(idxMultiEVTA));
  }
#ifdef NESTC_EVTA_PROFILE_AUTOTUNE
  bundle->append(", input1, input2, input3, ");
  bundle->append(std::to_string(ctx->getIdxMultiEVTA()));
#else
#include "VTASchedules.h"
  bool isTuned = 0;
  for (auto elem : convTune.ConvolutionTune_) {
    if (elem.isMatch(N, H, W, C, KN, KH, KW, C, stride_size, pad_size)) {
      bundle->append(", ");
      bundle->append(std::to_string(elem.nVirtualThread_));
      bundle->append(", ");
      bundle->append(std::to_string(elem.tileHSize_));
      bundle->append(", ");
      bundle->append(std::to_string(elem.tileWSize_));
      bundle->append(", ");
      bundle->append(std::to_string(ctx->getIdxMultiEVTA()));
      isTuned = 1;
      break;
    }
  }

  if (!isTuned) {
    bundle->append(", 1, 14, 14, ");
    bundle->append(std::to_string(ctx->getIdxMultiEVTA()));
  }
#endif
  bundle->append(");\n");
}

void generateEVTAConvolutionCall(const glow::VTAConvolutionInst *Inst,
                                 std::string *bundle, std::string *initConst,
                                 VTASaveContext *ctx) {
  auto pad = Inst->getPads();
  auto strides = Inst->getStrides();
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto filter = Inst->getFilter();
  auto filterDims = filter->dims();
  auto bias = Inst->getBias();
  int Nm = srcDims[0];
  int Cm = srcDims[1];
  int H = srcDims[2];
  int W = srcDims[3];
  int Ns = srcDims[4];
  int Cs = srcDims[5];

  int KNm = filterDims[0];
  int KCm = filterDims[1];
  int KH = filterDims[2];
  int KW = filterDims[3];
  int KNs = filterDims[4];

  int pad_size = pad[0];
  int stride_size = strides[0];
  bool doRelu = Inst->getFusedActivation() == RELU;

  bool doBias = false;

  const Tensor *tensorBias = NULL;
  auto vMap = ctx->getVariableMap();
  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == bias) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensorBias = &(constant->getPayload());
      }
      break;
    }
  }
  auto handleBias = tensorBias->getHandle<int32_t>();
  {
    for (size_t i = 0, e = handleBias.size(); i < e; i++) {
      auto data = handleBias.raw(i);
      if (data != 0) {
        doBias = true;
        break;
      }
    }
  }

  float inScale = src->getType()->getScale();
  float filterScale = filter->getType()->getScale();
  float outScale = Inst->getDest()->getType()->getScale();
  filterScale = 1 / filterScale;
  inScale = 1 / inScale;
  outScale = 1 / outScale;
  float matMulScale = inScale * filterScale;
  float scale = matMulScale / outScale;
  int shift = getExpofPowerofTwo(scale);

  auto filterSte = addConstantSymbolEntry(filter, ctx);
  bundle->append("  convolution_wo_tr(");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose, ");
  bundle->append(filterSte.name);
  bundle->append(", ");

  bundle->append("(int32_t *)");
  auto biasSte = addConstantSymbolEntry(bias, ctx);
  bundle->append(biasSte.name);
  bundle->append(", ");

  bundle->append(Inst->getName());
  bundle->append("_output_bef_transpose, ");

  bundle->append(std::to_string(Nm * Ns));
  bundle->append(", ");
  bundle->append(std::to_string(H));
  bundle->append(", ");
  bundle->append(std::to_string(W));
  bundle->append(", ");
  bundle->append(std::to_string(Cm * Cs));
  bundle->append(", ");
  bundle->append(std::to_string(KNm * KNs));
  bundle->append(", ");
  bundle->append(std::to_string(KH));
  bundle->append(", ");
  bundle->append(std::to_string(KW));
  bundle->append(", ");
  bundle->append(std::to_string(pad_size));
  bundle->append(", ");
  bundle->append(std::to_string(stride_size));
  bundle->append(", ");
  bundle->append(std::to_string(doRelu));
  bundle->append(", ");
  bundle->append(std::to_string(doBias));
  bundle->append(", ");
  bundle->append(std::to_string(shift));
  bundle->append(", ");
  bundle->append(std::to_string(dest->dims()[2]));
  bundle->append(", ");
  bundle->append(std::to_string(dest->dims()[3]));
  bundle->append(", vtaCmdH");
  uint32_t idxMultiEVTA = ctx->getIdxMultiEVTA();
  if (idxMultiEVTA) {
    bundle->append(std::to_string(idxMultiEVTA));
  }
#include "VTASchedules.h"
  bool isTuned = 0;
  for (auto elem : convTune.ConvolutionTune_) {
    if (elem.isMatch(Nm * Ns, H, W, Cm * Cs, KNm * KNs, KH, KW, Cm * Cs,
                     stride_size, pad_size)) {
      bundle->append(", ");
      bundle->append(std::to_string(elem.nVirtualThread_));
      bundle->append(", ");
      bundle->append(std::to_string(elem.tileHSize_));
      bundle->append(", ");
      bundle->append(std::to_string(elem.tileWSize_));
      bundle->append(", ");
      bundle->append(std::to_string(ctx->getIdxMultiEVTA()));
      isTuned = 1;
      break;
    }
  }

  if (!isTuned) {
    bundle->append(", 1, 14, 14, ");
    bundle->append(std::to_string(ctx->getIdxMultiEVTA()));
  }

  bundle->append(");\n");
}

void generateBNNVTAConvolutionCall(const glow::ConvolutionInst *Inst,
                                   std::string *bundle, std::string *initConst,
                                   VTASaveContext *ctx) {
  auto pad = Inst->getPads();
  auto strides = Inst->getStrides();
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto filter = Inst->getFilter();
  auto filterDims = filter->dims();
  auto bias = Inst->getBias();
  int N = srcDims[0];
  int H = srcDims[1];
  int W = srcDims[2];
  int C = srcDims[3];

  C = C / 8;

  int KN = filterDims[0];
  int KH = filterDims[1];
  int KW = filterDims[2];
  int pad_size = pad[0];
  int stride_size = strides[0];
  bool doRelu = Inst->getFusedActivation() == RELU;

  bool doBias = false;

  const Tensor *tensorBias = NULL;
  auto vMap = ctx->getVariableMap();
  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == bias) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensorBias = &(constant->getPayload());
      }
      break;
    }
  }
  auto handleBias = tensorBias->getHandle<int32_t>();
  {
    for (size_t i = 0, e = handleBias.size(); i < e; i++) {
      auto data = handleBias.raw(i);
      if (data != 0) {
        doBias = true;
        break;
      }
    }
  }

  float inScale = src->getType()->getScale();
  float filterScale = filter->getType()->getScale();
  float outScale = Inst->getDest()->getType()->getScale();
  filterScale = 1 / filterScale;
  inScale = 1 / inScale;
  outScale = 1 / outScale;
  float matMulScale = inScale * filterScale;
  float scale = matMulScale / outScale;
  int shift = getExpofPowerofTwo(scale);

  auto filterSte = addConstantSymbolEntry(filter, ctx);
  bundle->append("  convolution_wo_tr(");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose, ");
  bundle->append(filterSte.name);
  bundle->append(", ");

  bundle->append("(int32_t *)");
  auto biasSte = addConstantSymbolEntry(bias, ctx);
  bundle->append(biasSte.name);
  bundle->append(", ");

  bundle->append(Inst->getName());
  bundle->append("_output_bef_transpose, ");

  bundle->append(std::to_string(N));
  bundle->append(", ");
  bundle->append(std::to_string(H));
  bundle->append(", ");
  bundle->append(std::to_string(W));
  bundle->append(", ");
  bundle->append(std::to_string(C));
  bundle->append(", ");
  bundle->append(std::to_string(KN));
  bundle->append(", ");
  bundle->append(std::to_string(KH));
  bundle->append(", ");
  bundle->append(std::to_string(KW));
  bundle->append(", ");
  bundle->append(std::to_string(pad_size));
  bundle->append(", ");
  bundle->append(std::to_string(stride_size));
  bundle->append(", ");
  bundle->append(std::to_string(doRelu));
  bundle->append(", ");
  bundle->append(std::to_string(doBias));
  bundle->append(", ");
  bundle->append("7"); // shift
  bundle->append(", ");
  bundle->append(std::to_string(dest->dims()[1]));
  bundle->append(", ");
  bundle->append(std::to_string(dest->dims()[2]));
  bundle->append(", vtaCmdH");
  uint32_t idxMultiEVTA = ctx->getIdxMultiEVTA();
  if (idxMultiEVTA) {
    bundle->append(std::to_string(idxMultiEVTA));
  }
#include "VTASchedules.h"
  bool isTuned = 0;
  for (auto elem : convTune.ConvolutionTune_) {
    if (elem.isMatch(N, H, W, C, KN, KH, KW, C, stride_size, pad_size)) {
      bundle->append(", ");
      bundle->append(std::to_string(elem.nVirtualThread_));
      bundle->append(", ");
      bundle->append(std::to_string(elem.tileHSize_));
      bundle->append(", ");
      bundle->append(std::to_string(elem.tileWSize_));
      bundle->append(", ");
      bundle->append(std::to_string(ctx->getIdxMultiEVTA()));
      isTuned = 1;
      break;
    }
  }

  if (!isTuned) {
    if (dest->dims()[1] != 8) {
      bundle->append(", 1, 8, 8, ");
    } else if (dest->dims()[1] == 8) {
      bundle->append(", 1, 8, 7, ");
    }
    bundle->append(std::to_string(ctx->getIdxMultiEVTA()));
  }

  bundle->append(");\n");
}

void generateOutputTranspose(const glow::ConvolutionInst *Inst,
                             std::string *bundle, std::string *initConst,
                             VTASaveContext *ctx) {
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto dest = Inst->getDest();
  auto destDims = dest->dims();
  auto filter = Inst->getFilter();

  int outN = destDims[0];
  int outH = destDims[1];
  int outW = destDims[2];
  int outC = destDims[3];

  auto destWeight = static_cast<WeightVar *>(dest);

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_output_transpose_start = clock();\n");
#endif
  bundle->append("  transpose_vtaio2nhwc(");
  bundle->append("(int8_t* )VTABufferGetVirtAddr(");
  bundle->append(Inst->getName());
  bundle->append("_output_bef_transpose), ");

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }
  bundle->append(std::to_string(outN));
  bundle->append(", ");
  bundle->append(std::to_string(outH));
  bundle->append(", ");
  bundle->append(std::to_string(outW));
  bundle->append(", ");
  bundle->append(std::to_string(outC));
  bundle->append(" );\n");

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_output_transpose_end = clock();\n");
  bundle->append("  prof_out<<\"");
  bundle->append(Inst->getKindName());
  bundle->append("_output_transpose:");
  bundle->append(Inst->getName());
  bundle->append(" : \"<<");
  bundle->append("(double)(");
  bundle->append(Inst->getName());
  bundle->append("_output_transpose_end - ");
  bundle->append(Inst->getName());
  bundle->append(
      "_output_transpose_start)/CLOCKS_PER_SEC*1000 << std::endl;\n");
#endif
}

void generateEVTAOutputTranspose(const glow::VTAConvolutionInst *Inst,
                                 std::string *bundle, std::string *initConst,
                                 VTASaveContext *ctx) {
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto dest = Inst->getDest();
  auto destDims = dest->dims();
  auto filter = Inst->getFilter();

  int Nm = destDims[0];
  int Cm = destDims[1];
  int H = destDims[2];
  int W = destDims[3];
  int Ns = destDims[4];
  int Cs = destDims[5];

  auto destWeight = static_cast<WeightVar *>(dest);

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_output_transpose_start = clock();\n");
#endif
  bundle->append("  VTABufferCopy(");
  bundle->append(Inst->getName());
  bundle->append("_output_bef_transpose, ");
  bundle->append(std::to_string(0));
  bundle->append(", ");

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }
  bundle->append(std::to_string(0));
  bundle->append(", ");
  bundle->append(std::to_string(Nm * Cm * H * W * Ns * Cs));
  bundle->append(", ");
  bundle->append(std::to_string(2));
  bundle->append(");\n");

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_output_transpose_end = clock();\n");
  bundle->append("  prof_out<<\"");
  bundle->append(Inst->getKindName());
  bundle->append("_output_transpose:");
  bundle->append(Inst->getName());
  bundle->append(" : \"<<");
  bundle->append("(double)(");
  bundle->append(Inst->getName());
  bundle->append("_output_transpose_end - ");
  bundle->append(Inst->getName());
  bundle->append(
      "_output_transpose_start)/CLOCKS_PER_SEC*1000 << std::endl;\n");
#endif
}

void generateVTABufferFree(const glow::ConvolutionInst *Inst,
                           std::string *bundle) {
  bundle->append("  VTABufferFree(");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose);\n");

  bundle->append("  VTABufferFree(");
  bundle->append(Inst->getName());
  bundle->append("_output_bef_transpose);\n");
}

void generateEVTABufferFree(const glow::VTAConvolutionInst *Inst,
                            std::string *bundle) {
  bundle->append("  VTABufferFree(");
  bundle->append(Inst->getName());
  bundle->append("_input_transpose);\n");

  bundle->append("  VTABufferFree(");
  bundle->append(Inst->getName());
  bundle->append("_output_bef_transpose);\n");
}

void saveVTAConvolutionInst(const glow::ConvolutionInst *Inst,
                            std::string *bundle, std::string *initConst,
                            VTASaveContext *ctx) {
  prepareVTAConvolutionInst(Inst, bundle, initConst, ctx);
  generateInputTranspose(Inst, bundle, initConst, ctx);
  declareOutputTranspose(Inst, bundle, initConst);

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_core_start = clock();\n");
#endif

  generateVTAConvolutionCall(Inst, bundle, initConst, ctx);

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_core_end = clock();\n");
  bundle->append("  prof_out<<\"");
  bundle->append(Inst->getKindName());
  bundle->append("_core:");
  bundle->append(Inst->getName());
  bundle->append(" : \"<<");
  bundle->append("(double)(");
  bundle->append(Inst->getName());
  bundle->append("_core_end - ");
  bundle->append(Inst->getName());
  bundle->append("_core_start)/CLOCKS_PER_SEC*1000 << std::endl;\n");
#endif

  generateOutputTranspose(Inst, bundle, initConst, ctx);
  generateVTABufferFree(Inst, bundle);

  return;
}
void saveEVTAConvolutionInst(const glow::VTAConvolutionInst *Inst,
                             std::string *bundle, std::string *initConst,
                             VTASaveContext *ctx) {

  prepareEVTAConvolutionInst(Inst, bundle, initConst, ctx);
  generateEVTAInputTranspose(Inst, bundle, initConst, ctx);
  declareEVTAOutputTranspose(Inst, bundle, initConst);

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_core_start = clock();\n");
#endif

  generateEVTAConvolutionCall(Inst, bundle, initConst, ctx);

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_core_end = clock();\n");
  bundle->append("  prof_out<<\"");
  bundle->append(Inst->getKindName());
  bundle->append("_core:");
  bundle->append(Inst->getName());
  bundle->append(" : \"<<");
  bundle->append("(double)(");
  bundle->append(Inst->getName());
  bundle->append("_core_end - ");
  bundle->append(Inst->getName());
  bundle->append("_core_start)/CLOCKS_PER_SEC*1000 << std::endl;\n");
#endif

  generateEVTAOutputTranspose(Inst, bundle, initConst, ctx);
  generateEVTABufferFree(Inst, bundle);

  return;
}

void saveBNNVTAConvolutionInst(const glow::ConvolutionInst *Inst,
                               std::string *bundle, std::string *initConst,
                               VTASaveContext *ctx) {
  prepareBNNVTAConvolutionInst(Inst, bundle, initConst, ctx);
  generateBNNInputTranspose(Inst, bundle, initConst, ctx);
  declareOutputTranspose(Inst, bundle, initConst);

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("vta_core_start = clock();\n");
#endif

  generateBNNVTAConvolutionCall(Inst, bundle, initConst, ctx);

#ifdef VTA_PROFILE
  bundle->append("  clock_t ");
  bundle->append(Inst->getName());
  bundle->append("_core_end = clock();\n");
  bundle->append("  prof_out<<\"");
  bundle->append(Inst->getKindName());
  bundle->append("_core:");
  bundle->append(Inst->getName());
  bundle->append(" : \"<<");
  bundle->append("(double)(");
  bundle->append(Inst->getName());
  bundle->append("_core_end - ");
  bundle->append(Inst->getName());
  bundle->append("_core_start)/CLOCKS_PER_SEC*1000 << std::endl;\n");
#endif

  generateOutputTranspose(Inst, bundle, initConst, ctx);
  generateVTABufferFree(Inst, bundle);

  return;
}

void saveConvolutionInst(const glow::ConvolutionInst *Inst, std::string *bundle,
                         std::string *initConst, VTASaveContext *ctx) {
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto filter = Inst->getFilter();
  auto filterDims = filter->dims();

  auto bias = Inst->getBias();
  assert(bias->dims()[0] == filterDims[0]);

  if (!Inst->getSrc()->getType()->isQuantizedType()) {
    saveFloatConvolutionInst(Inst, bundle, ctx);
  } else {
    float inScale = src->getType()->getScale();
    float filterScale = filter->getType()->getScale();
    float outScale = Inst->getDest()->getType()->getScale();
    float biasScale = bias->getType()->getScale();

    int32_t outOffset = Inst->getDest()->getType()->getOffset();
    int32_t inOffset = src->getType()->getOffset();
    int32_t filterOffset = filter->getType()->getOffset();
    int32_t biasOffset = bias->getType()->getOffset();

    filterScale = 1 / filterScale;
    inScale = 1 / inScale;
    biasScale = 1 / biasScale;
    outScale = 1 / outScale;
    float matMulScale = inScale * filterScale;
    float scale = matMulScale / outScale;

    int shift = getExpofPowerofTwo(scale);

    if (Inst->getGroup() == 1 && scale > 1 && srcDims[3] % 16 == 0 &&
        filterDims[0] % 16 == 0 && biasScale == matMulScale && outOffset == 0 &&
        inOffset == 0 && filterOffset == 0 && biasOffset == 0 && shift >= 0) {
      saveVTAConvolutionInst(Inst, bundle, initConst, ctx);
    } else if (Inst->getGroup() == srcDims[3] && scale > 1 &&
               biasScale == matMulScale && outOffset == 0 && inOffset == 0 &&
               filterOffset == 0 && biasOffset == 0 && shift >= 0) {
      // saveVTADWConvolutionInst(Inst, bundle, initConst, ctx);
      saveNonVTAConvolutionInst(Inst, bundle, initConst, ctx);
    } else {
      saveNonVTAConvolutionInst(Inst, bundle, initConst, ctx);
    }
  }
}

void saveBNNConvolutionInst(const glow::ConvolutionInst *Inst,
                            std::string *bundle, std::string *initConst,
                            VTASaveContext *ctx) {
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto filter = Inst->getFilter();
  auto filterDims = filter->dims();

  auto bias = Inst->getBias();
  assert(bias->dims()[0] == filterDims[0]);

  if (!Inst->getSrc()->getType()->isQuantizedType()) {
    saveFloatConvolutionInst(Inst, bundle, ctx);
  } else {
    float inScale = src->getType()->getScale();
    float filterScale = filter->getType()->getScale();
    float outScale = Inst->getDest()->getType()->getScale();
    float biasScale = bias->getType()->getScale();

    int32_t outOffset = Inst->getDest()->getType()->getOffset();
    int32_t inOffset = src->getType()->getOffset();
    int32_t filterOffset = filter->getType()->getOffset();
    int32_t biasOffset = bias->getType()->getOffset();

    filterScale = 1 / filterScale;
    inScale = 1 / inScale;
    biasScale = 1 / biasScale;
    outScale = 1 / outScale;

    float matMulScale = inScale * filterScale;
    float scale = matMulScale / outScale;

    int shift = getExpofPowerofTwo(scale);

    if (Inst->getGroup() == 1 && srcDims[3] % 128 == 0 &&
        filterDims[0] % 128 == 0) {
      saveBNNVTAConvolutionInst(Inst, bundle, initConst, ctx);
    } else if (srcDims[3] == 3 && filterDims[3] == 3 &&
               biasScale == matMulScale && outOffset == 0 && inOffset == 0 &&
               filterOffset == 0 && biasOffset == 0 && shift >= 0) {
      // savezerofillVTAConvolutionInst(Inst, bundle, initConst, ctx);
      saveBNNNonVTAConvolutionInst(Inst, bundle, initConst, ctx);
    } else {
      saveBNNNonVTAConvolutionInst(Inst, bundle, initConst, ctx);
    }
  }
}

void saveFullyConnectedInst(const glow::FullyConnectedInst *Inst,
                            std::string *bundle, VTASaveContext *ctx) {
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto destDims = dest->dims();
  auto filter = Inst->getWeights();
  auto filterDims = filter->dims();
  auto bias = Inst->getBias();

  assert(srcDims.size() == 2);
  assert(bias->dims().size() == 1);
  assert(filterDims.size() == 2);
  assert(destDims.size() == 2);

  float inScale = src->getType()->getScale();
  float filterScale = filter->getType()->getScale();
  float outScale = Inst->getDest()->getType()->getScale();
  float biasScale = bias->getType()->getScale();

  filterScale = 1 / filterScale;
  inScale = 1 / inScale;
  biasScale = 1 / biasScale;
  outScale = 1 / outScale;

  // save Weight file
  auto vMap = ctx->getVariableMap();
  const Tensor *tensor = NULL;

  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == filter) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensor = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensor);
  auto handle = tensor->getHandle<int8_t>();
  assert(handle.size() % 2 == 0);
  {
    auto fos = ctx->getWeightFileStream();
    int16_t data16 = 0;
    for (size_t i = 0, e = handle.size(); i < e; i++) {
      auto data = handle.raw(i);
      if (data > 127)
        data = 127.0;
      if (data < -128)
        data = -128.0;
      int8_t clip_data = std::floor(data);
      if (i % 2 == 0) {
        data16 = 0xff & clip_data;
      } else {
        data16 = data16 | clip_data << 8;
        fos->write((const char *)&data16, 2);
      }
    }
  }

  auto filterSte = addConstantSymbolEntry(filter, ctx);

  // save Bias file
  const Tensor *tensorBias = NULL;

  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == bias) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensorBias = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensorBias);
  auto handleBias = tensorBias->getHandle<int32_t>();
  assert(handleBias.size() % 2 == 0);
  {
    auto fos = ctx->getWeightFileStream();
    for (size_t i = 0, e = handleBias.size(); i < e; i++) {
      auto data = handleBias.raw(i);
      fos->write((const char *)&data, 4);
    }
  }
  auto biasSte = addConstantSymbolEntry(bias, ctx);

  assert(biasScale == inScale * filterScale);

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

  bundle->append("  fullyconnected(");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }
  bundle->append("1.0/");
  bundle->append(std::to_string(inScale));
  bundle->append(", 0, ");

  if (filterSte.kind == '0') {
    bundle->append("(int8_t *)VTABufferGetVirtAddr(");
    bundle->append(filterSte.name);
    bundle->append(")");
  } else {
    bundle->append(filterSte.name);
  }

  bundle->append(", ");
  bundle->append("1.0/");
  bundle->append(std::to_string(filterScale));
  bundle->append(", 0, ");

  if (biasSte.kind == '0') {
    bundle->append("(int8_t *)VTABufferGetVirtAddr(");
    bundle->append(biasSte.name);
    bundle->append(")");
  } else {
    bundle->append(biasSte.name);
  }

  bundle->append(", ");
  bundle->append("1.0/");
  bundle->append(std::to_string(biasScale));
  bundle->append(", 0, ");

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }
  bundle->append("1.0/");
  bundle->append(std::to_string(outScale));
  bundle->append(", 0, ");

  bundle->append(std::to_string(srcDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[1]));
  bundle->append(", ");
  bundle->append(std::to_string(filterDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(filterDims[1]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[1]));
  bundle->append(", ");
  bundle->append(std::to_string(1)); // doBias
  bundle->append(" );\n");
}

void saveBNNFullyConnectedInst(const glow::FullyConnectedInst *Inst,
                               std::string *bundle, VTASaveContext *ctx) {
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto destDims = dest->dims();
  auto filter = Inst->getWeights();
  auto filterDims = filter->dims();
  auto bias = Inst->getBias();

  assert(srcDims.size() == 2);
  assert(bias->dims().size() == 1);
  assert(filterDims.size() == 2);
  assert(destDims.size() == 2);

  float inScale = src->getType()->getScale();
  float filterScale = filter->getType()->getScale();
  float outScale = Inst->getDest()->getType()->getScale();
  float biasScale = bias->getType()->getScale();

  filterScale = 1 / filterScale;
  inScale = 1 / inScale;
  biasScale = 1 / biasScale;
  outScale = 1 / outScale;

  // save Weight file
  auto vMap = ctx->getVariableMap();
  const Tensor *tensor = NULL;

  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == filter) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensor = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensor);
  auto handle = tensor->getHandle<int8_t>();
  assert(handle.size() % 2 == 0);
  {
    auto fos = ctx->getWeightFileStream();
    int16_t data16 = 0;
    for (size_t i = 0, e = handle.size(); i < e; i++) {
      auto data = handle.raw(i);
      int8_t sign_data;
      if (data > 0)
        sign_data = 1;
      if (data <= 0)
        sign_data = -1;
      if (i % 2 == 0) {
        data16 = 0xff & sign_data;
      } else {
        data16 = data16 | sign_data << 8;
        fos->write((const char *)&data16, 2);
      }
    }
  }

  auto filterSte = addConstantSymbolEntry(filter, ctx);

  // save Bias file
  const Tensor *tensorBias = NULL;

  for (auto it = vMap->begin(); it != vMap->end(); it++) {
    if (it->second == bias) {
      auto storage = it->first;
      if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
        const auto *constant = llvm::cast<Constant>(storage);
        tensorBias = &(constant->getPayload());
      }
      break;
    }
  }
  assert(tensorBias);
  auto handleBias = tensorBias->getHandle<int32_t>();
  assert(handleBias.size() % 2 == 0);
  {
    auto fos = ctx->getWeightFileStream();
    for (size_t i = 0, e = handleBias.size(); i < e; i++) {
      auto data = handleBias.raw(i);
      fos->write((const char *)&data, 4);
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

  bundle->append("  bnn_fullyconnected(");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }
  bundle->append("1.0/");
  bundle->append(std::to_string(inScale));
  bundle->append(", 0, ");

  if (filterSte.kind == '0') {
    bundle->append("(int8_t *)VTABufferGetVirtAddr(");
    bundle->append(filterSte.name);
    bundle->append(")");
  } else {
    bundle->append(filterSte.name);
  }

  bundle->append(", ");
  bundle->append("1.0/");
  bundle->append(std::to_string(filterScale));
  bundle->append(", 0, ");

  if (biasSte.kind == '0') {
    bundle->append("(int32_t *)VTABufferGetVirtAddr(");
    bundle->append(biasSte.name);
    bundle->append(")");
  } else {
    bundle->append(biasSte.name);
  }

  bundle->append(", ");
  bundle->append("1.0/");
  bundle->append(std::to_string(biasScale));
  bundle->append(", 0, ");

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }
  bundle->append("1.0/");
  bundle->append(std::to_string(outScale));
  bundle->append(", 0, ");

  bundle->append(std::to_string(srcDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[1]));
  bundle->append(", ");
  bundle->append(std::to_string(filterDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(filterDims[1]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[1]));
  bundle->append(", ");
  bundle->append(std::to_string(1)); // doBias
  bundle->append(" );\n");
}

void saveAvgPoolInst(const glow::AvgPoolInst *Inst, std::string *bundle,
                     VTASaveContext *ctx) {
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto destDims = dest->dims();

  float inScale = src->getType()->getScale();
  float outScale = Inst->getDest()->getType()->getScale();

  inScale = 1 / inScale;
  outScale = 1 / outScale;

  auto pad = Inst->getPads();
  assert(pad[0] == pad[1]);
  int pad_size = pad[0];

  auto strides = Inst->getStrides();
  assert(strides[0] == strides[1]);
  int stride_size = strides[0];

  auto kernels = Inst->getKernels();
  int KH = kernels[0];
  int KW = kernels[1];

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

  bundle->append("  avgpool(");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }
  bundle->append("1.0/");
  bundle->append(std::to_string(inScale));
  bundle->append(", 0, ");

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }
  bundle->append("1.0/");
  bundle->append(std::to_string(outScale));
  bundle->append(", 0, ");

  bundle->append(std::to_string(srcDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[1]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[2]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[3]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[1]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[2]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[3]));
  bundle->append(", ");
  bundle->append(std::to_string(KH));
  bundle->append(", ");
  bundle->append(std::to_string(KW));
  bundle->append(", ");
  bundle->append(std::to_string(pad_size));
  bundle->append(", ");
  bundle->append(std::to_string(stride_size));
  bundle->append(" );\n");
}

void saveTensorViewInst(const glow::TensorViewInst *Inst, std::string *bundle,
                        VTASaveContext *ctx) {
  auto src = Inst->getSrc();
  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  bundle->append("  int8_t* ");
  bundle->append(Inst->getName().str());
  bundle->append(" = ");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
  } else {
    bundle->append(src->getName());
  }

  // bundle->append(Inst->getSrc()->getName().str());
  bundle->append(";\n");
  return;
}
void saveMaxPoolInst(const glow::MaxPoolInst *Inst, std::string *bundle,
                     VTASaveContext *ctx) {
  assert(Inst->getSrc()->getType()->isQuantizedType());
  auto pad = Inst->getPads();
  assert(pad[0] == pad[1]);
  auto strides = Inst->getStrides();
  assert(strides[0] == strides[1]);

  auto src = Inst->getSrc();
  auto srcDims = src->dims();

  auto dest = Inst->getDest();
  auto kernels = Inst->getKernels();

  int N = srcDims[0];
  int H = srcDims[1];
  int W = srcDims[2];
  int C = srcDims[3];
  int KH = kernels[0];
  int KW = kernels[1];

  int pad_size = pad[0];
  int stride_size = strides[0];

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

  bundle->append("  maxpool(");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }

  bundle->append(std::to_string(N));
  bundle->append(", ");
  bundle->append(std::to_string(H));
  bundle->append(", ");
  bundle->append(std::to_string(W));
  bundle->append(", ");
  bundle->append(std::to_string(C));
  bundle->append(", ");
  bundle->append(std::to_string(KH));
  bundle->append(", ");
  bundle->append(std::to_string(KW));
  bundle->append(", ");
  bundle->append(std::to_string(pad_size));
  bundle->append(", ");
  bundle->append(std::to_string(stride_size));
  bundle->append(" );\n");
}

void saveElemSignInst(const glow::ElementSignInst *Inst, std::string *bundle,
                      VTASaveContext *ctx) {
  assert(Inst->getSrc()->getType()->isQuantizedType());
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto destDims = dest->dims();

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

  bundle->append("  elemsign(");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }

  unsigned_t size = 1;
  for (int i = 0; i < destDims.size(); i++) {
    size *= destDims[i];
  }

  bundle->append(std::to_string(size));
  bundle->append(");\n ");
}

void saveQuantizeInst(const glow::QuantizeInst *Inst, std::string *bundle,
                      VTASaveContext *ctx) {

  auto src = Inst->getSrc();

  auto dest = Inst->getDest();
  auto destTy = dest->getType();
  auto scale = destTy->getScale();
  auto offset = destTy->getOffset();
  auto revScale = 1 / scale;

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

#if defined(VTA_BNN)
  bundle->append("  typecast(");
#else()
  bundle->append("  quantize(");
#endif()

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }

  bundle->append(std::to_string(src->size()));
  bundle->append(", 1/");
  bundle->append(std::to_string(revScale));
  bundle->append(", ");
  bundle->append(std::to_string(offset));
  bundle->append(" );\n");
}

void saveDequantizeInst(const glow::DequantizeInst *Inst, std::string *bundle,
                        VTASaveContext *ctx) {

  auto src = Inst->getSrc();
  auto srcTy = src->getType();

  auto dest = Inst->getDest();
  auto scale = srcTy->getScale();
  auto offset = srcTy->getOffset();
  auto revScale = 1 / scale;

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

#if defined(VTA_BNN)
  bundle->append("  typecast(");
#else
  bundle->append("  dequantize(");
#endif

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }

  bundle->append(std::to_string(src->size()));
  bundle->append(", 1/");
  bundle->append(std::to_string(revScale));
  bundle->append(", ");
  bundle->append(std::to_string(offset));
  bundle->append(" );\n");
}

void saveSplatInst(const glow::SplatInst *Inst, std::string *bundle,
                   VTASaveContext *ctx) {
  auto dest = Inst->getDest();
  auto destDims = dest->dims();

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

  bundle->append("  splat(");

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }

  unsigned_t size = 1;
  for (int i = 0; i < destDims.size(); i++) {
    size *= destDims[i];
  }

  bundle->append(std::to_string(size));
  bundle->append(", ");

  auto destTy = Inst->getDest()->getType();
  TensorQuantizationParams destQ{destTy->getScale(), destTy->getOffset()};
  float val = Inst->getValue();

  bundle->append(std::to_string((quantization::quantize(val, destQ))));
  bundle->append(" );\n");
}

void saveTransposeInst(const glow::TransposeInst *Inst, std::string *bundle,
                       VTASaveContext *ctx) {
  auto src = Inst->getSrc();
  auto srcDims = src->dims();

  auto dest = Inst->getDest();
  auto destDims = dest->dims();

  auto shuffle = Inst->getShuffle();

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

  bundle->append("  transpose(");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }

  bundle->append(std::to_string(srcDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[1]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[2]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[3]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[1]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[2]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[3]));
  bundle->append(", ");
  bundle->append(std::to_string(shuffle[0]));
  bundle->append(", ");
  bundle->append(std::to_string(shuffle[1]));
  bundle->append(", ");
  bundle->append(std::to_string(shuffle[2]));
  bundle->append(", ");
  bundle->append(std::to_string(shuffle[3]));
  bundle->append(" );\n");
}

void saveTranspose_6dim(const glow::TransposeInst *Inst, std::string *bundle,
                        VTASaveContext *ctx) {
  // TODO : consider group
  // auto group = Inst->getGroup();
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  auto dest = Inst->getDest();
  auto destDims = dest->dims();

  auto shuffle = Inst->getShuffle();

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

  bundle->append("  transpose_6dim(");
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }

  bundle->append(std::to_string(srcDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[1]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[2]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[3]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[4]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[5]));
  bundle->append(", ");

  bundle->append(std::to_string(destDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[1]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[2]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[3]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[4]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[5]));
  bundle->append(", ");

  bundle->append(std::to_string(shuffle[0]));
  bundle->append(", ");
  bundle->append(std::to_string(shuffle[1]));
  bundle->append(", ");
  bundle->append(std::to_string(shuffle[2]));
  bundle->append(", ");
  bundle->append(std::to_string(shuffle[3]));
  bundle->append(", ");
  bundle->append(std::to_string(shuffle[4]));
  bundle->append(", ");
  bundle->append(std::to_string(shuffle[5]));
  bundle->append(" );\n");
}

void saveElemAddInst(const glow::ElementAddInst *Inst, std::string *bundle,
                     VTASaveContext *ctx) {
  auto src0 = Inst->getLHS();
  auto src1 = Inst->getRHS();

  auto dest = Inst->getDest();
  auto destDims = dest->dims();

  auto srcWeight0 = static_cast<WeightVar *>(src0);
  if (srcWeight0->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight0, bundle, ctx);
  }

  auto srcWeight1 = static_cast<WeightVar *>(src1);
  if (srcWeight1->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight1, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

  bundle->append("  elemadd(");

  if (srcWeight0->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight0, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src0->getName());
    bundle->append(", ");
  }

  bundle->append("1.0/");
  bundle->append(std::to_string(1 / src0->getType()->getScale()));
  bundle->append(", 0, ");

  if (srcWeight1->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight1, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src1->getName());
    bundle->append(", ");
  }

  bundle->append("1.0/");
  bundle->append(std::to_string(1 / src1->getType()->getScale()));
  bundle->append(", 0, ");

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }

  bundle->append("1.0/");
  bundle->append(std::to_string(1 / dest->getType()->getScale()));
  bundle->append(", 0, ");

  unsigned_t size = 1;
  for (int i = 0; i < destDims.size(); i++) {
    size *= destDims[i];
  }

  bundle->append(std::to_string(size));
  bundle->append(" );\n");
}

void saveElemSubInst(const glow::ElementSubInst *Inst, std::string *bundle,
                     VTASaveContext *ctx) {

  if (Inst->getLHS()->getType()->isQuantizedType()) {
    auto src0 = Inst->getLHS();
    auto src1 = Inst->getRHS();
    auto dest = Inst->getDest();
    auto destDims = dest->dims();

    auto srcWeight0 = static_cast<WeightVar *>(src0);
    if (srcWeight0->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      addSymbolEntryGenBundle(srcWeight0, bundle, ctx);
    }

    auto srcWeight1 = static_cast<WeightVar *>(src1);
    if (srcWeight1->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      addSymbolEntryGenBundle(srcWeight1, bundle, ctx);
    }
    // TODO : yongin apply this code to other saves
    else {
      addConstantSymbolEntry(srcWeight1, ctx);
      {
        auto vMap = ctx->getVariableMap();
        const Tensor *tensor = NULL;
        for (auto it = vMap->begin(); it != vMap->end(); it++) {
          if (it->second == src1) {
            auto storage = it->first;
            if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
              const auto *constant = llvm::cast<Constant>(storage);
              tensor = &(constant->getPayload());
            }
            break;
          }
        }
        assert(tensor);
        auto handle = tensor->getHandle<int8_t>();
        auto fos = ctx->getWeightFileStream();
        int16_t data16 = 0;
        for (size_t i = 0, e = handle.size(); i < e; i++) {
          auto data = handle.raw(i);
          if (data > 127)
            data = 127.0;
          if (data < -128)
            data = -128.0;
          int8_t clip_data = std::floor(data);
          if (i % 2 == 0) {
            data16 = 0xff & clip_data;
          } else {
            data16 = data16 | clip_data << 8;
            fos->write((const char *)&data16, 2);
          }
        }
      }
    }

    auto destWeight = static_cast<WeightVar *>(dest);
    if (destWeight->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      addSymbolEntryGenBundle(destWeight, bundle, ctx);
    }

    bundle->append("  elemsub(");

    if (srcWeight0->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      auto ste = addSymbolEntry(srcWeight0, ctx);
      bundle->append(ste.name);
      bundle->append(", ");
    } else {
      auto ste = addConstantSymbolEntry(srcWeight0, ctx);
      if (ste.kind == '0') {
        bundle->append("(int8_t *)VTABufferGetVirtAddr(");
        bundle->append(src0->getName());
        bundle->append(")");
        bundle->append(", ");
      } else {
        bundle->append(src0->getName());
        bundle->append(", ");
      }
    }

    bundle->append("1.0/");
    bundle->append(std::to_string(1 / src0->getType()->getScale()));
    bundle->append(", 0, ");

    if (srcWeight1->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      auto ste = addSymbolEntry(srcWeight1, ctx);
      bundle->append(ste.name);
      bundle->append(", ");
    } else {
      auto ste = addConstantSymbolEntry(srcWeight1, ctx);
      if (ste.kind == '0') {
        bundle->append("(int8_t *)VTABufferGetVirtAddr(");
        bundle->append(src1->getName());
        bundle->append(")");
        bundle->append(", ");
      } else {
        bundle->append(src1->getName());
        bundle->append(", ");
      }
    }

    bundle->append("1.0/");
    bundle->append(std::to_string(1 / src1->getType()->getScale()));
    bundle->append(", 0, ");

    if (destWeight->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      auto ste = addSymbolEntry(destWeight, ctx);
      bundle->append(ste.name);
      bundle->append(", ");
    } else {
      bundle->append(dest->getName());
      bundle->append(", ");
    }

    bundle->append("1.0/");
    bundle->append(std::to_string(1 / dest->getType()->getScale()));
    bundle->append(", 0, ");

    unsigned_t size = 1;
    for (int i = 0; i < destDims.size(); i++) {
      size *= destDims[i];
    }

    bundle->append(std::to_string(size));
    bundle->append(" );\n");
  } else {
    auto src0 = Inst->getLHS();
    auto src1 = Inst->getRHS();
    auto dest = Inst->getDest();
    auto destDims = dest->dims();

    auto srcWeight0 = static_cast<WeightVar *>(src0);
    if (srcWeight0->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      addSymbolEntryGenBundle(srcWeight0, bundle, ctx);
    }

    auto srcWeight1 = static_cast<WeightVar *>(src1);
    if (srcWeight1->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      addSymbolEntryGenBundle(srcWeight1, bundle, ctx);
    }
    // TODO : yongin apply this code to other saves
    else {
      addConstantSymbolEntry(srcWeight1, ctx);
      {
        auto vMap = ctx->getVariableMap();
        const Tensor *tensor = NULL;
        for (auto it = vMap->begin(); it != vMap->end(); it++) {
          if (it->second == src1) {
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
        auto fos = ctx->getWeightFileStream();
        for (size_t i = 0, e = handle.size(); i < e; i++) {
          auto data = handle.raw(i);
          fos->write((const char *)&data, 4);
        }
      }
    }

    auto destWeight = static_cast<WeightVar *>(dest);
    if (destWeight->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      addSymbolEntryGenBundle(destWeight, bundle, ctx);
    }

    bundle->append("  elemsub_f32(");

    if (srcWeight0->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      auto ste = addSymbolEntry(srcWeight0, ctx);
      bundle->append(ste.name);
      bundle->append(", ");
    } else {
      auto ste = addConstantSymbolEntry(srcWeight0, ctx);
      if (ste.kind == '0') {
        bundle->append("(int8_t *)VTABufferGetVirtAddr(");
        bundle->append(src0->getName());
        bundle->append(")");
        bundle->append(", ");
      } else {
        bundle->append(src0->getName());
        bundle->append(", ");
      }
    }

    bundle->append("1.0/");
    bundle->append(std::to_string(1 / 1));
    bundle->append(", 0, ");

    if (srcWeight1->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      auto ste = addSymbolEntry(srcWeight1, ctx);
      bundle->append(ste.name);
      bundle->append(", ");
    } else {
      auto ste = addConstantSymbolEntry(srcWeight1, ctx);
      if (ste.kind == '0') {
        bundle->append("(int8_t *)VTABufferGetVirtAddr(");
        bundle->append(src1->getName());
        bundle->append(")");
        bundle->append(", ");
      } else {
        bundle->append(src1->getName());
        bundle->append(", ");
      }
    }

    bundle->append("1.0/");
    bundle->append(std::to_string(1 / 1));
    bundle->append(", 0, ");

    if (destWeight->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      auto ste = addSymbolEntry(destWeight, ctx);
      bundle->append(ste.name);
      bundle->append(", ");
    } else {
      bundle->append(dest->getName());
      bundle->append(", ");
    }

    bundle->append("1.0/");
    bundle->append(std::to_string(1 / 1));
    bundle->append(", 0, ");

    unsigned_t size = 1;
    for (int i = 0; i < destDims.size(); i++) {
      size *= destDims[i];
    }

    bundle->append(std::to_string(size));
    bundle->append(" );\n");
  }
}

void saveElemDivInst(const glow::ElementDivInst *Inst, std::string *bundle,
                     VTASaveContext *ctx) {
  if (Inst->getLHS()->getType()->isQuantizedType()) {
    auto src0 = Inst->getLHS();
    auto src1 = Inst->getRHS();

    auto dest = Inst->getDest();
    auto destDims = dest->dims();

    auto srcWeight0 = static_cast<WeightVar *>(src0);
    if (srcWeight0->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      addSymbolEntryGenBundle(srcWeight0, bundle, ctx);
    }

    auto srcWeight1 = static_cast<WeightVar *>(src1);
    if (srcWeight1->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      addSymbolEntryGenBundle(srcWeight1, bundle, ctx);
    } else {
      addConstantSymbolEntry(srcWeight1, ctx);
      {
        auto vMap = ctx->getVariableMap();
        const Tensor *tensor = NULL;
        for (auto it = vMap->begin(); it != vMap->end(); it++) {
          if (it->second == src1) {
            auto storage = it->first;
            if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
              const auto *constant = llvm::cast<Constant>(storage);
              tensor = &(constant->getPayload());
            }
            break;
          }
        }
        assert(tensor);
        auto handle = tensor->getHandle<int8_t>();
        auto fos = ctx->getWeightFileStream();
        int16_t data16 = 0;
        for (size_t i = 0, e = handle.size(); i < e; i++) {
          auto data = handle.raw(i);
          if (data > 127)
            data = 127.0;
          if (data < -128)
            data = -128.0;
          int8_t clip_data = std::floor(data);
          if (i % 2 == 0) {
            data16 = 0xff & clip_data;
          } else {
            data16 = data16 | clip_data << 8;
            fos->write((const char *)&data16, 2);
          }
        }
      }
    }

    auto destWeight = static_cast<WeightVar *>(dest);
    if (destWeight->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      addSymbolEntryGenBundle(destWeight, bundle, ctx);
    }

    bundle->append("  elemdiv(");

    if (srcWeight0->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      auto ste = addSymbolEntry(srcWeight0, ctx);
      bundle->append(ste.name);
      bundle->append(", ");
    } else {
      auto ste = addConstantSymbolEntry(srcWeight0, ctx);
      if (ste.kind == '0') {
        bundle->append("(int8_t *)VTABufferGetVirtAddr(");
        bundle->append(src0->getName());
        bundle->append(")");
        bundle->append(", ");
      } else {
        bundle->append(src0->getName());
        bundle->append(", ");
      }
    }

    bundle->append("1.0/");
    bundle->append(std::to_string(1 / src0->getType()->getScale()));
    bundle->append(", 0, ");

    if (srcWeight1->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      auto ste = addSymbolEntry(srcWeight1, ctx);
      bundle->append(ste.name);
      bundle->append(", ");
    } else {
      auto ste = addConstantSymbolEntry(srcWeight1, ctx);
      if (ste.kind == '0') {
        bundle->append("(int8_t *)VTABufferGetVirtAddr(");
        bundle->append(src1->getName());
        bundle->append(")");
        bundle->append(", ");
      } else {
        bundle->append(src1->getName());
        bundle->append(", ");
      }
    }

    bundle->append("1.0/");
    bundle->append(std::to_string(1 / src1->getType()->getScale()));
    bundle->append(", 0, ");

    if (destWeight->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      auto ste = addSymbolEntry(destWeight, ctx);
      bundle->append(ste.name);
      bundle->append(", ");
    } else {
      bundle->append(dest->getName());
      bundle->append(", ");
    }

    bundle->append("1.0/");
    bundle->append(std::to_string(1 / dest->getType()->getScale()));
    bundle->append(", 0, ");

    unsigned_t size = 1;
    for (int i = 0; i < destDims.size(); i++) {
      size *= destDims[i];
    }

    bundle->append(std::to_string(size));
    bundle->append(" );\n");
  } else {
    auto src0 = Inst->getLHS();
    auto src1 = Inst->getRHS();

    auto dest = Inst->getDest();
    auto destDims = dest->dims();

    auto srcWeight0 = static_cast<WeightVar *>(src0);
    if (srcWeight0->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      addSymbolEntryGenBundle(srcWeight0, bundle, ctx);
    }

    auto srcWeight1 = static_cast<WeightVar *>(src1);
    if (srcWeight1->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      addSymbolEntryGenBundle(srcWeight1, bundle, ctx);
    } else {
      addConstantSymbolEntry(srcWeight1, ctx);
      {
        auto vMap = ctx->getVariableMap();
        const Tensor *tensor = NULL;
        for (auto it = vMap->begin(); it != vMap->end(); it++) {
          if (it->second == src1) {
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
        auto fos = ctx->getWeightFileStream();
        for (size_t i = 0, e = handle.size(); i < e; i++) {
          auto data = handle.raw(i);
          fos->write((const char *)&data, 4);
        }
      }
    }

    auto destWeight = static_cast<WeightVar *>(dest);
    if (destWeight->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      addSymbolEntryGenBundle(destWeight, bundle, ctx);
    }

    bundle->append("  elemdiv_f32(");

    if (srcWeight0->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      auto ste = addSymbolEntry(srcWeight0, ctx);
      bundle->append(ste.name);
      bundle->append(", ");
    } else {
      auto ste = addConstantSymbolEntry(srcWeight0, ctx);
      if (ste.kind == '0') {
        bundle->append("(int8_t *)VTABufferGetVirtAddr(");
        bundle->append(src0->getName());
        bundle->append(")");
        bundle->append(", ");
      } else {
        bundle->append(src0->getName());
        bundle->append(", ");
      }
    }

    bundle->append("1.0/");
    bundle->append(std::to_string(1 / 1));
    bundle->append(", 0, ");

    if (srcWeight1->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      auto ste = addSymbolEntry(srcWeight1, ctx);
      bundle->append(ste.name);
      bundle->append(", ");
    } else {
      auto ste = addConstantSymbolEntry(srcWeight1, ctx);
      if (ste.kind == '0') {
        bundle->append("(int8_t *)VTABufferGetVirtAddr(");
        bundle->append(src1->getName());
        bundle->append(")");
        bundle->append(", ");
      } else {
        bundle->append(src1->getName());
        bundle->append(", ");
      }
    }

    bundle->append("1.0/");
    bundle->append(std::to_string(1 / 1));
    bundle->append(", 0, ");

    if (destWeight->getMutability() ==
        glow::WeightVar::MutabilityKind::Mutable) {
      auto ste = addSymbolEntry(destWeight, ctx);
      bundle->append(ste.name);
      bundle->append(", ");
    } else {
      bundle->append(dest->getName());
      bundle->append(", ");
    }

    bundle->append("1.0/");
    bundle->append(std::to_string(1 / 1));
    bundle->append(", 0, ");

    unsigned_t size = 1;
    for (int i = 0; i < destDims.size(); i++) {
      size *= destDims[i];
    }

    bundle->append(std::to_string(size));
    bundle->append(" );\n");
  }
}

void saveElemMaxInst(const glow::ElementMaxInst *Inst, std::string *bundle,
                     VTASaveContext *ctx) {
  auto src0 = Inst->getLHS();
  auto src1 = Inst->getRHS();

  auto dest = Inst->getDest();
  auto destDims = dest->dims();

  auto srcWeight0 = static_cast<WeightVar *>(src0);
  if (srcWeight0->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight0, bundle, ctx);
  }

  auto srcWeight1 = static_cast<WeightVar *>(src1);
  if (srcWeight1->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight1, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

  bundle->append("  elemmax(");

  if (srcWeight0->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight0, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src0->getName());
    bundle->append(", ");
  }

  bundle->append("1.0/");
  bundle->append(std::to_string(1 / src0->getType()->getScale()));
  bundle->append(", 0, ");

  if (srcWeight1->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight1, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(src1->getName());
    bundle->append(", ");
  }

  bundle->append("1.0/");
  bundle->append(std::to_string(1 / src1->getType()->getScale()));
  bundle->append(", 0, ");

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }

  bundle->append("1.0/");
  bundle->append(std::to_string(1 / dest->getType()->getScale()));
  bundle->append(", 0, ");

  unsigned_t size = 1;
  for (int i = 0; i < destDims.size(); i++) {
    size *= destDims[i];
  }

  bundle->append(std::to_string(size));
  bundle->append(" );\n");
}

void saveDebugPrintInst(const glow::DebugPrintInst *Inst, std::string *bundle,
                        VTASaveContext *ctx) {
  auto *src = Inst->getSrc();
  auto srcDims = src->dims();
  auto elemType = src->getElementType();
  auto srcWeight = static_cast<WeightVar *>(src);

  std::string format = Inst->getFormat();
  std::string filename = Inst->getFileName();
  auto pos = filename.find_last_of("/");
  filename.erase(0, pos);
  filename = "./debug" + filename;
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Constant) {
    addConstantSymbolEntry(src, ctx);
  }
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }
  bundle->append("  debugprint(");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    auto ste = addConstantSymbolEntry(srcWeight, ctx);
    if (ste.kind == '0') {
      bundle->append("(int8_t *)VTABufferGetVirtAddr(");
      bundle->append(src->getName());
      bundle->append(")");
      bundle->append(", ");
    } else {
      bundle->append(src->getName());
      bundle->append(", ");
    }
  }

  unsigned_t size = 1;
  for (int i = 0; i < srcDims.size(); i++) {
    size *= srcDims[i];
  }

  bundle->append(std::to_string(size));
  bundle->append(", \"");
  bundle->append(filename);
  bundle->append("\", ");
  if (elemType == ElemKind::Int8QTy) {
    bundle->append("1);\n");
  } else if (elemType == ElemKind::FloatTy) {
    bundle->append("0);\n");
  } else if (elemType == ElemKind::Int32QTy) {
    bundle->append("0);\n");
  } else {
    std::string msg = Inst->getKindName();
    msg.append(" is an unhandled element type.");
    llvm_unreachable(msg.c_str());
  }
}
void saveSoftMaxInst(const glow::SoftMaxInst *Inst, std::string *bundle,
                     VTASaveContext *ctx) {
  auto src = Inst->getSrc();
  auto srcDims = src->dims();

  auto dest = Inst->getDest();
  auto destDims = dest->dims();

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

  bundle->append("  softmax(");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");

  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }
  bundle->append(std::to_string(srcDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(srcDims[1]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[0]));
  bundle->append(", ");
  bundle->append(std::to_string(destDims[1]));
  bundle->append(" );\n");
}

void saveReluInst(const glow::ReluInst *Inst, std::string *bundle,
                  VTASaveContext *ctx) {
  auto src = Inst->getSrc();
  auto srcDims = src->dims();
  float inScale = src->getType()->getScale();

  auto dest = Inst->getDest();
  float outScale = Inst->getDest()->getType()->getScale();

  auto srcWeight = static_cast<WeightVar *>(src);
  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(srcWeight, bundle, ctx);
  }

  auto destWeight = static_cast<WeightVar *>(dest);
  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    addSymbolEntryGenBundle(destWeight, bundle, ctx);
  }

  bundle->append("  relu(");

  if (srcWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(srcWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");

  } else {
    bundle->append(src->getName());
    bundle->append(", ");
  }

  if (destWeight->getMutability() == glow::WeightVar::MutabilityKind::Mutable) {
    auto ste = addSymbolEntry(destWeight, ctx);
    bundle->append(ste.name);
    bundle->append(", ");
  } else {
    bundle->append(dest->getName());
    bundle->append(", ");
  }

  int src_size = 1;

  for (int i = 0; i < srcDims.size(); i++) {
    src_size *= srcDims[i];
  }
  bundle->append(std::to_string(src_size));
  if (inScale != outScale) {
    bundle->append(", ");
    bundle->append(std::to_string(inScale));
    bundle->append(", ");
    bundle->append(std::to_string(outScale));
  }

  bundle->append(" );\n");
}

struct BundleSaveCtx {
  std::string bundle;
  std::string initConst;
  std::string destroyConst;
  std::string includeHeader;
  std::string symbolTable;
  std::string bundleConfig;
};

void initBundleSave(struct BundleSaveCtx &bundleCtx, llvm::StringRef bundleName,
                    llvm::StringRef mainEntryName) {
  auto &bundle = bundleCtx.bundle;
  // bundleCtx.bundle = new std::string();
  bundle = "";
  bundle.append("int ");
  bundle.append(mainEntryName);
#ifdef NESTC_EVTA_PROFILE_AUTOTUNE
  bundle.append("(uint8_t *constantWeight, uint8_t *mutableWeight, uint8_t "
                "*activations, int input1, int input2, int input3){\n");
#else
  bundle.append("(uint8_t *constantWeight, uint8_t *mutableWeight, uint8_t "
                "*activations){\n");
#endif
#ifdef VTA_ALLINONE_FUNCTION
  bundle.append("  ");
  bundle.append(bundleName);
  bundle.append("_load_module(constantWeight);\n");
#endif
#ifdef VTA_RESET_CMDH
  bundle.append("  vtaCmdH = VTATLSCommandHandle();\n");
#endif

#ifdef VTA_PROFILE
  bundle.append("  std::ofstream prof_out(\"profileResult.txt\");\n");
#endif

  bundleCtx.initConst = "";
  bundleCtx.destroyConst = "";

  // Generate Include Header
  bundleCtx.includeHeader = "#include \"";
  bundleCtx.includeHeader.append(mainEntryName);
  bundleCtx.includeHeader.append(".h\"\n");
}

void finalizeBundleEntrySave(struct BundleSaveCtx &bundleCtx,
                             llvm::StringRef bundleName,
                             llvm::StringRef mainEntryName,
                             VTASaveContext &ctx) {
  auto &bundle = bundleCtx.bundle;
#ifdef VTA_ALLINONE_FUNCTION
  bundle.append("  ");
  bundle.append(bundleName);
  bundle.append("_destroy_module();\n");
#endif
#ifdef VTA_RESET_CMDH
  bundle.append("  VTARuntimeShutdown();\n");
#endif

  bundle.append("  return 0;\n");
  bundle.append("}");

  // Generate SymbolTable struct
  auto &symbolTable = bundleCtx.symbolTable;
  symbolTable = "SymbolTableEntry symbolTableEntry_";
  symbolTable.append(bundleName);
  symbolTable.append("[");
  symbolTable.append(std::to_string(ctx.getSymbols()->size()));
  symbolTable.append("]={");
  for (auto sym : *(ctx.getSymbols())) {
    symbolTable.append("{\"");
    symbolTable.append(sym.name);
    symbolTable.append("\",");
    symbolTable.append(std::to_string(sym.offset));
    symbolTable.append(",");
    symbolTable.append(std::to_string(sym.size));
    symbolTable.append(",'1'},");
  }
  symbolTable.erase(symbolTable.size() - 1, 1);
  symbolTable.append("};\n");

  // Generate BundleConfig struct
  auto &bundleConfig = bundleCtx.bundleConfig;
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

  auto &initConst = bundleCtx.initConst;
  initConst.append("namespace namespace_");
  initConst.append(bundleName);
  initConst.append(" {\n");
  // Generate Init Constant values
  for (auto ste : *ctx.getConstantSymbols()) {
    if (ste.kind == '0') {
      initConst.append("int");
      initConst.append(std::to_string(8));
      initConst.append("_t* ");
      initConst.append(ste.name);
      initConst.append(";\n");
    }
#ifdef VTA_MEMOPT_DISABLE
    else if (ste.kind == '1') {
      initConst.append("int");
      initConst.append(std::to_string(8));
      initConst.append("_t* ");
      initConst.append(ste.name);
      initConst.append(";\n");
    }
#endif
  }
  initConst.append("}\n");
  initConst.append("using namespace namespace_");
  initConst.append(bundleName);
  initConst.append(";\n");
  initConst.append("extern VTACommandHandle vtaCmdH");
  if (ctx.getIdxMultiEVTA()) {
    initConst.append(std::to_string(ctx.getIdxMultiEVTA()));
  }
  initConst.append(";\n");

  initConst.append("\nvoid ");
  initConst.append(mainEntryName);
  initConst.append("_load_module(uint8_t *constantWeight){\n");
  // initConst.append("  xlnk_reset();\n");
  // initConst.append("  vtaCmdH = VTATLSCommandHandle();\n");

  for (auto ste : *ctx.getConstantSymbols()) {
    if (ste.kind == '0') {
      initConst.append("  ");
      initConst.append(ste.name);
      initConst.append(" = (int8_t *)VTABufferAlloc(");
      initConst.append(std::to_string(ste.size));
      initConst.append(");\n");
      initConst.append("  VTABufferCopy((int8_t *)(constantWeight + ");
      initConst.append(std::to_string(ste.offset));
      initConst.append("), 0, ");
      initConst.append(ste.name);
      initConst.append(", 0, ");
      initConst.append(std::to_string(ste.size));
      initConst.append(", 1);\n");
    }
#ifdef VTA_MEMOPT_DISABLE
    else if (ste.kind == '1') {
      initConst.append("  ");
      initConst.append(ste.name);
      initConst.append(" = (int8_t *)malloc(");
      initConst.append(std::to_string(ste.size));
      initConst.append(");\n");
    }
#endif
  }
  initConst.append("}\n");

  auto &destroyConst = bundleCtx.destroyConst;
  destroyConst.append("\nvoid ");
  destroyConst.append(mainEntryName);
  destroyConst.append("_destroy_module(){\n");
  for (auto ste : *ctx.getConstantSymbols()) {
    if (ste.kind == '0') {
      destroyConst.append("  ");
      destroyConst.append("VTABufferFree(");
      destroyConst.append(ste.name);
      destroyConst.append(");\n");
    }
#ifdef VTA_MEMOPT_DISABLE
    else if (ste.kind == '1') {
      destroyConst.append("  ");
      destroyConst.append("free(");
      destroyConst.append(ste.name);
      destroyConst.append(");\n");
    }
#endif
  }
  destroyConst.append("}\n");
}

void exportBundleEntry(llvm::StringRef outputDir, llvm::StringRef bundleName,
                       struct BundleSaveCtx &bundleCtx) {
  std::string outputFile = outputDir;
  if (outputFile[outputFile.size() - 1] != '/')
    outputFile.append("/");
  outputFile.append(bundleName);
  outputFile.append(".cpp");

  std::ofstream fos(outputFile.c_str(), std::ios::out);
  fos.write(VTA_SAVE_COMMON.c_str(), VTA_SAVE_COMMON.size());
  fos.write(bundleCtx.includeHeader.c_str(), bundleCtx.includeHeader.size());
  auto &symbolTable = bundleCtx.symbolTable;
  fos.write(symbolTable.c_str(), symbolTable.size());
  auto &bundleConfig = bundleCtx.bundleConfig;
  fos.write(bundleConfig.c_str(), bundleConfig.size());
  fos.write(bundleCtx.initConst.c_str(), bundleCtx.initConst.size());
  fos.write(bundleCtx.destroyConst.c_str(), bundleCtx.destroyConst.size());
  fos.write(bundleCtx.bundle.c_str(), bundleCtx.bundle.size());
  fos.close();
}

void exportVTARuntimeHeader(llvm::StringRef outputDir) {
  std::string vtaRuntimeHeaderFile = outputDir;
  vtaRuntimeHeaderFile.append("/");
  vtaRuntimeHeaderFile.append("VTARuntime");
  vtaRuntimeHeaderFile.append(".h");

  std::ofstream ros(vtaRuntimeHeaderFile.c_str(), std::ios::out);
  ros.write(VTA_RUNTIME_HEADER.c_str(), VTA_RUNTIME_HEADER.size());
  ros.close();
}

void exportBundleHeader(llvm::StringRef outputDir, llvm::StringRef bundleName,
                        llvm::StringRef mainEntryName, VTASaveContext &ctx) {
  std::string vtaBundleHeaderFile = outputDir;
  vtaBundleHeaderFile.append("/");
  vtaBundleHeaderFile.append(mainEntryName);
  vtaBundleHeaderFile.append(".h");

  std::ofstream bhos(vtaBundleHeaderFile.c_str(), std::ios::out);
  std::string bundleHeaderDefineName = "VTA_BUNDLE_";
  bundleHeaderDefineName.append(bundleName);
  std::transform(bundleHeaderDefineName.begin(), bundleHeaderDefineName.end(),
                 bundleHeaderDefineName.begin(), ::toupper);

  std::string bundleHeaderDefine = "#ifndef ";
  bundleHeaderDefine.append(bundleHeaderDefineName);
  bundleHeaderDefine.append("\n#define ");
  bundleHeaderDefine.append(bundleHeaderDefineName);
  bundleHeaderDefine.append("\n");

  bhos.write(bundleHeaderDefine.c_str(), bundleHeaderDefine.size());
  bhos.write(VTA_BUNDLE_HEADER_0.c_str(), VTA_BUNDLE_HEADER_0.size());
  std::string bundleHeader = "";
  bundleHeader.append("extern BundleConfig ");
  bundleHeader.append(bundleName);
  bundleHeader.append("_config;\n");
  bundleHeader.append("void ");
  bundleHeader.append(mainEntryName);
  bundleHeader.append("_load_module(uint8_t *constantWeight);\n");
  bundleHeader.append("void ");
  bundleHeader.append(mainEntryName);
  bundleHeader.append("_destroy_module();\n");
  bundleHeader.append("int ");
  bundleHeader.append(mainEntryName);
#ifdef NESTC_EVTA_PROFILE_AUTOTUNE
  bundleHeader.append(
      "(uint8_t *constantWeight, uint8_t *mutableWeight, uint8_t *activations, "
      "int input1, int input2, int input3);\n");
#else
  bundleHeader.append("(uint8_t *constantWeight, uint8_t *mutableWeight, "
                      "uint8_t *activations);\n");
#endif
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

  for (auto ste : *ctx.getSymbols()) {
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
  bhos.write(VTA_BUNDLE_HEADER_1.c_str(), VTA_BUNDLE_HEADER_1.size());

  std::string bundleHeaderEndif = "#endif //";
  bundleHeaderEndif.append(bundleHeaderDefineName);
  bundleHeaderEndif.append("\n");
  bhos.write(bundleHeaderEndif.c_str(), bundleHeaderEndif.size());

  bhos.close();
}

}
using namespace vta_save;

void VTA::save(Function *F, llvm::StringRef outputDir,
               llvm::StringRef bundleName,
               llvm::StringRef mainEntryName) const {
  if (((idxMultiEVTA & 1) + ((idxMultiEVTA >> 1) & 1) +
       ((idxMultiEVTA >> 2) & 1) + ((idxMultiEVTA >> 3) & 1)) != 1) {
    llvm::errs() << "Not supported Multi-EVTA combination\n"
                    "Please use one EVTA at one time\n";
    std::exit(1);
  }

  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());
  VTASaveContext ctx(&IR->getVariableMap());

  for (int i = 0; i < 4; i++) {
    if (((idxMultiEVTA >> i) & 1) == 1) {
      ctx.setIdxMultiEVTA(i);
      break;
    }
  }

  std::string weightFileName = outputDir;
  weightFileName.append("/");
  weightFileName.append(bundleName);
  weightFileName.append(".weights.bin");
  std::ofstream wfos(weightFileName.c_str(), std::ios::binary);
  ctx.setWeightFileStream(&wfos);

  struct BundleSaveCtx bundleCtx;
  auto &bundle = bundleCtx.bundle;
  auto &initConst = bundleCtx.initConst;
  auto &destroyConst = bundleCtx.destroyConst;

  initBundleSave(bundleCtx, bundleName, mainEntryName);

  for (const auto &I : IR->getInstrs()) {

#ifdef VTA_PROFILE
    if (I.getKind() != Kinded::Kind::DebugPrintInstKind) {
      bundle.append("  clock_t ");
      bundle.append(I.getName());
      bundle.append("_start = clock();\n");
    }
#endif

    bundle.append("\n  //Run ");
    bundle.append(I.getKindName());
    bundle.append(" : ");
    bundle.append(I.getName());
    bundle.append("\n");

    switch (I.getKind()) {
    case Kinded::Kind::ConvolutionInstKind: {
      auto I2 = llvm::cast<ConvolutionInst>(&I);
#if defined(VTA_BNN)
      saveBNNConvolutionInst(I2, &bundle, &initConst, &ctx);
#else
      saveConvolutionInst(I2, &bundle, &initConst, &ctx);
#endif
      break;
    }
    case Kinded::Kind::VTAConvolutionInstKind: {
      auto I2 = llvm::cast<VTAConvolutionInst>(&I);
      saveEVTAConvolutionInst(I2, &bundle, &initConst, &ctx);
      break;
    }
    case Kinded::Kind::AllocActivationInstKind: {
      auto I2 = llvm::cast<AllocActivationInst>(&I);
#ifndef VTA_MEMOPT_DISABLE
      bundle.append("  int8_t *");
      bundle.append(I2->getName());
      bundle.append(" = (int8_t *)malloc(");
      bundle.append(std::to_string(I2->getSizeInBytes()));
      bundle.append(");\n");
#endif
      // for not constant but represented as a constant value
      auto csyms = ctx.getConstantSymbols();
      csyms->push_back({I2->getName().data(), 0, I2->getSizeInBytes(), '1'});

      break;
    }
    case Kinded::Kind::DeallocActivationInstKind: {
#ifndef VTA_MEMOPT_DISABLE
      auto I2 = llvm::cast<DeallocActivationInst>(&I);
      bundle.append("  free(");
      bundle.append(I2->getSrc()->getName());
      bundle.append(");\n");
#endif
      break;
    }
    case Kinded::Kind::MaxPoolInstKind: {
      auto I2 = llvm::cast<MaxPoolInst>(&I);
      saveMaxPoolInst(I2, &bundle, &ctx);
      break;
    }
    case Kinded::Kind::QuantizeInstKind: {
      auto I2 = llvm::cast<QuantizeInst>(&I);
      saveQuantizeInst(I2, &bundle, &ctx);
      break;
    }
    case Kinded::Kind::TransposeInstKind: {
      auto I2 = llvm::cast<TransposeInst>(&I);
      auto srcDimSize = I2->getSrc()->dims().size();

      if (srcDimSize == 4) {
        saveTransposeInst(I2, &bundle, &ctx);
      } else if (srcDimSize == 6) {
        saveTranspose_6dim(I2, &bundle, &ctx);
      } else {
        llvm::errs() << "Not supported dimenstion in transpose\n"
                        "Please use 4 dim or 6 dim\n";
        std::exit(1);
      }

      break;
    }
    case Kinded::Kind::SplatInstKind: {
      auto I2 = llvm::cast<SplatInst>(&I);
      saveSplatInst(I2, &bundle, &ctx);
      break;
    }
    case Kinded::Kind::ElementMaxInstKind: {
      auto I2 = llvm::cast<ElementMaxInst>(&I);
      saveElemMaxInst(I2, &bundle, &ctx);
      break;
    }
    case Kinded::Kind::ElementAddInstKind: {
      auto I2 = llvm::cast<ElementAddInst>(&I);
      saveElemAddInst(I2, &bundle, &ctx);
      break;
    }
    case Kinded::Kind::ElementSubInstKind: {
      auto I2 = llvm::cast<ElementSubInst>(&I);
      saveElemSubInst(I2, &bundle, &ctx);
      break;
    }
    case Kinded::Kind::ElementDivInstKind: {
      auto I2 = llvm::cast<ElementDivInst>(&I);
      saveElemDivInst(I2, &bundle, &ctx);
      break;
    }
    case Kinded::Kind::AvgPoolInstKind: {
      auto I2 = llvm::cast<AvgPoolInst>(&I);
      saveAvgPoolInst(I2, &bundle, &ctx);
      break;
    }
    case Kinded::Kind::TensorViewInstKind: {
      auto I2 = llvm::cast<TensorViewInst>(&I);
      saveTensorViewInst(I2, &bundle, &ctx);

      // for not constant but represented as a constant value
      auto csyms = ctx.getConstantSymbols();
      csyms->push_back({I2->getName().data(), 0, I2->getSizeInBytes(), '1'});
      break;
    }
    case Kinded::Kind::FullyConnectedInstKind: {
      auto I2 = llvm::cast<FullyConnectedInst>(&I);
#if defined(VTA_BNN)
      saveBNNFullyConnectedInst(I2, &bundle, &ctx);
#else
      saveFullyConnectedInst(I2, &bundle, &ctx);
#endif
      break;
    }
    case Kinded::Kind::DequantizeInstKind: {
      auto I2 = llvm::cast<DequantizeInst>(&I);
      saveDequantizeInst(I2, &bundle, &ctx);
      break;
    }
    case Kinded::Kind::SoftMaxInstKind: {
      auto I2 = llvm::cast<SoftMaxInst>(&I);
      saveSoftMaxInst(I2, &bundle, &ctx);
      break;
    }
    case Kinded::Kind::ReluInstKind: {
      auto I2 = llvm::cast<ReluInst>(&I);
      saveReluInst(I2, &bundle, &ctx);
      break;
    }
    case Kinded::Kind::DebugPrintInstKind: {
      auto I2 = llvm::cast<DebugPrintInst>(&I);
      saveDebugPrintInst(I2, &bundle, &ctx);
      break;
    }
    case Kinded::Kind::ElementSignInstKind: {
      auto I2 = llvm::cast<ElementSignInst>(&I);
      saveElemSignInst(I2, &bundle, &ctx);
      break;
    }
    default:
      std::string msg = I.getKindName();
      msg.append(" is an unhandled instruction");

      llvm_unreachable(msg.c_str());
    }
#ifdef VTA_PROFILE
    if (I.getKind() != Kinded::Kind::DebugPrintInstKind) {
      bundle.append("  clock_t ");
      bundle.append(I.getName());
      bundle.append("_end = clock();\n");
      bundle.append("  prof_out<<\"");
      bundle.append(I.getKindName());
      bundle.append(":");
      bundle.append(I.getName());
      bundle.append(" : \"<<");
      bundle.append("(double)(");
      bundle.append(I.getName());
      bundle.append("_end - ");
      bundle.append(I.getName());
      bundle.append("_start)/CLOCKS_PER_SEC*1000 << std::endl;\n");
    }
#endif
  }
#ifdef VTA_PROFILE
  bundle.append("  prof_out.close();\n");
#endif
  wfos.close();
  finalizeBundleEntrySave(bundleCtx, bundleName, mainEntryName, ctx);
  exportBundleEntry(outputDir, bundleName, bundleCtx);
  exportVTARuntimeHeader(outputDir);
  exportBundleHeader(outputDir, bundleName, mainEntryName, ctx);

  return;
}
