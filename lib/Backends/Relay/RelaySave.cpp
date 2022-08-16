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
#include "Relay.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace glow;

// quoted for c++11
template <typename CharT> struct out_quoted {
  const CharT *s;
  CharT delim;
  CharT escape;
};

template <typename CharT>
out_quoted<CharT> quoted(const CharT *s, CharT delim = CharT('"'),
                         CharT escape = CharT('\\')) {
  return {s, delim, escape};
}

template <typename CharT>
out_quoted<CharT> quoted(const std::basic_string<CharT> &s,
                         CharT delim = CharT('"'), CharT escape = CharT('\\')) {
  return {s.c_str(), delim, escape};
}

template <typename CharT>
std::ostream &operator<<(std::ostream &os, const out_quoted<CharT> &q) {
  os << q.delim;
  for (const CharT *p = q.s; *p; p++) {
    if (*p == q.delim || *p == q.escape)
      os << q.escape << *p;
    else
      os << *p;
  }
  return os << q.delim;
}

// header wrapping
std::string hh(std::string org) {
  if (org.length() == 0)
    return "";
  std::stringstream ss11;
  ss11 << "header.append(" << quoted(org) << ")\n";
  return ss11.str();
}

// cpp code wrapping
std::string cc(std::string org) {
  if (org.length() == 0)
    return "";
  std::stringstream ss11;
  ss11 << "cpp.append(" << quoted(org) << ")\n";
  return ss11.str();
}

struct SaveCtx {
  std::string pyRelayCode; // tvm relay 처리를 위한 코드.
  std::string partHeaderGen; // gencode에서 part.h 생성하는 부분. python코드
  std::string partCppGen; // gencode에서 part.cpp 생성하는 부분. python코드
  std::string partMakeGen; // module 생성. Makefile
};

// context저장을 위해 vtasave context 가져옴.
// 사실상 weight는 object에 tvm형태로 저장되기 때문에 크게 의미는 없음.
// 직접 로드해서 binding해서 사용하는 경우만 의미가 있을듯

class RelaySaveContext {
public:
  RelaySaveContext() {
    constantWeightVarsMemSize = 0;
    mutableWeightVarsMemSize = 0;
    global_var_declare = "";
    var_declare = "";
  }
  std::vector<struct SymbolTableEntry> *getSymbols() { return &syms; }
  std::vector<struct SymbolTableEntry> *getConstantSymbols() { return &csyms; }
  uint64_t getCMemSize() { return constantWeightVarsMemSize; }
  void setCMemSize(uint64_t size) { constantWeightVarsMemSize = size; }
  uint64_t getMMemSize() { return mutableWeightVarsMemSize; }
  void setMMemSize(uint64_t size) { mutableWeightVarsMemSize = size; }
  void setGenCodeFileStream(std::ofstream *fos) { wfos = fos; }
  std::ofstream *getGenCodeFileStream() { return wfos; }
  /*
  void writeRelayCode()
  {

  }
  void writePartHeaderGenCode()
  {

  }
  void writePartCppGenCode()
  {

  }
*/

  std::string global_var_declare; // global variable
  std::string var_declare;        // variable in load module function
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string> input_DLTensor_names;
  std::vector<std::string> output_DLTensor_names;
  int qnn_mode = 0;

private:
  std::vector<struct SymbolTableEntry> syms;  // symbol
  std::vector<struct SymbolTableEntry> csyms; // constant
  uint64_t constantWeightVarsMemSize;
  uint64_t mutableWeightVarsMemSize;
  std::ofstream *wfos;
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

SymbolTableEntry addSymbolEntry(WeightVar *wgt, RelaySaveContext *ctx) {
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

// mutable
// inout = 1 = input, 2=output
SymbolTableEntry addSymbolEntryGenBundle(WeightVar *wgt, RelaySaveContext *ctx,
                                         int inout) {
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

  ctx->var_declare.append("cpp.append(\"");
  ctx->var_declare.append("  uint8_t* ");
  ctx->var_declare.append(ste.name);
  ctx->var_declare.append(" = (uint8_t*)mutableWeight + ");
  ctx->var_declare.append(std::to_string(ste.offset));
  ctx->var_declare.append(";");
  ctx->var_declare.append("\")\n");

  TypeRef T = wgt->getType();
  if (inout == 1) {

    ctx->input_names.push_back(ste.name);
    ctx->input_DLTensor_names.push_back("input_" + std::to_string(ste.offset));

    ctx->global_var_declare.append(
        cc("DLTensor input_" + std::to_string(ste.offset) + ";"));

    ctx->var_declare.append("cpp.append(\"");
    ctx->var_declare.append("std::vector<int64_t> input_shape_" +
                            std::to_string(ste.offset) + " = {");

    if (T->elementType_ == ElemKind::FloatTy) {
      for (int i = 0; i < T->numSizes_; i++) {
        ctx->var_declare.append(std::to_string((int)T->sizes_[i]));
        ctx->var_declare.append((i != T->numSizes_ - 1) ? "," : " ");
      }
    }

    ctx->var_declare.append("};");
    ctx->var_declare.append("\")\n");

    ctx->var_declare.append(cc("input_" + std::to_string(ste.offset) +
                               ".data = " + ste.name + ";"));
    ctx->var_declare.append(cc("input_" + std::to_string(ste.offset) +
                               ".device = DLDevice{kDLCPU, 0};"));
    ctx->var_declare.append(cc("input_" + std::to_string(ste.offset) +
                               ".ndim = " + std::to_string(T->numSizes_) +
                               ";"));
    ctx->var_declare.append(cc("input_" + std::to_string(ste.offset) +
                               ".dtype = DLDataType{kDLFloat, 32, 1};"));
    ctx->var_declare.append(cc("input_" + std::to_string(ste.offset) +
                               ".shape = input_shape_" +
                               std::to_string(ste.offset) + ".data();"));
    ctx->var_declare.append(
        cc("input_" + std::to_string(ste.offset) + ".strides = nullptr;"));
    ctx->var_declare.append(
        cc("input_" + std::to_string(ste.offset) + ".byte_offset = 0;"));

  } else {

    ctx->output_names.push_back(ste.name);
    ctx->output_DLTensor_names.push_back("output_" +
                                         std::to_string(ste.offset));

    ctx->global_var_declare.append(
        cc("DLTensor output_" + std::to_string(ste.offset) + ";"));

    ctx->var_declare.append("cpp.append(\"");
    ctx->var_declare.append("std::vector<int64_t> output_shape_" +
                            std::to_string(ste.offset) + " = {");
    if (T->elementType_ == ElemKind::FloatTy) {
      for (int i = 0; i < T->numSizes_; i++) {
        ctx->var_declare.append(std::to_string((int)T->sizes_[i]));
        ctx->var_declare.append((i != T->numSizes_ - 1) ? "," : " ");
      }
    }
    ctx->var_declare.append("};");
    ctx->var_declare.append("\");\n");

    ctx->var_declare.append(cc("output_" + std::to_string(ste.offset) +
                               ".data = " + (std::string)ste.name + ";"));
    ctx->var_declare.append(cc("output_" + std::to_string(ste.offset) +
                               ".device = DLDevice{kDLCPU, 0};"));
    ctx->var_declare.append(cc("output_" + std::to_string(ste.offset) +
                               ".ndim = " + std::to_string(T->numSizes_) +
                               ";"));
    ctx->var_declare.append(cc("output_" + std::to_string(ste.offset) +
                               ".dtype = DLDataType{kDLFloat, 32, 1};"));
    ctx->var_declare.append(cc("output_" + std::to_string(ste.offset) +
                               ".shape = output_shape_" +
                               std::to_string(ste.offset) + ".data();"));
    ctx->var_declare.append(
        cc("output_" + std::to_string(ste.offset) + ".strides = nullptr;"));
    ctx->var_declare.append(
        cc("output_" + std::to_string(ste.offset) + ".byte_offset = 0;"));
  }

  return ste;
}

SymbolTableEntry addConstantSymbolEntry(Value *val, RelaySaveContext *ctx) {
  auto csyms = ctx->getConstantSymbols();
  struct SymbolTableEntry ste;
  for (auto s : *csyms) {
    auto name = (const char *)(val->getName().data());
    if (strcmp(s.name, name) == 0) {
      ste = s;
      return ste;
    }
  }
  csyms->push_back(
      {val->getName().data(), ctx->getCMemSize(), val->getSizeInBytes(), '0'});
  ctx->setCMemSize(ctx->getCMemSize() + val->getSizeInBytes());
  ste = csyms->back();
  return ste;
}

// typedef struct {
//
//  void* data;
//
//  int device_type;  // 1=CPU
//  int devicc_id;   // 0
//
//  int ndim;
//  /*! \brief The data type of the pointer*/
//  uint8_t dtype;  // 0=signed int, 1=uint, 2=float, 3=opaque,
//  4=float16,5=complex uint8_t bits; // num of bits 8,16,32 uint16_t lanes; //
//  used for vector type
//
//  /*! \brief The shape of the tensor */
//  int64_t* shape;
//  /*!
//   * \brief strides of the tensor (in number of elements, not bytes)
//   *  can be NULL, indicating tensor is compact and row-majored.
//   */
//  int64_t* strides;
//  /*! \brief The offset in bytes to the beginning pointer to data */
//  uint64_t byte_offset;
//} DLTensor__;

// dltensor로 저장해야 tvm 에서 바로 load가능
/*
void SaveParams(ofstream *strm, const Map<String, NDArray>& params) {
  std::vector<std::string> names;
  std::vector<const DLTensor*> arrays;
  for (auto& p : params) {
    names.push_back(p.first);
    arrays.push_back(p.second.operator->());
  }

  uint64_t header = kTVMNDArrayListMagic, reserved = 0;
  strm->Write(header);
  strm->Write(reserved);
  strm->Write(names);
  {
    uint64_t sz = static_cast<uint64_t>(arrays.size());
    strm->Write(sz);
    for (size_t i = 0; i < sz; ++i) {
      tvm::runtime::SaveDLTensor(strm, arrays[i]);
    }
  }
}
*/

void initCtx(struct SaveCtx &Ctx, llvm::StringRef bundleName,
             llvm::StringRef mainEntryName) {

  auto &inc = Ctx.partHeaderGen;
  auto &cpp = Ctx.partCppGen;
  auto &py = Ctx.pyRelayCode;
  auto &mk = Ctx.partMakeGen;

  inc = "";
  cpp = "";
  py = "";
  mk = "";

  // part.h

  inc.append("\n\n## output header");
  inc.append("\nheader = []\n");
  inc.append(hh("// Bundle API auto-generated header file. Do not edit!"));
  inc.append(hh("// tvm nest-c Tools version: 2021-07-07"));

  inc.append(hh("#ifndef _GLOW_BUNDLE_" + (std::string)bundleName + "_H"));
  inc.append(hh("#define _GLOW_BUNDLE_" + (std::string)bundleName + "_H"));

  inc.append(hh("#include <dlpack/dlpack.h>"));
  inc.append(hh("#include <stdint.h>"));
  inc.append(hh("#include <vector>"));

  inc.append(
      hh("// ---------------------------------------------------------------"));
  inc.append(hh("//                       Common definitions"));
  inc.append(
      hh("// ---------------------------------------------------------------"));
  inc.append(hh("#ifndef _GLOW_BUNDLE_COMMON_DEFS"));
  inc.append(hh("#define _GLOW_BUNDLE_COMMON_DEFS"));

  inc.append(hh("// Glow bundle error code for correct execution."));
  inc.append(hh("#define GLOW_SUCCESS 0"));

  inc.append(hh("// a node info table for TVM node debugging"));
  inc.append(hh("struct TVMNodeInfoTable {"));
  inc.append(hh("\t// Name of a variable."));
  inc.append(hh("\tconst char *name;"));
  inc.append(hh("\t// dim"));
  inc.append(hh("\tuint64_t dim;"));
  inc.append(hh("\t// size"));
  inc.append(hh("\tuint64_t size;"));
  inc.append(hh("\t// type. kDLInt, kDLFloat"));
  inc.append(hh("\tDLDataType type;"));
  inc.append(hh("\t// shape string. ex {1,64,64,3}"));
  inc.append(hh("\tint64_t shape[6];"));
  inc.append(hh("};"));

  inc.append(
      hh("// Type describing a symbol table entry of a generated bundle."));
  inc.append(hh("struct SymbolTableEntry {"));
  inc.append(hh("\t// Name of a variable."));
  inc.append(hh("\tconst char *name;"));
  inc.append(hh("\t// Offset of the variable inside the memory area."));
  inc.append(hh("\tuint64_t offset;"));
  inc.append(hh("\t// The number of elements inside this variable."));
  inc.append(hh("\tuint64_t size;"));
  inc.append(
      hh("\t// Variable kind: 1 if it is a mutable variable, 0 otherwise."));
  inc.append(hh("\tchar kind;"));
  inc.append(hh("};"));

  inc.append(hh("// Type describing the config of a generated bundle."));
  inc.append(hh("struct BundleConfig {"));
  inc.append(hh("\t// Size of the constant weight variables memory area."));
  inc.append(hh("\tuint64_t constantWeightVarsMemSize;"));
  inc.append(hh("\t// Size of the mutable weight variables memory area."));
  inc.append(hh("\tuint64_t mutableWeightVarsMemSize;"));
  inc.append(hh("\t// Size of the activations memory area."));
  inc.append(hh("\tuint64_t activationsMemSize;"));
  inc.append(hh("\t// Alignment to be used for weights and activations."));
  inc.append(hh("\tuint64_t alignment;"));
  inc.append(hh("\t// Number of symbols in the symbol table."));
  inc.append(hh("\tuint64_t numSymbols;"));
  inc.append(hh("\t// Symbol table."));
  inc.append(hh("\tconst SymbolTableEntry *symbolTable;"));
  inc.append(hh("};"));

  inc.append(hh("#endif"));

  // part.cpp
  cpp.append("\n\n## cpp bundle wrapper");
  cpp.append("\ncpp = []\n");

  cpp.append(cc("// Generate Include Header"));
  cpp.append(cc("#include \"" + (std::string)mainEntryName + ".h\""));

  cpp.append(cc("#include <dlpack/dlpack.h>"));
  cpp.append(cc("#include <tvm/runtime/module.h>"));
  cpp.append(cc("#include <tvm/runtime/packed_func.h>"));
  cpp.append(cc("#include <tvm/runtime/registry.h>"));
  cpp.append(cc(""));
  cpp.append(cc("#include <cstdio>"));
  cpp.append(cc("#include <assert.h>"));
  cpp.append(cc("using namespace std;"));

  //
  py.append("import numpy as np\nimport tvm\nfrom tvm import te, "
            "runtime\nimport tvm.relay as relay\n"
            "from tvm.relay.frontend.common import infer_type\nfrom "
            "tvm.relay.testing import check_grad, run_infer_type, "
            "run_opt_pass, _np_randn_from_type\n"
            "import onnx\nfrom tvm.relay import op as _op\nimport json\nimport "
            "math\n\ndef load_wgt(filename, "
            "shape,dtype):\n\tf=open(filename,\"rb\")\n\td=f.read()\n\treturn "
            "np.frombuffer(d, dtype=dtype).reshape(shape)\n\n");
}

void finalizeCtx(struct SaveCtx &Ctx, llvm::StringRef outputDir,
                 llvm::StringRef bundleName, llvm::StringRef mainEntryName,
                 RelaySaveContext &procCtx, std::string target,
                 std::string target_host, std::string export_option,
                 uint32_t tvm_opt_level, std::string required_pass,
                 std::string disabled_pass, std::string debug_mode) {

  auto &inc = Ctx.partHeaderGen;
  auto &cpp = Ctx.partCppGen;
  auto &py = Ctx.pyRelayCode;
  auto &mk = Ctx.partMakeGen;

  std::string relay_mkpath =
      (std::string)outputDir + "/relay__" + (std::string)bundleName;
  std::string module_mkpath =
      (std::string)outputDir + "/module__" + (std::string)bundleName;
  std::string module_rel_path = "module__" + (std::string)bundleName;

  //#input에 대해서는 이름을 알 수 있지만
  //#output은 별도 name을 유지하지 않음. index로 가져옴.
  //#name은 그냥 output으로 통일
  //#total data size는 의미가 좀 다름. serialized된 것이라 순수 data만 나타낼 수
  //없음. #따로 계산해서 가능하긴 한데 arch에 따라서 DLTensor가 달라지면 다를 수
  //있음
  inc.append(
      hh("// ---------------------------------------------------------------"));
  inc.append(hh("//                          Bundle API"));
  inc.append(
      hh("// ---------------------------------------------------------------"));
  /*
    inc.append(hh("// Model name: "));
    inc.append(hh(bundleName));
    inc.append(hh("// Total data size: " +
    std::to_string(procCtx.getMMemSize()))); inc.append(hh("// Placeholders:"));
    inc.append(hh("//"));

    for(auto ste : *procCtx.getSymbols()) {
      auto wgt = ste.wgt;
      assert(wgt);
      inc.append(hh("//   Name: "));
      inc.append(hh(ste.name ));
      //type
      inc.append(hh("//   Size: " + std::to_string(wgt->size()) +
    "(elements)")); inc.append(hh("//   Size: " +
    std::to_string(wgt->getSizeInBytes()) + "(bytes)")); inc.append(hh("//
    Offset: " + std::to_string(ste.offset) + "(bytes)"));

    };

    inc.append(hh("// NOTE: Placeholders are allocated within the
    \"mutableWeight\")")); inc.append(hh("// buffer and are identified using an
    offset relative to base.)")); inc.append(hh("//
    ---------------------------------------------------------------"));
  */
  inc.append(hh("#ifdef __cplusplus"));
  inc.append(hh("extern \"C\" {"));
  inc.append(hh("#endif"));

  inc.append(hh("// Bundle memory configuration (memory layout)."));
  inc.append(hh("extern BundleConfig " + (std::string)bundleName + "_config;"));

  inc.append(hh("// Bundle entry point (inference function). Returns 0"));
  inc.append(hh("// for correct execution or some error code otherwise."));
  inc.append(hh("int " + (std::string)mainEntryName +
                "_load_module(uint8_t *constantWeight);"));
  inc.append(hh("int " + (std::string)mainEntryName +
                "(uint8_t *constantWeight, uint8_t *mutableWeight, uint8_t "
                "*activations);"));
  inc.append(hh("int " + (std::string)mainEntryName + "_destroy_module();"));
  inc.append(hh("#ifdef __cplusplus"));
  inc.append(hh("}"));
  inc.append(hh("#endif"));
  inc.append(hh("#endif"));

  inc.append("with open(\"" + module_rel_path + "/" + (std::string)bundleName +
             ".h\",\"w\") as f_h:\n");
  inc.append("  for item in header:\n");
  inc.append("    f_h.write(\"%s\\n\" % item)\n");

  /*
    std::string llvm_option="";
    std::string export_option="";
    if(target_arch == "aarch64") {
      llvm_option = " -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon";
      export_option = ", cc='aarch64-linux-gnu-g++'";
    }
  */
  if (tvm_opt_level > 4)
    tvm_opt_level = 0;

  // check interface params
  if (target == "")
    target = "llvm";
  if (target == "aarch64") {
    target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon";
  }
  if (target_host == "aarch64") {
    target_host = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu";
  }
  if (export_option == "aarch64") {
    export_option = ", cc='aarch64-linux-gnu-g++'";
  }
  std::string target_host_str = "";
  if (target_host != "") {
    target_host_str = ",host=\"" + target_host + "\"";
  }

  std::string opt_str = "";
  if (required_pass != "") {
    std::stringstream sss;
    sss << ",required_pass=" << quoted(required_pass);
    opt_str += sss.str();
  }
  if (disabled_pass != "") {
    std::stringstream sss;
    sss << ",disabled_pass=" << quoted(disabled_pass);
    opt_str += sss.str();
  }

  std::string additional_pass = "";
  if (procCtx.qnn_mode == 1) {
    additional_pass += "\nrelay_mod = relay.transform.InferType()(relay_mod) \
                      \nrelay_mod = relay.qnn.transform.CanonicalizeOps()(relay_mod)";
  }

  py.append(
      "\ndesired_layouts = { \"nn.conv2d\": [\"NCHW\", \"default\"], \"qnn.conv2d\": [\"NCHW\", \"default\"]}  \
             \nseq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(), \
             \n\t\trelay.transform.ConvertLayout(desired_layouts)]) \
             \nwith tvm.transform.PassContext(opt_level=3): \
            \n\t\trelay_mod = seq(tvm.IRModule.from_expr(func))" +
      additional_pass + " \
            \n\n");
  if (target == "mali") {
    py.append("target=tvm.target.mali()");
  } else {
    py.append("target=tvm.target.Target(\"" + target + "\"" + target_host_str +
              ") ");
  }
  py.append("\nwith tvm.transform.PassContext(opt_level=" +
            std::to_string(tvm_opt_level) + opt_str + "): ");
  if (target == "mali") {
    py.append(
        "\n\t\tlib = relay.build(relay_mod, target=target, target_host=\"llvm "
        "-device=arm_cpu -mtriple=aarch64-linux-gnu\",params=params)");
  } else {
    py.append("\n\t\tlib = relay.build(relay_mod, target,params=params)");
  }
  py.append("\nlib.export_library(\"" + module_rel_path + "/" +
            (std::string)bundleName + "_tvm.so\"" + export_option + ")        \
    \n# = lib.get_params()\
    \n#    for item in b:\
    \n#        print(item)\
    \n#        print(item.nbytes)");

  if (debug_mode != "") {
    py.append("\ngraph = lib.get_graph_json() \
    \nwith open(\"" +
              relay_mkpath + "/graph.json\",\"w\") as f_graph: \
    \n  f_graph.write(\"%s\\n\" % graph)\n \
    \ngraph_node = json.loads(graph)\n \
    \nidx=0 \
    \nnode_str=[]\
    \nnode_str.append('int debug_node_count=' + str(len(graph_node['nodes'])) +';\\n')\
    \nnode_str.append('struct TVMNodeInfoTable debug_node_info[]={')\
    \nmax_size = 0\
    \nwhile idx < len(graph_node['nodes']):\
    \n  if idx!=0:\
    \n    node_str.append(',')\
    \n  size = 4\
    \n  type_str = \"kDLFloat\"\
    \n  if graph_node['attrs']['dltype'][1][idx]==\"int8\":\
    \n    size = 1\
    \n    type_str = \"kDLInt\"\
    \n  elif graph_node['attrs']['dltype'][1][idx]==\"int32\":\
    \n    size = 4\
    \n    type_str = \"kDLInt\"\
    \n  total_size = size *math.prod(graph_node['attrs']['shape'][1][idx])\
    \n  if total_size > max_size:\
    \n     max_size=total_size\
    \n  arr = ','.join(str(e) for e in graph_node['attrs']['shape'][1][idx])\
    \n  node_str.append(\"{\\\"\" + graph_node['nodes'][idx]['name'] + \"\\\",\" + str(len(graph_node['attrs']['shape'][1][idx])) + \",\" + str(total_size)+ \",{\" + type_str + \",\" + str(size*8) + \",1},{\"+arr+\"}}\")\
    \n  idx += 1\
    \nnode_str.append('};\\n')\
    \nnode_str.append('int debug_buf_size='+str(max_size)+';\\n')\
    \nnode_str = ''.join(node_str)");
  }

  if (debug_mode != "") {
    std::string debugNodeInfoTable;
    debugNodeInfoTable = "TVMNodeInfoTable debugNode[] = {";
    // for(auto sym : *(procCtx.getSymbols())){

    debugNodeInfoTable.append("};");
  }

  // Generate SymbolTable struct
  std::string symbolTable;
  symbolTable = "SymbolTableEntry symbolTableEntry_";
  symbolTable.append(bundleName);
  symbolTable.append("[");
  symbolTable.append(std::to_string(procCtx.getSymbols()->size()));
  symbolTable.append("]={");
  for (auto sym : *(procCtx.getSymbols())) {
    symbolTable.append("{\"");
    symbolTable.append(sym.name);
    symbolTable.append("\",");
    symbolTable.append(std::to_string(sym.offset));
    symbolTable.append(",");
    symbolTable.append(std::to_string(sym.size));
    symbolTable.append(",'1'},");
  }
  symbolTable.erase(symbolTable.size() - 1, 1);
  symbolTable.append("};");

  // std::cout << symbolTable.c_str();

  cpp.append(cc(symbolTable));

  // cpp
  cpp.append(
      cc("struct BundleConfig " + (std::string)bundleName + "_config = {"));
  cpp.append(cc("   " + std::to_string(procCtx.getCMemSize()) + ","));
  cpp.append(cc("   " + std::to_string(procCtx.getMMemSize()) + ","));
  cpp.append(cc("   0, 64, ")); // alignment always 64?
  cpp.append(cc("   " + std::to_string(procCtx.getSymbols()->size()) + ","));
  cpp.append(cc("   symbolTableEntry_" + (std::string)bundleName));
  cpp.append(cc("};"));

  cpp.append(cc("namespace namespace_" + (std::string)bundleName + " {"));
  // global var
  cpp.append(procCtx.global_var_declare); // mutable variables
  // create the graph executor module
  cpp.append(cc("tvm::runtime::Module gmod;"));
  cpp.append(cc("tvm::runtime::PackedFunc set_input;"));
  cpp.append(cc("tvm::runtime::PackedFunc get_output;"));
  cpp.append(cc("tvm::runtime::PackedFunc run;"));

  if (debug_mode != "") {
    cpp.append(cc("tvm::runtime::PackedFunc get_node_output;"));
    cpp.append(cc("tvm::runtime::PackedFunc debug_get_output;"));
  }

  cpp.append(cc("} // end namespace"));
  cpp.append(cc("using namespace namespace_" + (std::string)bundleName + ";"));

  //
  std::string dl_dev_str = "kDLCPU"; // DLCPU

  if (target.find("cuda") != std::string::npos) {
    dl_dev_str = "kDLCUDA";
  } else if ((target.find("opencl") != std::string::npos) ||
             (target.find("mali") != std::string::npos)) {
    dl_dev_str = "kDLOpenCL";
  }

  // function load_module
  cpp.append(cc("int " + (std::string)bundleName +
                "_load_module(uint8_t* constantWeight) {"));
  cpp.append(cc("  DLDevice dev{" + dl_dev_str + ", 0};"));
  // cpp.append(cc(" tvm::runtime::Module mod_factory =
  // tvm::runtime::Module::LoadFromFile(\"" +  module_mkpath + "/" +
  // (std::string)bundleName + "_tvm.so\");"))  ;
  cpp.append(cc(" tvm::runtime::Module mod_factory = "
                "tvm::runtime::Module::LoadFromFile(\"./" +
                (std::string)bundleName + "_tvm.so\");"));

  // create the graph executor module
  if (debug_mode == "") {
    cpp.append(cc(" gmod = mod_factory.GetFunction(\"default\")(dev);"));
  } else {
    cpp.append(cc(
        " gmod = mod_factory.GetFunction(\"debug_create\")(\"default\",dev);"));
    cpp.append(cc(" get_node_output = gmod.GetFunction(\"get_node_output\");"));
    cpp.append(
        cc(" debug_get_output = gmod.GetFunction(\"debug_get_output\");"));
  }

  cpp.append(cc(" set_input = gmod.GetFunction(\"set_input\");"));
  cpp.append(cc(" get_output = gmod.GetFunction(\"get_output\");"));
  cpp.append(cc(" run = gmod.GetFunction(\"run\");"));
  cpp.append(cc("  return 0;"));
  cpp.append(cc("}"));

  // function run
  cpp.append(cc("int " + (std::string)bundleName +
                "(uint8_t* constantWeight, uint8_t *mutableWeight, uint8_t "
                "*activations) {"));
  cpp.append(cc(""));
  cpp.append(procCtx.var_declare); // mutable variables

  // set the right input
  for (int i = 0; i < procCtx.input_names.size(); i++) {
    cpp.append(cc(" set_input(\"" + procCtx.input_names[i] + "\", &" +
                  procCtx.input_DLTensor_names[i] + ");"));
  }

  // run the code
  cpp.append(cc(" run();"));

  // get the output
  for (int i = 0; i < procCtx.output_DLTensor_names.size(); i++) {
    cpp.append(cc("  get_output(" + std::to_string(i) + ", &" +
                  procCtx.output_DLTensor_names[i] + ");"));
  }

  if (debug_mode != "") {
    cpp.append(cc("  char filename[255];"));
    cpp.append(cc("  DLTensor dbg_out;"));
    cpp.append(cc(" char* dbg_buf = (char*)malloc(debug_buf_size);"));
    cpp.append(cc(" dbg_out.data = dbg_buf;"));
    cpp.append(cc(" dbg_out.device = DLDevice{kDLCPU, 0};"));
    cpp.append(cc(" dbg_out.strides = nullptr;"));
    cpp.append(cc(" dbg_out.byte_offset = 0; "));
    cpp.append(cc(" for(int k=0;k<debug_node_count;k++) {"));
    cpp.append(cc("   dbg_out.ndim = debug_node_info[k].dim;"));
    cpp.append(cc("   dbg_out.dtype = debug_node_info[k].type;"));
    cpp.append(cc("   dbg_out.shape = debug_node_info[k].shape;"));
    cpp.append(cc("   debug_get_output(debug_node_info[k].name,&dbg_out);"));

    if (debug_mode == "txt") {
      cpp.append(cc("   sprintf(filename, \"" + module_rel_path +
                    "/%d__%s.txt\", k, debug_node_info[k].name);"));
      cpp.append(cc("   FILE *fp = fopen(filename ,\"w+\");"));
      cpp.append(cc("   if(fp==0) { // partition mode"));
      cpp.append(cc("     sprintf(filename, \"Relay/" + module_rel_path +
                    "/%d__%s.txt\", k, debug_node_info[k].name);"));
      cpp.append(cc("     fp = fopen(filename ,\"w+\");"));
      cpp.append(cc("   }"));
      cpp.append(cc("   for(int "
                    "m=0;m<debug_node_info[k].size;m+=(debug_node_info[k].type."
                    "bits/8)) {"));
      cpp.append(cc("     if(debug_node_info[k].type.code == kDLInt && "
                    "debug_node_info[k].type.bits==8) {"));
      cpp.append(cc("       fprintf(fp,\"%d \", *(char*)(dbg_buf+m));"));
      cpp.append(cc("     } else if(debug_node_info[k].type.code == kDLInt && "
                    "debug_node_info[k].type.bits==32) {"));
      cpp.append(cc("       fprintf(fp,\"%d \", *(int*)(dbg_buf+m));"));
      cpp.append(cc("     } else if(debug_node_info[k].type.code == kDLUInt && "
                    "debug_node_info[k].type.bits==8) {"));
      cpp.append(
          cc("       fprintf(fp,\"%d \", *(unsigned char*)(dbg_buf+m));"));
      cpp.append(cc("     } else if(debug_node_info[k].type.code == kDLUInt && "
                    "debug_node_info[k].type.bits==32) {"));
      cpp.append(
          cc("       fprintf(fp,\"%d \", *(unsigned int*)(dbg_buf+m));"));
      cpp.append(cc("     } else if(debug_node_info[k].type.code == kDLFloat "
                    "&& debug_node_info[k].type.bits==32) {"));
      cpp.append(cc("        fprintf(fp,\"%.6f \", *(float*)(dbg_buf+m));"));
      cpp.append(cc("     }"));
      cpp.append(cc("   }"));
      cpp.append(cc("   fclose(fp);"));

      cpp.append(cc("   debug_get_output(debug_node_info[k].name,&dbg_out);"));
    } else if (debug_mode == "bin") {
      cpp.append(cc("   debug_get_output(debug_node_info[k].name,&dbg_out);"));
      cpp.append(cc("   sprintf(filename, \"" + module_rel_path +
                    "/%d__%s.bin\", k, debug_node_info[k].name);"));
      cpp.append(cc("   FILE *fp = fopen(filename ,\"wb+\");"));
      cpp.append(cc("   if(fp==0) { // partition mode"));
      cpp.append(cc("     sprintf(filename, \"Relay/" + module_rel_path +
                    "/%d__%s.bin\", k, debug_node_info[k].name);"));
      cpp.append(cc("     fp = fopen(filename ,\"wb+\");"));
      cpp.append(cc("   }"));
      cpp.append(
          cc("   int r = fwrite( dbg_buf, 1, debug_node_info[k].size, fp);"));
      cpp.append(cc("   if (r < debug_node_info[k].size) { "));
      cpp.append(cc("      printf(\"dump error: %s, write %d / %ld \\n\", "
                    "filename, r, debug_node_info[k].size); "));
      cpp.append(cc("   }"));
      cpp.append(cc("   fclose(fp);"));
    }

    cpp.append(cc("  }"));
  }

  cpp.append(cc("  return 0;"));
  cpp.append(cc("}"));

  // function destory_module
  cpp.append(cc("int " + (std::string)bundleName + "_destroy_module() {"));
  cpp.append(cc("  return 0;"));
  cpp.append(cc("}"));

  if (debug_mode != "") {
    cpp.append("cpp.insert(9,node_str)\n");
  }

  cpp.append("with open(\"" + module_rel_path + "/" + (std::string)bundleName +
             ".cpp\",\"w\") as f_cpp:\n");
  cpp.append("  for item in cpp:\n");
  cpp.append("    f_cpp.write(\"%s\\n\" % item)\n");

  mk.append("DMLC_CORE=${TVM_HOME}/3rdparty/dmlc-core\n\n");
  mk.append("PKG_CFLAGS = -std=c++14 -g -fPIC\\\n");
  mk.append("\t-I${TVM_HOME}/include\\\n");
  mk.append("\t-I${DMLC_CORE}/include\\\n");
  mk.append("\t-I${TVM_HOME}/3rdparty/dlpack/include\\\n");
  mk.append("\t-DDMLC_USE_LOGGING_LIBRARY=\\<tvm/runtime/logging.h\\>\n\n");
  mk.append("PKG_LDFLAGS = -L${TVM_LIBRARY_PATH} -ldl -pthread \n\n");

  mk.append(".PHONY: clean all\n\n");
  mk.append("all: " + (std::string)bundleName + ".o\n\n");
  mk.append((std::string)bundleName + ".o: " + (std::string)bundleName +
            ".cpp\n");
  mk.append("\t@mkdir -p $(@D)\n");
  if (target.find("aarch64") != std::string::npos ||
      target_host.find("aarch64") != std::string::npos) {
    mk.append("\taarch64-linux-gnu-g++ $(PKG_CFLAGS) -c -o $@  $^ "
              "-ltvm_runtime $(PKG_LDFLAGS)\n");
  } else {
    mk.append(
        "\t$(CXX) $(PKG_CFLAGS) -c -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)\n");
  }

  mk.append("clean:\n");
  mk.append("\trm -rf lib\n");
}

static int getInOut(const Value *V) {

  for (const auto &U : ValueUses(V)) {
    Instruction *user = U.get();

    // Ignore deallocs.
    if (user->getKind() == Kinded::Kind::DeallocActivationInstKind) {
      continue;
    }

    auto op = U.getOperand();

    // Ignore the readers.
    if (op.second == OperandKind::In) {
      return 1;
    } else if (op.second == OperandKind::Out) {
      return 2;
    }
  }

  return 0;
}

void Relay::save(Function *F, llvm::StringRef outputDir,
                 llvm::StringRef bundleName,
                 llvm::StringRef mainEntryName) const {

  struct SaveCtx saveCtx;   // ctx for save files
  RelaySaveContext procCtx; // ctx for processing data
  std::string variableString = "";

  std::vector<std::string> params_name;

  int status;
  std::string relay_mkpath =
      (std::string)outputDir + "/relay__" + (std::string)bundleName;
  status = mkdir(relay_mkpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  std::string module_mkpath =
      (std::string)outputDir + "/module__" + (std::string)bundleName;
  status = mkdir(module_mkpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  initCtx(saveCtx, bundleName, mainEntryName);
  std::stringstream pyss;

  auto &inc = saveCtx.partHeaderGen;
  auto &cpp = saveCtx.partCppGen;
  auto &py = saveCtx.pyRelayCode;
  auto &mk = saveCtx.partMakeGen;

  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());

  //  for (const auto &VV : IR->getVariableMap()) {
  // }

  // check in/out count first. use last input as output if there is no output
  int in_count = 0;
  int out_count = 0;
  for (const auto &W : IR->getWeights()) {

    if (W->getMutability() != WeightVar::MutabilityKind::Constant) {
      int type = getInOut(W);
      if (type == 2) {
        out_count++;
      } else if (type == 1) {
        in_count++;
      }
    }
  }

  int mutable_count = 0;
  std::string output_temp_name = "";
  WeightVar *output_temp_var;
  for (const auto &W : IR->getWeights()) {
    // Exception for SoftMaxGradNode
    if ((std::string)W->getName() == "selected") {
      continue;
    }

    // debug
    // std::cout << W->getKindName() << "] " << W->toString() << std::endl;

    // weight 별도 저장
    // tvm에서 활용하는 params 형태로 변경가능한게 좋음가 좋음
    // int32 갯수
    // {name, }
    std::string dtype_str = "";
    TypeRef T = W->getType();

    // std::cout << T->toString() << std::endl;

    if (T->elementType_ == ElemKind::FloatTy) {
      pyss << (std::string)W->getName() << " = relay.var(\""
           << (std::string)W->getName() << "\", shape=(";
      for (int i = 0; i < T->numSizes_; i++) {
        pyss << (int)T->sizes_[i] << ((i != T->numSizes_ - 1) ? "," : "");
      }
      if (T->numSizes_ == 1)
        pyss << ",";
      pyss << "),dtype=\"float32\")" << std::endl;
      dtype_str = "\"float32\"";

    } else if (T->elementType_ == ElemKind::Int32QTy) {
      pyss << (std::string)W->getName() << " = relay.var(\""
           << (std::string)W->getName() << "\", shape=(";
      for (int i = 0; i < T->numSizes_; i++) {
        pyss << (int)T->sizes_[i] << ((i != T->numSizes_ - 1) ? "," : "");
      }
      if (T->numSizes_ == 1)
        pyss << ",";
      pyss << "),dtype=\"int32\")" << std::endl;
      dtype_str = "\"int32\"";

    } else if (T->elementType_ == ElemKind::Int8QTy) {
      pyss << (std::string)W->getName() << " = relay.var(\""
           << (std::string)W->getName() << "\", shape=(";
      for (int i = 0; i < T->numSizes_; i++) {
        pyss << (int)T->sizes_[i] << ((i != T->numSizes_ - 1) ? "," : "");
      }
      if (T->numSizes_ == 1)
        pyss << ",";
      pyss << "),dtype=\"int8\")" << std::endl;
      dtype_str = "\"int8\"";

    } else {

      std::cout << (int)T->elementType_ << std::endl;
    }

    // save weight
    if (W->getMutability() == WeightVar::MutabilityKind::Constant) {

      auto vMap = IR->getVariableMap();
      const Tensor *tensor = NULL;
      for (auto it = vMap.begin(); it != vMap.end(); it++) {
        if (it->second == W) {
          auto storage = it->first;
          if (storage->getKind() == glow::Kinded::Kind::ConstantKind) {
            const auto *constant = llvm::cast<Constant>(storage);
            tensor = &(constant->getPayload());
          }
          break;
        }
      }
      assert(tensor);
      if (T->elementType_ == ElemKind::FloatTy) {
        auto handle = tensor->getHandle<float>();

        std::ofstream fos(relay_mkpath + "/" + (std::string)W->getName(),
                          std::ios::out);
        for (size_t i = 0, e = handle.size(); i < e; i++) {
          auto data = handle.raw(i);
          fos.write((const char *)&data, 4);
        }
      } else if (T->elementType_ == ElemKind::Int32QTy) {
        auto handle = tensor->getHandle<int32_t>();

        std::ofstream fos(relay_mkpath + "/" + (std::string)W->getName(),
                          std::ios::out);
        for (size_t i = 0, e = handle.size(); i < e; i++) {
          auto data = handle.raw(i);
          fos.write((const char *)&data, 4);
        }
      } else if (T->elementType_ == ElemKind::Int8QTy) {
        auto handle = tensor->getHandle<int8_t>();

        std::ofstream fos(relay_mkpath + "/" + (std::string)W->getName(),
                          std::ios::out);
        for (size_t i = 0, e = handle.size(); i < e; i++) {
          auto data = handle.raw(i);
          fos.write((const char *)&data, 1);
        }
      }

      addConstantSymbolEntry(W, &procCtx);

      // load in python
      // pyss<< "W_" << (std::string) W->getName() << "= load_wgt(" << "\"" <<
      // relay_mkpath << "/" << (std::string) W->getName() << "\",(";
      pyss << "W_" << (std::string)W->getName() << "= load_wgt("
           << "\"relay__" << (std::string)bundleName << "/"
           << (std::string)W->getName() << "\",(";
      for (int i = 0; i < T->numSizes_; i++) {
        pyss << (int)T->sizes_[i] << ((i != T->numSizes_ - 1) ? "," : "");
      }
      if (T->numSizes_ == 1)
        pyss << ",";
      pyss << "), " << dtype_str << ")" << std::endl;

      params_name.push_back(W->getName());

    } else // mutable
    {

      int type = getInOut(W);
      if (type == 2) {
        std::cout << "OUT // " << W->getKindName() << " :  " << W->toString()
                  << std::endl;
      } else if (type == 1) {
        std::cout << "IN // " << W->getKindName() << " :  " << W->toString()
                  << std::endl;
      }

      mutable_count++;
      if (out_count == 0 && mutable_count == in_count) {
        output_temp_name = W->getName().data();
        output_temp_var = W;
        // std::cout << "TEMP_OUTPUT" << output_temp_name;

      } else {
        addSymbolEntryGenBundle(W, &procCtx, type);
      }
    }
  }

  std::cout << "mutable var checked" << std::endl;

  pyss << "params = {";
  for (auto s : params_name) {
    pyss << "\"" << s << "\":W_" << s << ",";
  }
  pyss << "}" << std::endl;

  int ir_idx = 0;
  // Instruction *prevInst;
  std::string prev_alloc_name = "";
  TypeRef prev_alloc_type;

  for (auto &I : IR->getInstrs()) {

    // std::cout<< (std::string)I.toString()<<std::endl;

    switch (I.getKind()) {
    case Kinded::Kind::DeallocActivationInstKind: {
      // auto II =  static_cast<DeallocActivationInst*>(&I);
      // std::cout<< "del " << (std::string)II->getSrc()->getName()<<std::endl;
      // DO nothing
    } break;

    case Kinded::Kind::ReluInstKind: {
      // y2 = relay.nn.relu(y2)
      auto II = static_cast<ReluInst *>(&I);

      pyss << (std::string)II->getOperand(0).first->getName()
           << " = relay.nn.relu("
           << (std::string)II->getOperand(1).first->getName() << ")"
           << std::endl;
      // std::cout<< (std::string)I.toString()<<std::endl;

    } break;

    case Kinded::Kind::AllocActivationInstKind: {

      // inputs = {"input": ((1, 224, 224, 3), "uint8")}
      // x = relay.var("x", shape=(1, 3, 224, 224))
      //  data = relay.var("data", relay.TensorType(d_shape, "float32"))
      // weight = relay.var("weight", relay.TensorType(w_shape, "float32"))
      auto II = static_cast<AllocActivationInst *>(&I);
      TypeRef T = II->getTy();

      prev_alloc_type = T;
      prev_alloc_name = (std::string)I.getName();

      /*
              if(T->elementType_ == ElemKind::FloatTy) {
                std::cout<< (std::string) I.getName() << " = relay.var(\"" <<
         (std::string) I.getName() << "\", shape=( "; for(int
         i=0;i<T->numSizes_;i++) { std::cout<< (int)T->sizes_[i] <<
         ((i!=T->numSizes_-1) ? "," : "");
                }
                if(T->numSizes_ == 1) std::cout<< ",";
                std::cout<< "),dtype=\"float32\")"<<std::endl;

              }
              */

    } break;

    case Kinded::Kind::MaxPoolInstKind: {
      // net = relay.nn.max_pool2d(net,  pool_size=(3, 3), strides=(2, 2),
      // padding=(1, 1))

      auto II = static_cast<MaxPoolInst *>(&I);
      auto kernel = II->getKernels();
      auto stride = II->getStrides();
      auto padding = II->getPads();

      pyss << (std::string)II->getOperand(0).first->getName()
           << " = relay.nn.max_pool2d("
           << (std::string)II->getOperand(1).first->getName() << ",";
      //  std::cout<< kernel.size() << stride.size() << padding.size();

      bool prev = false;
      if (kernel.size() > 0) {
        pyss << "pool_size=(";
        for (int i = 0; i < kernel.size(); i++) {
          pyss << kernel[i];
          if (i != kernel.size() - 1)
            pyss << ",";
        }
        pyss << ")";
        prev = true;
      }

      if (stride.size() > 0) {
        if (prev)
          pyss << ",";
        pyss << "strides=(";
        for (int i = 0; i < stride.size(); i++) {
          pyss << stride[i];
          if (i != stride.size() - 1)
            pyss << ",";
        }
        pyss << ")";
        prev = true;
      }

      if (padding.size() > 0) {
        if (prev)
          pyss << ",";
        pyss << "padding=(";
        for (int i = 0; i < padding.size(); i++) {
          pyss << padding[i];
          if (i != padding.size() - 1)
            pyss << ",";
        }
        pyss << ")";
        prev = true;
      }

      pyss << ",layout=\"NHWC\")" << std::endl;
      //   std::cout<< (std::string) I.getName() << " = relay.add(" <<
      //   std::endl;

    }

    break;

    case Kinded::Kind::QuantizeInstKind: {

      procCtx.qnn_mode = 1;

      // sum1 = relay.add(a, b)
      auto II = static_cast<QuantizeInst *>(&I);
      auto src = II->getSrc();
      auto dest = II->getDest();

      if (dest->getName() != prev_alloc_name ||
          !prev_alloc_type->isQuantizedType()) {
        std::cout << "[ERROR] can't get quantized type info" << std::endl;
        break;
      }
      pyss << (std::string)dest->getName() << " = relay.qnn.op.quantize("
           << (std::string)src->getName() << ", relay.const("
           << prev_alloc_type->getScale() << ",\"float32\"),relay.const("
           << prev_alloc_type->getOffset() << ",\"int32\"), out_dtype=";
      if (prev_alloc_type->elementType_ == ElemKind::Int8QTy) {
        pyss << "\"int8\"";
      } else if (prev_alloc_type->elementType_ == ElemKind::Int32QTy) {
        pyss << "\"int32\"";
      }
      pyss << ")" << std::endl;

      std::cout << (std::string)dest->getName() << " = relay.qnn.op.quantize("
                << (std::string)src->getName() << ","
                << prev_alloc_type->getScale() << ","
                << prev_alloc_type->getOffset() << ", out_dtype=";
      if (prev_alloc_type->elementType_ == ElemKind::Int8QTy) {
        std::cout << "\"int8\"";
      } else if (prev_alloc_type->elementType_ == ElemKind::Int32QTy) {
        std::cout << "\"int32\"";
      }
      std::cout << ")" << std::endl;

      //   std::cout<< (std::string) I.getName() << " = relay.add(" <<
      //   std::endl;

      // net = relay.nn.avg_pool2d(net,  pool_size=(3, 3), strides=(2, 2),
      // padding=(1, 1),count_include_pad=True)

    } break;
    case Kinded::Kind::DequantizeInstKind: {
      procCtx.qnn_mode = 1;
      // net = relay.nn.avg_pool2d(net,  pool_size=(3, 3), strides=(2, 2),
      // padding=(1, 1),count_include_pad=True)

      auto II = static_cast<DequantizeInst *>(&I);
      auto src = II->getSrc();
      auto dest = II->getDest();

      if (src->getName() != prev_alloc_name ||
          !prev_alloc_type->isQuantizedType()) {
        std::cout << "[ERROR] can't get quantized type info" << std::endl;
        break;
      }

      pyss << (std::string)dest->getName() << " = relay.qnn.op.dequantize("
           << (std::string)src->getName() << ",";

      TypeRef srcT = src->getType();
      pyss << "input_scale=relay.const(" << srcT->getScale()
           << ", \"float32\"),";
      pyss << "input_zero_point=relay.const(" << srcT->getOffset()
           << ", \"int32\")";

      // [TODO] check layout
      // axis: int.  The channel axis for quantization. Default value is -1
      // which corresponds to the last axis.

      pyss << ")" << std::endl;

    } break;
    case Kinded::Kind::AvgPoolInstKind: {
      // net = relay.nn.avg_pool2d(net,  pool_size=(3, 3), strides=(2, 2),
      // padding=(1, 1),count_include_pad=True)

      auto II = static_cast<AvgPoolInst *>(&I);
      auto kernel = II->getKernels();
      auto stride = II->getStrides();
      auto padding = II->getPads();

      if (procCtx.qnn_mode == 1) {
        pyss << (std::string)II->getOperand(1).first->getName()
             << " = relay.cast("
             << (std::string)II->getOperand(1).first->getName() << ",\"int32\")"
             << std::endl;
      }

      pyss << (std::string)II->getOperand(0).first->getName()
           << " = relay.nn.avg_pool2d("
           << (std::string)II->getOperand(1).first->getName() << ",";
      //  std::cout<< kernel.size() << stride.size() << padding.size();

      bool prev = false;
      if (kernel.size() > 0) {
        pyss << "pool_size=(";
        for (int i = 0; i < kernel.size(); i++) {
          pyss << kernel[i];
          if (i != kernel.size() - 1)
            pyss << ",";
        }
        pyss << ")";
        prev = true;
      }

      if (stride.size() > 0) {
        if (prev)
          pyss << ",";
        pyss << "strides=(";
        for (int i = 0; i < stride.size(); i++) {
          pyss << stride[i];
          if (i != stride.size() - 1)
            pyss << ",";
        }
        pyss << ")";
        prev = true;
      }

      if (padding.size() > 0) {
        if (prev)
          pyss << ",";
        pyss << "padding=(";
        for (int i = 0; i < padding.size(); i++) {
          pyss << padding[i];
          if (i != padding.size() - 1)
            pyss << ",";
        }
        pyss << ")";
        prev = true;
      }

      pyss << ",layout=\"NHWC\")" << std::endl;
      //   std::cout<< (std::string) I.getName() << " = relay.add(" <<
      //   std::endl;

      if (procCtx.qnn_mode == 1) {
        pyss << (std::string)II->getOperand(0).first->getName()
             << " = relay.cast(relay.clip("
             << (std::string)II->getOperand(0).first->getName()
             << ",-128,127),\"int8\")" << std::endl;
      }

    }

    break;

    case Kinded::Kind::ElementAddInstKind: {
      // sum1 = relay.add(a, b)
      auto II = static_cast<ElementAddInst *>(&I);
      pyss << (std::string)II->getOperand(0).first->getName() << " = relay.add("
           << (std::string)II->getOperand(1).first->getName() << ","
           << (std::string)II->getOperand(2).first->getName() << ")"
           << std::endl;

      //   std::cout<< (std::string) I.getName() << " = relay.add(" <<
      //   std::endl;

    } break;

    case Kinded::Kind::ElementSubInstKind: {
      auto II = static_cast<ElementSubInst *>(&I);
      pyss << (std::string)II->getOperand(0).first->getName()
           << " = relay.subtract("
           << (std::string)II->getOperand(1).first->getName() << ","
           << (std::string)II->getOperand(2).first->getName() << ")"
           << std::endl;
    } break;

    case Kinded::Kind::ElementDivInstKind: {
      auto II = static_cast<ElementDivInst *>(&I);
      pyss << (std::string)II->getOperand(0).first->getName()
           << " = relay.divide("
           << (std::string)II->getOperand(1).first->getName() << ","
           << (std::string)II->getOperand(2).first->getName() << ")"
           << std::endl;

    }

    break;
    case Kinded::Kind::ElementMulInstKind: {
      auto II = static_cast<ElementMulInst *>(&I);
      pyss << (std::string)II->getOperand(0).first->getName()
           << " = relay.multiply("
           << (std::string)II->getOperand(1).first->getName() << ","
           << (std::string)II->getOperand(2).first->getName() << ")"
           << std::endl;

    }

    break;
    case Kinded::Kind::TransposeInstKind: {
      auto II = static_cast<TransposeInst *>(&I);
      auto src = II->getSrc();
      auto dest = II->getDest();
      auto shuffle = II->getShuffle();

      pyss << (std::string)dest->getName() << " = relay.transpose("
           << (std::string)src->getName() << ", axes=[";
      for (int i = 0; i < shuffle.size(); i++) {
        pyss << shuffle[i];
        if (i != shuffle.size() - 1)
          pyss << ",";
      }
      pyss << "])" << std::endl;

    } break;

    case Kinded::Kind::ConvolutionInstKind: {
      //    y = relay.nn.conv2d(x, w, strides=[2, 2], padding=[1, 1, 1, 1],
      //    kernel_size=[3, 3])
      /*        out = relay.nn.conv2d(
          x,
          kernel,
          kernel_size=k_shape[2:4],
          groups=groups,
          padding=padding,
          strides=strides,
          dilation=dilation,
          channels=channels,
      )
      */
      auto II = static_cast<ConvolutionInst *>(&I);
      // value*
      auto dst = II->getDest();
      auto src = II->getSrc();
      auto filter = II->getFilter();
      auto bias = II->getBias();

      // std::cout<< (std::string)I.toString()<<std::endl;

      pyss << "conv2d_" << ir_idx;
      TypeRef srcT = src->getType();
      if (srcT->elementType_ == ElemKind::Int32QTy ||
          srcT->elementType_ == ElemKind::Int8QTy) {
        procCtx.qnn_mode = 1;
        pyss << " = relay.qnn.op.conv2d(";
        pyss << (std::string)src->getName() << ","
             << (std::string)filter->getName() << ",";
        pyss << "input_zero_point=relay.const(" << srcT->getOffset()
             << ", \"int32\"),";
        pyss << "kernel_zero_point=relay.const("
             << filter->getType()->getOffset() << ", \"int32\"),";
        pyss << "input_scale=relay.const(" << srcT->getScale()
             << ", \"float32\"),";
        pyss << "kernel_scale=relay.const(" << filter->getType()->getScale()
             << ", \"float32\"),";
        pyss << "out_dtype=\"int32\",";

      } else {
        pyss << " = relay.nn.conv2d(";
        pyss << (std::string)src->getName() << ","
             << (std::string)filter->getName() << ",";
      }

      // vec<int>
      auto kernel = II->getKernels();
      auto stride = II->getStrides();
      auto padding = II->getPads();

      auto group = II->getGroup();
      auto dilation = II->getDilation();
      // convolutionLayout
      auto layout = II->getLayout();
      // FusedActivation
      auto fused_act = II->getFusedActivation();
      bool doRelu = fused_act == RELU;

      // bias는 별도 처리해야될 듯 => .nn.bias_add
      // fused_act도 별도 처리해야될듯
      // filter scale, offset 확인

      //  std::cout<< (std::string) dst->getName() << " = relay.nn.conv2d(" <<
      //  (std::string)src->getName() << "," << (std::string)filter->getName()
      //  << ",";

      bool prev = false;
      if (kernel.size() > 0) {
        pyss << "kernel_size=(";
        for (int i = 0; i < kernel.size(); i++) {
          pyss << kernel[i];
          if (i != kernel.size() - 1)
            pyss << ",";
        }
        pyss << ")";
        prev = true;
      }

      if (stride.size() > 0) {
        if (prev)
          pyss << ",";
        pyss << "strides=(";
        for (int i = 0; i < stride.size(); i++) {
          pyss << stride[i];
          if (i != stride.size() - 1)
            pyss << ",";
        }
        pyss << ")";
        prev = true;
      }

      if (padding.size() > 0) {
        if (prev)
          pyss << ",";
        pyss << "padding=(";
        for (int i = 0; i < padding.size(); i++) {
          pyss << padding[i];
          if (i != padding.size() - 1)
            pyss << ",";
        }
        pyss << ")";
        prev = true;
      }

      if (prev)
        pyss << ",";
      pyss << "groups=" << group;

      if (srcT->elementType_ == ElemKind::Int32QTy ||
          srcT->elementType_ == ElemKind::Int8QTy) {
        pyss << ", dilation=(" << dilation << "," << dilation << ")";
      } else {
        pyss << ", dilation=" << dilation;
      }

      // 넘어오는것은 data layout밖에 없는듯
      pyss << ", data_layout=";
      if (layout == ConvolutionLayout::NHWC) {
        pyss << "\"NHWC\""
             << ",kernel_layout=\"OHWI\"";
      } else {
        pyss << "\"NCHW\"";
      }

      //해당 정보로 넘어는 것이 없어보임
      // kernel layout?  default는 OIHW
      // 넘어노는 값은 OHWI 인데 정보가 따로 안보임.
      // OHWI는 처리 못하므로 OIHW로 변경해줌
      TypeRef T = filter->getType();
      pyss << ",channels=" << (int)T->sizes_[0];

      // std::cout << ",kernel_layout=\"OHWI\"";

      // out_layout?

      pyss << ")" << std::endl;

      if (doRelu) {

        pyss << "bias_" << ir_idx << "=relay.nn.bias_add("
             << "conv2d_" << ir_idx << "," << (std::string)bias->getName()
             << ",axis=3)" << std::endl;
        if (srcT->elementType_ == ElemKind::Int32QTy ||
            srcT->elementType_ == ElemKind::Int8QTy) {
          if (dst->getType()->elementType_ == ElemKind::Int8QTy) {

            // pyss << "bias_" << ir_idx << "_clip = relay.clip( bias_" <<
            // ir_idx << ",-128,127)" << std::endl; pyss << (std::string)
            // dst->getName() << "=relay.cast( relay.nn.relu( bias_" << ir_idx
            // <<
            // "_clip ), \"int8\")" << std::endl;

            pyss << (std::string)dst->getName()
                 << "= relay.cast(relay.clip( relay.nn.relu( bias_" << ir_idx
                 << "),-128,127), \"int8\")" << std::endl;
            // pyss << (std::string) dst->getName() << "= relay.nn.relu( bias_"
            // << ir_idx << "_clip )" << std::endl;
          }
        } else {
          pyss << (std::string)dst->getName() << "=relay.nn.relu( bias_"
               << ir_idx << " )" << std::endl;
        }

      } else {
        if (srcT->elementType_ == ElemKind::Int32QTy ||
            srcT->elementType_ == ElemKind::Int8QTy) {
          if (dst->getType()->elementType_ == ElemKind::Int8QTy) {
            //[TODO] clip
            pyss << "bias_added_" << ir_idx << "=relay.nn.bias_add("
                 << "conv2d_" << ir_idx << "," << (std::string)bias->getName()
                 << ",axis=3)" << std::endl;
            pyss << (std::string)dst->getName()
                 << "= relay.cast(relay.clip( bias_added_" << ir_idx
                 << ",-128,127), \"int8\")" << std::endl;
          }
        } else {
          pyss << (std::string)dst->getName() << "=relay.nn.bias_add("
               << "conv2d_" << ir_idx << "," << (std::string)bias->getName()
               << ",axis=3)" << std::endl;
        }
      }

    }

    break;

    case Kinded::Kind::FullyConnectedInstKind: {
      //        std::cout<< (std::string) I.getName() << " = " << std::endl;
      auto II = static_cast<FullyConnectedInst *>(&I);

      auto dest = II->getDest();
      auto src = II->getSrc();
      auto weight = II->getWeights();
      auto bias = II->getBias();

      pyss << "fc_" << ir_idx;

      TypeRef srcT = src->getType();
      if (srcT->elementType_ == ElemKind::Int32QTy ||
          srcT->elementType_ == ElemKind::Int8QTy) {
        procCtx.qnn_mode = 1;
        pyss << " = relay.qnn.op.dense(";
      } else {
        pyss << " = relay.nn.dense(";
      }

      pyss << (std::string)src->getName() << ",_op.transpose("
           << (std::string)weight->getName() << ", [1, 0])";

      if (srcT->elementType_ == ElemKind::Int32QTy ||
          srcT->elementType_ == ElemKind::Int8QTy) {
        TypeRef wT = weight->getType();
        pyss << ",input_zero_point=relay.const(" << srcT->getOffset()
             << ", \"int32\"),";
        pyss << "kernel_zero_point=relay.const("
             << weight->getType()->getOffset() << ", \"int32\"),";
        pyss << "input_scale=relay.const(" << srcT->getScale()
             << ", \"float32\"),";
        pyss << "kernel_scale=relay.const(" << weight->getType()->getScale()
             << ", \"float32\"),";
        pyss << "units= " << wT->sizes_[1];
      }
      pyss << ")" << std::endl;

      pyss << (std::string)dest->getName() << " = relay.nn.bias_add(fc_"
           << ir_idx << ", " << (std::string)bias->getName() << ")"
           << std::endl;

    }

    break;

    case Kinded::Kind::TensorViewInstKind: {
      // std::cout<< "TENSORVIEW " << I.toString() << " = " << std::endl;
      auto II = static_cast<TensorViewInst *>(&I);
      auto T = II->getTy();

      pyss << "new_shape = []" << std::endl;
      for (int i = 0; i < sizeof(T->sizes_) / sizeof(dim_t); i++) {
        if (T->sizes_[i] == 0)
          break;
        pyss << "new_shape.append(" << (int)T->sizes_[i] << ")" << std::endl;
      }
      pyss << (std::string)II->getName() << " = _op.reshape("
           << (std::string)II->getSrc()->getName() << ",new_shape)"
           << std::endl;

      if (output_temp_name == (std::string)II->getSrc()->getName()) {

        output_temp_var->setName((std::string)II->getName());
        output_temp_var->setType(T);
        addSymbolEntryGenBundle(output_temp_var, &procCtx, 2);
      }
    } break;

    case Kinded::Kind::CopyInstKind: {
      auto II = static_cast<CopyInst *>(&I);
      auto src = II->getSrc();
      auto dest = II->getDest();
      pyss << (std::string)dest->getName() << " = "
           << (std::string)src->getName() << std::endl;
      break;
    }
    case Kinded::Kind::WeightVarKind:
      pyss << "WeightVarKind " << I.toString() << " = " << std::endl;
      break;

    case Kinded::Kind::LocalResponseNormalizationInstKind: {
      //  auto *A = new LocalResponseNormalizationInst(uniqueName(name), Dest,
      //  Src, Scale, HalfWindowSize, Alpha, Beta, K);
      std::cout << (std::string)I.toString() << std::endl;
      auto II = static_cast<LocalResponseNormalizationInst *>(&I);

      auto dest = II->getDest();
      auto src = II->getSrc();
      auto scale = II->getScale();                 // Value
      auto alpha = II->getAlpha();                 // f
      auto beta = II->getBeta();                   // f
      auto bias = II->getK();                      // f
      auto size = II->getHalfWindowSize() * 2 + 1; // unsigned

      // axis = channel. default = 1 for NCHW
      std::cout << (std::string)scale->toString() << std::endl;

      std::cout << (std::string)dest->getName() << " = relay.nn.lrn("
                << (std::string)src->getName() << ", size=" << size
                << ", bias=" << bias << ",alpha=" << alpha << ",beta=" << beta
                << ")" << std::endl;
      pyss << (std::string)dest->getName() << " = relay.nn.lrn("
           << (std::string)src->getName() << ", size=" << size
           << ", bias=" << bias << ",alpha=" << alpha << ",beta=" << beta << ")"
           << std::endl;

    }

    break;

    case Kinded::Kind::TouchInstKind:
      // do nothing
      break;

    case Kinded::Kind::InsertTensorInstKind: {
      // InsertTensorInst is not matched with relay.concat perfectly. assume
      // that insertTensor is used only for concat(count=1)
      auto II = static_cast<InsertTensorInst *>(&I);

      auto dest = II->getDest();
      auto src = II->getSrc();
      auto count = II->getCount();     // unsigned_t
      auto axis = II->getAxis();       // unsigned_t
      auto offsets = II->getOffsets(); // arrayref<dim_t>

      auto srcT = src->getType();
      auto destT = dest->getType();

      if (count != 1) {
        std::cout << "[ERROR] cannot handle with relay.concat" << std::endl;
      }

      if (destT->sizes_[axis] < offsets[axis] + srcT->sizes_[axis]) {
        std::cout << "[ERROR] invalid size" << std::endl;
      }

      std::cout << "d:" << destT->sizes_[axis] << "off:" << offsets[axis]
                << "src:" << srcT->sizes_[axis] << std::endl;

      if (offsets[axis] == 0) {
        std::cout << (std::string)dest->getName() << " = "
                  << (std::string)src->getName() << std::endl;
        pyss << (std::string)dest->getName() << " = "
             << (std::string)src->getName() << std::endl;
      } else {

        std::cout << (std::string)dest->getName() << " = relay.concatenate( ["
                  << (std::string)dest->getName() << " , "
                  << (std::string)src->getName() << "]," << axis << ")"
                  << std::endl;
        pyss << (std::string)dest->getName() << " = relay.concatenate( ["
             << (std::string)dest->getName() << " , "
             << (std::string)src->getName() << "]," << axis << ")" << std::endl;
      }

      std::cout << "\n" << I.toString() << std::endl;

    }
    // do nothing
    break;

    case Kinded::Kind::SoftMaxInstKind: {
      std::cout << (std::string)I.toString() << std::endl;
      auto II = static_cast<SoftMaxInst *>(&I);

      auto dest = II->getDest();
      auto src = II->getSrc();

      pyss << (std::string)dest->getName() << " = relay.nn.softmax("
           << (std::string)src->getName() << ")" << std::endl;

    }

    break;

    default:
      std::cout << "\n\n[ERROR] !! not added " << I.getKindName() << std::endl;
    }

    /*
        if( strcmp(I.getKindName(),"relu")==0) {
        } else if (strcmp(I.getKindName(),"allocactivation")==0) {
        } else if (strcmp(I.getKindName(),"elementadd")==0) {
        } else if (strcmp(I.getKindName(),"elementsub")==0) {
          } else if (strcmp(I.getKindName(),"elementdiv")==0) {
            } else if (strcmp(I.getKindName(),"convolution")==0) {
              } else if (strcmp(I.getKindName(),"deallocactivation")==0) {
                } else if (strcmp(I.getKindName(),"maxpool")==0) {
                  } else if (strcmp(I.getKindName(),"avgpool")==0) {
                  } else if (strcmp(I.getKindName(),"tensorview")==0) {
                    } else if (strcmp(I.getKindName(),"fullyconnected")==0) {
                    } else
                    {
                      std::cout << "\n\n!! not added " <<
       I.getKindName()<<std::endl;
                    }
      */
    // std::cout<<"SAVE INSTRUCTION FOR "<<(std::string) I.getName() << " ||| "
    // <<I.getType()<< " --- " << I.toString()<<std::endl;
    // std::cout<<I.getKind()<<I.getSrc()<<I.getDest()<<I.getFilter()<<std::endl;

    ir_idx++;
  }

  if (procCtx.output_names.size() > 1) {
    pyss << "\n\noutput_tuples = relay.Tuple([";
    for (int i = 0; i < procCtx.output_names.size(); i++) {
      pyss << procCtx.output_names[i] << ",";
    }
    pyss << "])\n\nfunc = "
            "relay.Function(relay.analysis.free_vars(output_tuples), "
            "output_tuples)";
  } else if (procCtx.output_names.size() == 1) {
    pyss << "\n\nfunc = relay.Function(relay.analysis.free_vars("
         << procCtx.output_names[0] << "), " << procCtx.output_names[0] << ")";
  } else {

    //  std::cout << pyss.str();
    std::cout << "NO OUTPUT variables. check mutable variables!!" << std::endl;
    assert(procCtx.output_names.size() >= 1);
  }

  py.append(pyss.str());

  std::cout << "--relay-target = " << target_ << std::endl;
  std::cout << "--relay-opt-level = " << opt_level_ << std::endl;

  finalizeCtx(saveCtx, outputDir, bundleName, mainEntryName, procCtx, target_,
              target_host_, export_option_, opt_level_, required_pass_,
              disabled_pass_, debug_mode_);

  // write genCode.py
  std::string pyFileName = relay_mkpath;
  // std::cout << pyFileName.c_str();
  pyFileName.append("/genCode.py");
  std::ofstream wfos(pyFileName.c_str(), std::ios::out);
  wfos.write(py.c_str(), py.size());
  wfos.write(inc.c_str(), inc.size());
  wfos.write(cpp.c_str(), cpp.size());
  wfos.close();

  std::string mkFileName = module_mkpath;
  mkFileName.append("/Makefile");
  std::ofstream mfos(mkFileName.c_str(), std::ios::out);
  mfos.write(mk.c_str(), mk.size());
  mfos.close();

  return;
}
