#include "Relay.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/IR/IRUtils.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>

using namespace glow;


// quoted for c++11
template<typename CharT>
struct out_quoted
{
    const CharT* s;
    CharT delim;
    CharT escape;
};

template<typename CharT>
out_quoted<CharT> quoted(const CharT* s, CharT delim = CharT('"'), CharT escape = CharT('\\')) { return { s, delim, escape }; }

template<typename CharT>
out_quoted<CharT> quoted(const std::basic_string<CharT>& s, CharT delim = CharT('"'), CharT escape = CharT('\\')) { return { s.c_str(), delim, escape }; }

template<typename CharT>
std::ostream& operator<<(std::ostream& os, const out_quoted<CharT>& q)
{
    os << q.delim;
    for(const CharT* p = q.s; *p; p++)
    {
        if(*p == q.delim || *p == q.escape) os << q.escape << *p;
        else os << *p;
    }
    return os << q.delim;
}

// header wrapping
std::string hh(std::string org) {
   std::stringstream ss11;
   ss11 << "header.append(" << quoted(org) << ")\n";
  return ss11.str();
}

// cpp code wrapping
std::string cc(std::string org) {
   std::stringstream ss11;
   ss11 << "cpp.append(" << quoted(org) << ")\n";
  return ss11.str();
}




struct SaveCtx{
  std::string pyRelayCode;       // tvm relay 처리를 위한 코드. 
  std::string partHeaderGen;    // gencode에서 part.h 생성하는 부분. python코드
  std::string partCppGen;    // gencode에서 part.cpp 생성하는 부분. python코드
  std::string partMakeGen;  // module 생성. Makefile
};


// context저장을 위해 vtasave context 가져옴.
// 사실상 weight는 object에 tvm형태로 저장되기 때문에 크게 의미는 없음.
// 직접 로드해서 binding해서 사용하는 경우만 의미가 있을듯

class RelaySaveContext{
public:
  RelaySaveContext()
  {
    constantWeightVarsMemSize = 0;
    mutableWeightVarsMemSize = 0;
    var_declare = "";
    
  }
  std::vector<struct SymbolTableEntry> *getSymbols(){
    return &syms;
  }
  std::vector<struct SymbolTableEntry> *getConstantSymbols(){
    return &csyms;
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
  void setGenCodeFileStream(std::ofstream *fos)
  {
    wfos= fos;
  }
  std::ofstream* getGenCodeFileStream(){
    return wfos;
  }
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

  std::string var_declare;  // variable in main function
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string> input_DLTensor_names;
  std::vector<std::string> output_DLTensor_names;


private:

  std::vector<struct SymbolTableEntry> syms;    //symbol
  std::vector<struct SymbolTableEntry> csyms;   //constant
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

SymbolTableEntry addSymbolEntry(WeightVar* wgt, RelaySaveContext *ctx){
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

// mutable
// inout = 1 = input, 2=output
SymbolTableEntry addSymbolEntryGenBundle(WeightVar* wgt, RelaySaveContext *ctx, int inout){
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

  syms->push_back({wgt->getName().data(), ctx->getMMemSize(), wgt->size(), '1', wgt});
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
  if(inout == 1) {

      ctx->input_names.push_back(ste.name);
      ctx->input_DLTensor_names.push_back("input_" + std::to_string(ste.offset));
      
      ctx->var_declare.append(cc("DLTensor input_" + std::to_string(ste.offset) + ";"));

      ctx->var_declare.append("cpp.append(\"");
      ctx->var_declare.append("std::vector<int64_t> input_shape_" + std::to_string(ste.offset) + " = {");

      if(T->elementType_ == ElemKind::FloatTy) {
          for(int i=0;i<T->numSizes_;i++) {
             ctx->var_declare.append( std::to_string((int)T->sizes_[i]) );
             ctx->var_declare.append( (i!=T->numSizes_-1) ? "," : " "); 
          }
      }
      
      ctx->var_declare.append("};");
      ctx->var_declare.append("\")\n");

      ctx->var_declare.append(cc("input_" + std::to_string(ste.offset) + ".data = " + ste.name + ";"));
      ctx->var_declare.append(cc("input_" + std::to_string(ste.offset) + ".device = DLDevice{kDLCPU, 0};"));
      ctx->var_declare.append(cc("input_" + std::to_string(ste.offset) + ".ndim = " + std::to_string(T->numSizes_) + ";"));
      ctx->var_declare.append(cc("input_" + std::to_string(ste.offset) + ".dtype = DLDataType{kDLFloat, 32, 1};"));
      ctx->var_declare.append(cc("input_" + std::to_string(ste.offset) + ".shape = input_shape_" + std::to_string(ste.offset) + ".data();"));
      ctx->var_declare.append(cc("input_" + std::to_string(ste.offset) + ".strides = nullptr;"));
      ctx->var_declare.append(cc("input_" + std::to_string(ste.offset) + ".byte_offset = 0;"));

      

  }
  else {
    
    ctx->output_names.push_back(ste.name);
    ctx->output_DLTensor_names.push_back("output_" + std::to_string(ste.offset));

    ctx->var_declare.append(cc("DLTensor output_" + std::to_string(ste.offset) + ";"));
    ctx->var_declare.append("cpp.append(\"");
    ctx->var_declare.append("std::vector<int64_t> output_shape_" + std::to_string(ste.offset) + " = {");
    if(T->elementType_ == ElemKind::FloatTy) {
        for(int i=0;i<T->numSizes_;i++) {
             ctx->var_declare.append( std::to_string((int)T->sizes_[i]) );
             ctx->var_declare.append( (i!=T->numSizes_-1) ? "," : " "); 
        }
    }
    ctx->var_declare.append("};");
    ctx->var_declare.append("\");\n");

    ctx->var_declare.append(cc("output_" + std::to_string(ste.offset) + ".data = " + (std::string)ste.name + ";"));
    ctx->var_declare.append(cc("output_" + std::to_string(ste.offset) + ".device = DLDevice{kDLCPU, 0};"));
    ctx->var_declare.append(cc("output_" + std::to_string(ste.offset) + ".ndim = "+ std::to_string(T->numSizes_) + ";"));
    ctx->var_declare.append(cc("output_" + std::to_string(ste.offset) + ".dtype = DLDataType{kDLFloat, 32, 1};"));
    ctx->var_declare.append(cc("output_" + std::to_string(ste.offset) + ".shape = output_shape_" + std::to_string(ste.offset) + ".data();"));
    ctx->var_declare.append(cc("output_" + std::to_string(ste.offset) + ".strides = nullptr;"));
    ctx->var_declare.append(cc("output_" + std::to_string(ste.offset) + ".byte_offset = 0;"));

  }


  return ste;
}

SymbolTableEntry addConstantSymbolEntry(Value* val, RelaySaveContext *ctx){
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




//typedef struct {
//  
//  void* data;
//
//  int device_type;  // 1=CPU
//  int devicc_id;   // 0
//  
//  int ndim;
//  /*! \brief The data type of the pointer*/
//  uint8_t dtype;  // 0=signed int, 1=uint, 2=float, 3=opaque, 4=float16,5=complex
//  uint8_t bits; // num of bits 8,16,32
//  uint16_t lanes; // used for vector type
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





void initCtx(struct SaveCtx &Ctx,
                    llvm::StringRef bundleName,
                    llvm::StringRef mainEntryName){

  auto& inc = Ctx.partHeaderGen;
  auto& cpp = Ctx.partCppGen;
  auto& py = Ctx.pyRelayCode;
  auto& mk = Ctx.partMakeGen;

  inc = "";
  cpp = "";
  py = "";
  mk="";


  // part.h
  
  inc.append("\n\n## output header");
  inc.append("\nheader = []\n");
  inc.append(hh("// Bundle API auto-generated header file. Do not edit!"));
  inc.append(hh("// tvm nest-c Tools version: 2021-07-07"));

  inc.append(hh("#ifndef _GLOW_BUNDLE_" + (std::string)bundleName + "_H"));
  inc.append(hh("#define _GLOW_BUNDLE_" + (std::string)bundleName + "_H"));
  
  inc.append(hh("#include <stdint.h>"));
  inc.append(hh("// ---------------------------------------------------------------"));
  inc.append(hh("//                       Common definitions"));
  inc.append(hh("// ---------------------------------------------------------------"));
  inc.append(hh("#ifndef _GLOW_BUNDLE_COMMON_DEFS"));
  inc.append(hh("#define _GLOW_BUNDLE_COMMON_DEFS"));
  
  inc.append(hh("// Glow bundle error code for correct execution."));
  inc.append(hh("#define GLOW_SUCCESS 0"));

  inc.append(hh("// Type describing a symbol table entry of a generated bundle."));
  inc.append(hh("struct SymbolTableEntry {"));
  inc.append(hh("\t// Name of a variable."));
  inc.append(hh("\tconst char *name;"));
  inc.append(hh("\t// Offset of the variable inside the memory area."));
  inc.append(hh("\tuint64_t offset;"));
  inc.append(hh("\t// The number of elements inside this variable."));
  inc.append(hh("\tuint64_t size;"));
  inc.append(hh("\t// Variable kind: 1 if it is a mutable variable, 0 otherwise."));
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


  //part.cpp
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


  //
  py.append("import numpy as np\nimport tvm\nfrom tvm import te, runtime\nimport tvm.relay as relay\n"
        "from tvm.relay.frontend.common import infer_type\nfrom tvm.relay.testing import check_grad, run_infer_type, run_opt_pass, _np_randn_from_type\n"
        "import onnx\nfrom tvm.relay import op as _op\n\ndef load_wgt(filename, shape):\n\tf=open(filename,\"rb\")\n\td=f.read()\n\treturn np.frombuffer(d, dtype=np.float32).reshape(shape)\n\n");


}

void finalizeCtx(struct SaveCtx &Ctx,  llvm::StringRef outputDir, llvm::StringRef bundleName,
                             llvm::StringRef mainEntryName, RelaySaveContext &procCtx) {

  auto& inc = Ctx.partHeaderGen;
  auto& cpp = Ctx.partCppGen;
  auto& py = Ctx.pyRelayCode;
  auto& mk = Ctx.partMakeGen;



  //#input에 대해서는 이름을 알 수 있지만
  //#output은 별도 name을 유지하지 않음. index로 가져옴.
  //#name은 그냥 output으로 통일
  //#total data size는 의미가 좀 다름. serialized된 것이라 순수 data만 나타낼 수 없음.
  //#따로 계산해서 가능하긴 한데 arch에 따라서 DLTensor가 달라지면 다를 수 있음
  inc.append(hh("// ---------------------------------------------------------------"));
  inc.append(hh("//                          Bundle API"));
  inc.append(hh("// ---------------------------------------------------------------"));
/*
  inc.append(hh("// Model name: "));
  inc.append(hh(bundleName));
  inc.append(hh("// Total data size: " + std::to_string(procCtx.getMMemSize())));
  inc.append(hh("// Placeholders:"));
  inc.append(hh("//"));

  for(auto ste : *procCtx.getSymbols()) {
    auto wgt = ste.wgt;
    assert(wgt);
    inc.append(hh("//   Name: "));
    inc.append(hh(ste.name ));
    //type
    inc.append(hh("//   Size: " + std::to_string(wgt->size()) + "(elements)"));
    inc.append(hh("//   Size: " + std::to_string(wgt->getSizeInBytes()) + "(bytes)"));
    inc.append(hh("//   Offset: " + std::to_string(ste.offset) + "(bytes)"));

  };

  inc.append(hh("// NOTE: Placeholders are allocated within the \"mutableWeight\")"));
  inc.append(hh("// buffer and are identified using an offset relative to base.)"));
  inc.append(hh("// ---------------------------------------------------------------"));
*/
  inc.append(hh("#ifdef __cplusplus"));
  inc.append(hh("extern \"C\" {"));
  inc.append(hh("#endif"));


  inc.append(hh("// Bundle memory configuration (memory layout)."));
  inc.append(hh("extern BundleConfig " + (std::string)bundleName + "_config;"));

  inc.append(hh("// Bundle entry point (inference function). Returns 0"));
  inc.append(hh("// for correct execution or some error code otherwise."));
  inc.append(hh("int " + (std::string)mainEntryName + "(uint8_t *constantWeight, uint8_t *mutableWeight, uint8_t *activations);"));
  inc.append(hh("#ifdef __cplusplus"));
  inc.append(hh("}"));
  inc.append(hh("#endif"));
  inc.append(hh("#endif"));


    inc.append("with open(\"" +  (std::string)outputDir + "/module/" + (std::string)bundleName + ".h\",\"w\") as f_h:\n");
    inc.append("  for item in header:\n");
    inc.append("    f_h.write(\"%s\\n\" % item)\n");


  py.append("\ndesired_layouts = { \"nn.conv2d\": [\"NCHW\", \"default\"]}  \
             \nseq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(), \
             \n\t\trelay.transform.ConvertLayout(desired_layouts)]) \
             \nwith tvm.transform.PassContext(opt_level=3): \
            \n\t\trelay_mod = seq(tvm.IRModule.from_expr(func)) \
            \n\n#target=\"llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon\" \
            \ntarget=\"llvm\" \
        \nwith tvm.transform.PassContext(opt_level=0): \
        \n\t\tlib = relay.build(relay_mod, target,params=params) \
    \n#cross_compile = 'aarch64-linux-gnu-c++' \
    \n#lib.export_library(\"output.so\", cc=cross_compile)    \
    \nlib.export_library(\"" + (std::string)outputDir + "/module/" + (std::string)bundleName + "_tvm.so\")        \
    \n# = lib.get_params()\
    \n#    for item in b:\
    \n#        print(item)\
    \n#        print(item.nbytes)");

    // Generate SymbolTable struct
    std::string symbolTable;
    symbolTable = "SymbolTableEntry symbolTableEntry_";
    symbolTable.append(bundleName);
    symbolTable.append("[");
    symbolTable.append(std::to_string(procCtx.getSymbols()->size()));
    symbolTable.append("]={");
    for(auto sym : *(procCtx.getSymbols())){
      symbolTable.append("{\"");
      symbolTable.append(sym.name);
      symbolTable.append("\",");
      symbolTable.append(std::to_string(sym.offset));
      symbolTable.append(",");
      symbolTable.append(std::to_string(sym.size));
      symbolTable.append(",'1'},");
    }
    symbolTable.erase(symbolTable.size()-1,1);
    symbolTable.append("};");

    //std::cout << symbolTable.c_str();

    cpp.append(cc(symbolTable));

    //cpp
     cpp.append(cc("struct BundleConfig " +  (std::string)bundleName + "_config = {"));
     cpp.append(cc("   " +  std::to_string(procCtx.getCMemSize()) + ","));
     cpp.append(cc("   " +  std::to_string(procCtx.getMMemSize()) + ","));
     cpp.append(cc("   0, 64, "));   // alignment always 64?
     cpp.append(cc("   " +  std::to_string(procCtx.getSymbols()->size()) + ","));
     cpp.append(cc("   symbolTableEntry_" + (std::string)bundleName));
     cpp.append(cc("};"));

     //function
     cpp.append(cc("int " + (std::string)bundleName + "(uint8_t* constantWeight, uint8_t *mutableWeight, uint8_t *activations) {" ));
     cpp.append(cc("" ));
     cpp.append(procCtx.var_declare); // mutable variables
     cpp.append(cc("  DLDevice dev{kDLCPU, 0};"))  ;
     cpp.append(cc(" tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(\"" +  (std::string)outputDir + "/module/" + (std::string)bundleName + "_tvm.so\");"))  ;
  
     // create the graph executor module
     cpp.append(cc(" tvm::runtime::Module gmod = mod_factory.GetFunction(\"default\")(dev);"));
     cpp.append(cc(" tvm::runtime::PackedFunc set_input = gmod.GetFunction(\"set_input\");"));
    cpp.append(cc(" tvm::runtime::PackedFunc get_output = gmod.GetFunction(\"get_output\");"));
    cpp.append(cc(" tvm::runtime::PackedFunc run = gmod.GetFunction(\"run\");"));

    // set the right input
    for(int i=0;i<procCtx.input_names.size();i++) {
      cpp.append(cc(" set_input(\"" + procCtx.input_names[i] + "\", &" + procCtx.input_DLTensor_names[i]+ ");"));
    }
    
  
    // run the code
    cpp.append(cc(" run();"));
  
    // get the output
    for(int i=0;i<procCtx.output_DLTensor_names.size();i++) {    
      cpp.append(cc("  get_output(" + std::to_string(i) + ", &" + procCtx.output_DLTensor_names[i] + ");"));
    }

    cpp.append(cc("  return 0;"));
    cpp.append(cc("}"));

    cpp.append("with open(\"" +  (std::string)outputDir + "/module/" + (std::string)bundleName + ".cpp\",\"w\") as f_cpp:\n");
    cpp.append("  for item in cpp:\n");
    cpp.append("    f_cpp.write(\"%s\\n\" % item)\n");



    mk.append("DMLC_CORE=${TVM_HOME}/3rdparty/dmlc-core\n\n");
    mk.append("PKG_CFLAGS = -std=c++14 -g -fPIC\\\n");
    mk.append("\t-I${TVM_HOME}/include\\\n");
    mk.append("\t-I${DMLC_CORE}/include\\\n");
    mk.append("\t-I${TVM_HOME}/3rdparty/dlpack/include\\\n");
    mk.append("\t-DDMLC_USE_LOGGING_LIBRARY=\\<tvm/runtime/logging.h\\>\n\n");
    mk.append("PKG_LDFLAGS = -L${TVM_HOME}/build -ldl -pthread \n\n");
    mk.append(".PHONY: clean all\n\n");
    mk.append("all: " + (std::string)bundleName + ".o\n\n");
    mk.append((std::string)bundleName + ".o: " + (std::string)bundleName + ".cpp " + (std::string)bundleName + "_tvm.so\n");
	  mk.append("\t@mkdir -p $(@D)\n");
    mk.append("\t$(CXX) $(PKG_CFLAGS) -c -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)\n");
    mk.append("clean:\n");
    mk.append("\trm -rf lib\n");




}


static int getInOut(const Value *V) {
  
  for (const auto &U : ValueUses(V)) {
    Instruction *user = U.get();

    // Ignore deallocs.
    if( user->getKind() == Kinded::Kind::DeallocActivationInstKind) {
      continue;
    }

    auto op = U.getOperand();

    // Ignore the readers.
    if (op.second == OperandKind::In) {
      return 1;
    } else if(op.second == OperandKind::Out) {
      return 2;
    }
  }

  return 0;
}


void Relay::save(Function *F, llvm::StringRef outputDir,
               llvm::StringRef bundleName,
               llvm::StringRef mainEntryName) const {


  struct SaveCtx saveCtx;   //ctx for save files
  RelaySaveContext procCtx; //ctx for processing data
  std::string variableString="";
  
  std::vector<std::string> params_name;

  int status;
  std::string mkpath = (std::string)outputDir + "/relay";
  status = mkdir(mkpath.c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  mkpath = (std::string)outputDir + "/module";
  status = mkdir(mkpath.c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  
  initCtx(saveCtx,bundleName,mainEntryName);
  std::stringstream pyss;

  auto& inc = saveCtx.partHeaderGen;
  auto& cpp = saveCtx.partCppGen;
  auto& py = saveCtx.pyRelayCode;
  auto& mk = saveCtx.partMakeGen;



  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());

//  for (const auto &VV : IR->getVariableMap()) {
// }

  
  for (const auto &W : IR->getWeights()) {
    
      //debug
   // std::cout << W->getKindName() << "] " << W->toString() << std::endl;

    // weight 별도 저장
    // tvm에서 활용하는 params 형태로 변경가능한게 좋음가 좋음
    // int32 갯수
    // {name, }
        TypeRef T = W->getType();
      
        if(T->elementType_ == ElemKind::FloatTy) {
          pyss<< (std::string) W->getName() << " = relay.var(\"" << (std::string) W->getName() << "\", shape=(";
          for(int i=0;i<T->numSizes_;i++) {
             pyss<< (int)T->sizes_[i] << ((i!=T->numSizes_-1) ? "," : ""); 
          }
          if(T->numSizes_ == 1) pyss<< ",";
          pyss<< "),dtype=\"float32\")"<<std::endl;
         
        }  else {
           pyss<< "!TYPE:"<<(int)T->elementType_ << std::endl;
        } 


        //save weight
        if(W->getMutability() == WeightVar::MutabilityKind::Constant) {

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
          auto handle = tensor->getHandle<float>();
          
          std::ofstream fos( (std::string) outputDir + "/relay/" + (std::string)W->getName(), std::ios::out);        
          for (size_t i = 0, e = handle.size(); i < e; i++) {
            auto data = handle.raw(i);
            fos.write((const char *)&data, 4);
          }

          addConstantSymbolEntry(W, &procCtx);

          //load in python
          pyss<< "W_" << (std::string) W->getName() << "= load_wgt(" << "\"" << (std::string) outputDir << "/relay/" << (std::string) W->getName() << "\",(";
          for(int i=0;i<T->numSizes_;i++) {
             pyss<< (int)T->sizes_[i] << ((i!=T->numSizes_-1) ? "," : ""); 
          }
          if(T->numSizes_ == 1) pyss<< ",";
          pyss<< "))"<<std::endl;

          params_name.push_back(W->getName() );

        } else  // mutable
        {
          

          int type = getInOut(W);
          if (type == 2) {
            std::cout << "OUT // " << W->getKindName() << " :  " << W->toString() << std::endl;
          } else if (type==1) {
            std::cout << "IN // " << W->getKindName() << " :  " << W->toString() << std::endl;
          }
            
            
            addSymbolEntryGenBundle(W,&procCtx,type);
        }

    
  

  }

  pyss<<"params = {";
  for(auto s : params_name) {
    pyss<< "\"" << s << "\":W_" << s << ",";
  }
  pyss<<"}"<<std::endl;

  int ir_idx=0;
  for (auto &I : IR->getInstrs()) {

    //std::cout<< (std::string)I.toString()<<std::endl;
            

    switch(I.getKind()) {
      case Kinded::Kind::DeallocActivationInstKind:
      {
        auto II =  static_cast<DeallocActivationInst*>(&I);
       //std::cout<< "del " << (std::string)II->getSrc()->getName()<<std::endl;
       //DO nothing
      }
        break;

      case Kinded::Kind::ReluInstKind:
      {
        //y2 = relay.nn.relu(y2)
         auto II =  static_cast<ReluInst*>(&I);
         
         pyss<< (std::string) II->getOperand(0).first->getName() << " = relay.nn.relu(" <<  (std::string) II->getOperand(1).first->getName() << ")"<<std::endl;
         //std::cout<< (std::string)I.toString()<<std::endl;

      }
        break;

      case Kinded::Kind::AllocActivationInstKind:
      {
        /*
        //inputs = {"input": ((1, 224, 224, 3), "uint8")}
        //x = relay.var("x", shape=(1, 3, 224, 224))
//  data = relay.var("data", relay.TensorType(d_shape, "float32"))
        //weight = relay.var("weight", relay.TensorType(w_shape, "float32"))        
        auto II =  static_cast<AllocActivationInst*>(&I);
        TypeRef T = II->getTy();
        //std::cout<< (int)T->elementType_ << std::endl;
        //std::cout<< (int)T->numSizes_ << std::endl;

        if(T->elementType_ == ElemKind::FloatTy) {
          std::cout<< (std::string) I.getName() << " = relay.var(\"" << (std::string) I.getName() << "\", shape=( ";
          for(int i=0;i<T->numSizes_;i++) {
             std::cout<< (int)T->sizes_[i] << ((i!=T->numSizes_-1) ? "," : ""); 
          }
          if(T->numSizes_ == 1) std::cout<< ",";
          std::cout<< "),dtype=\"float32\")"<<std::endl;
           
        }
        */
        
      }
        break;

      case Kinded::Kind::MaxPoolInstKind:
      {
        // net = relay.nn.max_pool2d(net,  pool_size=(3, 3), strides=(2, 2), padding=(1, 1))

        auto II =  static_cast<MaxPoolInst*>(&I);
        auto kernel = II->getKernels();
        auto stride = II->getStrides();
        auto padding = II->getPads();

        pyss<< (std::string) II->getOperand(0).first->getName() << " = relay.nn.max_pool2d(" <<  (std::string)II->getOperand(1).first->getName() << ",";
      //  std::cout<< kernel.size() << stride.size() << padding.size();
     
        bool prev=false;
        if(kernel.size() > 0) {
         pyss << "pool_size=(";
          for(int i=0;i<kernel.size();i++) {
            pyss << kernel[i];
              if(i!=kernel.size()-1) pyss<< ",";
          } 
          pyss << ")";
          prev=true;
        }

        if(stride.size() > 0) {
          if(prev) pyss << ",";
          pyss << "strides=(";
          for(int i=0;i<stride.size();i++) {
            pyss << stride[i];
            if(i!=stride.size()-1) pyss << ",";
          } 
          pyss << ")";
          prev=true;
        }        

        if(padding.size() > 0) {
          if(prev) pyss<< ",";
          pyss << "padding=(";
          for(int i=0;i<padding.size();i++) {
            pyss << padding[i];
            if(i!=padding.size()-1) pyss << ",";
          } 
          pyss << ")";
          prev=true;
        }               

        pyss<< ",layout=\"NHWC\")"<<std::endl;
     //   std::cout<< (std::string) I.getName() << " = relay.add(" << std::endl;

      }

       
        break;

      case Kinded::Kind::AvgPoolInstKind:
      {
        // net = relay.nn.avg_pool2d(net,  pool_size=(3, 3), strides=(2, 2), padding=(1, 1),count_include_pad=True)

        auto II =  static_cast<AvgPoolInst*>(&I);
        auto kernel = II->getKernels();
        auto stride = II->getStrides();
        auto padding = II->getPads();

        pyss<< (std::string) II->getOperand(0).first->getName() << " = relay.nn.avg_pool2d(" <<  (std::string)II->getOperand(1).first->getName() << ",";
      //  std::cout<< kernel.size() << stride.size() << padding.size();
     
        bool prev=false;
        if(kernel.size() > 0) {
          pyss << "pool_size=(";
          for(int i=0;i<kernel.size();i++) {
            pyss << kernel[i];
              if(i!=kernel.size()-1) pyss << ",";
          } 
          pyss << ")";
          prev=true;
        }

        if(stride.size() > 0) {
          if(prev) pyss << ",";
         pyss << "strides=(";
          for(int i=0;i<stride.size();i++) {
            pyss << stride[i];
            if(i!=stride.size()-1) pyss << ",";
          } 
          pyss << ")";
          prev=true;
        }        

        if(padding.size() > 0) {
          if(prev) pyss<< ",";
          pyss << "padding=(";
          for(int i=0;i<padding.size();i++) {
            pyss << padding[i];
            if(i!=padding.size()-1) pyss << ",";
          } 
          pyss << ")";
          prev=true;
        }               

        pyss<< ",layout=\"NHWC\")"<<std::endl;
     //   std::cout<< (std::string) I.getName() << " = relay.add(" << std::endl;

      }

    
        break;

      case Kinded::Kind::ElementAddInstKind:
      {
      // sum1 = relay.add(a, b)
        auto II =  static_cast<ElementAddInst*>(&I);
        pyss<< (std::string) II->getOperand(0).first->getName() << " = relay.add(" <<  (std::string)II->getOperand(1).first->getName() << "," <<  (std::string)II->getOperand(2).first->getName()<<")"<<std::endl;
   
     //   std::cout<< (std::string) I.getName() << " = relay.add(" << std::endl;

      }
        break;

      case Kinded::Kind::ElementSubInstKind:
      {
        auto II =  static_cast<ElementSubInst*>(&I);
        pyss<< (std::string) II->getOperand(0).first->getName() << " = relay.subtract(" <<  (std::string)II->getOperand(1).first->getName() << "," <<  (std::string)II->getOperand(2).first->getName()<<")"<<std::endl;
      }
        break;

      case Kinded::Kind::ElementDivInstKind:
      {
        auto II =  static_cast<ElementAddInst*>(&I);
        pyss<< (std::string) II->getOperand(0).first->getName() << " = relay.divide(" <<  (std::string)II->getOperand(1).first->getName() << "," <<  (std::string)II->getOperand(2).first->getName()<<")"<<std::endl;

      }

    
        break;
    case Kinded::Kind::TransposeInstKind: 
    {
      auto II = static_cast<TransposeInst*>(&I);
      auto src = II->getSrc();
      auto dest = II->getDest();      
      auto shuffle = II->getShuffle();

      pyss<< (std::string) dest->getName() << " = relay.transpose(" <<  (std::string)src->getName() << ", axes=[" ;
      for(int i=0;i<shuffle.size();i++) {
            pyss << shuffle[i];
            if(i!=shuffle.size()-1) pyss << ",";
      }       
      pyss<<"])"<<std::endl;

      
    }        
     break;

      case Kinded::Kind::ConvolutionInstKind:
      {
        //    y = relay.nn.conv2d(x, w, strides=[2, 2], padding=[1, 1, 1, 1], kernel_size=[3, 3])
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
          auto II =  static_cast<ConvolutionInst*>(&I);
          // value*
          auto dst= II->getDest();
          auto src= II->getSrc();
          auto filter = II->getFilter();
          auto bias = II->getBias();

         pyss<< "conv2d_" << ir_idx << " = relay.nn.conv2d(" <<  (std::string)src->getName() << "," << (std::string)filter->getName() << ",";


          //vec<int>
          auto kernel = II->getKernels();
          auto stride = II->getStrides();
          auto padding = II->getPads();

          auto group = II->getGroup();
          auto dilation = II->getDilation();
          // convolutionLayout 
          auto layout = II->getLayout();
          //FusedActivation 
          auto fused_act = II->getFusedActivation();
          bool doRelu = fused_act == RELU;

          // bias는 별도 처리해야될 듯 => .nn.bias_add
          //fused_act도 별도 처리해야될듯
          // filter scale, offset 확인

      //  std::cout<< (std::string) dst->getName() << " = relay.nn.conv2d(" <<  (std::string)src->getName() << "," << (std::string)filter->getName() << ",";

        bool prev=false;
        if(kernel.size() > 0) {
          pyss << "kernel_size=(";
          for(int i=0;i<kernel.size();i++) {
            pyss << kernel[i];
              if(i!=kernel.size()-1) pyss << ",";
          } 
          pyss << ")";
          prev=true;
        }

        if(stride.size() > 0) {
          if(prev) pyss<< ",";
          pyss<< "strides=(";
          for(int i=0;i<stride.size();i++) {
            pyss << stride[i];
            if(i!=stride.size()-1) pyss << ",";
          } 
          pyss << ")";
          prev=true;
        }        

        if(padding.size() > 0) {
          if(prev) pyss << ",";
          pyss << "padding=(";
          for(int i=0;i<padding.size();i++) {
            pyss << padding[i];
            if(i!=padding.size()-1) pyss << ",";
          } 
          pyss << ")";
          prev=true;
        }               
        
        if(prev) pyss << ",";
        pyss << "groups=" << group ;
        pyss << ", dilation=" << dilation;

        // 넘어오는것은 data layout밖에 없는듯
        pyss << ", data_layout=";
        if(layout == ConvolutionLayout::NHWC) {
          pyss << "\"NHWC\"" << ",kernel_layout=\"OHWI\"";
        } else {
          pyss << "\"NCHW\"";
        }

        //해당 정보로 넘어는 것이 없어보임
        // kernel layout?  default는 OIHW
        // 넘어노는 값은 OHWI 인데 정보가 따로 안보임.
        // OHWI는 처리 못하므로 OIHW로 변경해줌
        TypeRef T = filter->getType();
        pyss << ",channels=" << (int)T->sizes_[0] ;

        //std::cout << ",kernel_layout=\"OHWI\"";


        // out_layout?

        pyss<< ")"<<std::endl;

        
        if(doRelu) {
          
            pyss << "bias_" << ir_idx << "=relay.nn.bias_add(" << "conv2d_" << ir_idx << "," << (std::string) bias->getName() << ",axis=3)" << std::endl;
            pyss << (std::string) dst->getName() << "=relay.nn.relu( bias_" << ir_idx << " )" << std::endl;
        } else 
        {
           pyss << (std::string) dst->getName() << "=relay.nn.bias_add(" << "conv2d_" << ir_idx << "," << (std::string) bias->getName() << ",axis=3)" << std::endl;
        }
        
      }
    
        break;

      case Kinded::Kind::FullyConnectedInstKind:
      {
//        std::cout<< (std::string) I.getName() << " = " << std::endl;
                auto II =  static_cast<FullyConnectedInst*>(&I);

               auto dest = II->getDest();
               auto src = II->getSrc();
               auto weight = II->getWeights();
               auto bias= II->getBias();

            pyss<< "fc_" << ir_idx << " = relay.nn.dense(" <<  (std::string)src->getName() << ",_op.transpose(" <<   (std::string)weight->getName()<<", [1, 0]))"<<std::endl;
            pyss<< (std::string) dest->getName() << " = relay.nn.bias_add(fc_" << ir_idx << ", " << (std::string)bias->getName()  << ")" << std::endl;

      }

        break;

      case Kinded::Kind::TensorViewInstKind:
      {
          //std::cout<< "TENSORVIEW " << I.toString() << " = " << std::endl;
          auto II =  static_cast<TensorViewInst*>(&I);
          auto T = II->getTy();

          pyss << "new_shape = []" << std::endl;
          for(int i=0;i< sizeof(T->sizes_)/sizeof(dim_t);i++) {
              if(T->sizes_[i]==0) break;
              pyss << "new_shape.append(" << (int)T->sizes_[i] << ")" <<std::endl;
          }
          pyss<< (std::string) II->getName() << " = _op.reshape(" << (std::string)II->getSrc()->getName() << ",new_shape)" << std::endl;
      }
        break;

      case Kinded::Kind::CopyInstKind:
             pyss<< "CopyInstKind  " << I.toString() << " = " << std::endl;
        break;        

      case Kinded::Kind::WeightVarKind:
          pyss<< "WeightVarKind " << I.toString() << " = " << std::endl;
        break;

      default:
        std::cout << "\n\n[ERROR] !! not added " << I.getKindName()<<std::endl;

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
                  std::cout << "\n\n!! not added " << I.getKindName()<<std::endl;
                }
  */  
    //std::cout<<"SAVE INSTRUCTION FOR "<<(std::string) I.getName() << " ||| " <<I.getType()<< " --- " << I.toString()<<std::endl;    
    //std::cout<<I.getKind()<<I.getSrc()<<I.getDest()<<I.getFilter()<<std::endl;

    ir_idx++;
  }

  if(procCtx.output_names.size() > 1) {
    pyss << "\n\noutput_tuples = relay.Tuple([";
    for(int i=0;i<procCtx.output_names.size();i++) {
      pyss << procCtx.output_names[i] << ",";
    }
    pyss << "])\n\nfunc = relay.Function(relay.analysis.free_vars(output_tuples), output_tuples)";
  } else {
    pyss << "\n\nfunc = relay.Function(relay.analysis.free_vars(" << procCtx.output_names[0] << "), " << procCtx.output_names[0] <<")";
  }
  

  py.append(pyss.str());
  

  finalizeCtx(saveCtx,outputDir,bundleName, mainEntryName, procCtx);
  
  // write genCode.py
  std::string pyFileName = (std::string)outputDir + "/relay";
  //std::cout << pyFileName.c_str();
  pyFileName.append("/genCode.py");
  std::ofstream wfos(pyFileName.c_str(), std::ios::out);
  wfos.write(py.c_str(),py.size());
  wfos.write(inc.c_str(),inc.size());
  wfos.write(cpp.c_str(),cpp.size());
  wfos.close();

  std::string mkFileName = (std::string)outputDir + "/module";
  mkFileName.append("/Makefile");
  std::ofstream mfos(mkFileName.c_str(), std::ios::out);
  mfos.write(mk.c_str(),mk.size());
  mfos.close();


  return;
}


