/**
 * Copyright (c) 2021, Etri.
 *
 * This program or software including the accompanying associated documentation
 * ("Software") is the proprietary software of Etri and/or its licensors.
 * Etri reserves all rights in and to the Software and all intellectual
 * property therein. The Software is distributed on an "AS IS" basis, without
 * warranties or conditions of any kind, unless required by applicable law or a
 * written agreement.
 *
 * @file: NMPLLVMIRGen.cpp
 * @brief description: NMP specific Code Generation for the model.
 * @date: 11 18, 2021
 */

#include "NMPLLVMIRGen.h"
#include "NMPBackend.h"

#include <cmath>

#include "glow/Base/DimType.h"

#include "glow/LLVMIRCodeGen/AllocationsInfo.h"
#include "glow/LLVMIRCodeGen/CommandLine.h"
#include "glow/LLVMIRCodeGen/LLVMBackend.h"

#include "glow/Graph/Graph.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Backends/NMP/CommandLine.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace glow;
using namespace llvm;

// Search for the standard library bitcode file on disk and load it into an
// LLVM module. We search for the standard library around the current executable
// and also in the current directory.
static std::unique_ptr<llvm::Module>
loadStandardLibrary(llvm::LLVMContext *ctx, llvm::StringRef filename,
                    llvm::StringRef libjitBC) {
  using llvm::sys::path::append;
  using llvm::sys::path::parent_path;

  llvm::SMDiagnostic error;

  // Parse the compiled-in image of libjit and return the resulting Module.
  // checking for and reporting errors from parseIR.

  auto mod = llvm::parseIR(
      llvm::MemoryBufferRef(
          llvm::StringRef(reinterpret_cast<const char *>(libjitBC.data()),
                          libjitBC.size()),
          "libjit.bc"),
      error, *ctx);

  if (!mod) {
    error.print("NMPLLVMIRGen", llvm::errs());
  }
  return mod;
}

// Register a diagnostics handler that prevents the compiler from printing to
// stdout.
static void registerEmptyDiagHandler(llvm::LLVMContext &ctx) {
  ctx.setDiagnosticHandlerCallBack(
      [](const llvm::DiagnosticInfo &DI, void *Context) {
        // Do not emit any warnings or diagnostics when JITting.
      });
}

void NMPLLVMIRGen::addBundleObject(llvm::StringRef objectName) {
  // Add bundle object if not already added.
  auto it =
      std::find(bundleObjects_.begin(), bundleObjects_.end(), objectName.str());
  if (it == bundleObjects_.end()) {
    bundleObjects_.push_back(objectName.str());
  }
}

NMPLLVMIRGen::NMPLLVMIRGen(const IRFunction *F,
                           AllocationsInfo &allocationsInfo,
                           std::string mainEntryName, llvm::StringRef libjitBC)
    : LLVMIRGen(F, allocationsInfo, mainEntryName, libjitBC){};

NMPLLVMIRGen::NMPLLVMIRGen(const IRFunction *F,
                           AllocationsInfo &allocationsInfo,
                           std::string mainEntryName, llvm::StringRef libjitBC,
                           llvm::ArrayRef<llvm::MemoryBufferRef> objectRegistry)
    : LLVMIRGen(F, allocationsInfo, mainEntryName, libjitBC, objectRegistry){};

void NMPLLVMIRGen::initCodeGen() {
  // Load the nmp library as a new module.
  llmodule_ = loadStandardLibrary(&getLLVMContext(), "libnmp.bc", libjitBC_);
  CHECK(llmodule_.get()) << "Unable to load the nmp library.";
  // provides a dummy diagnostics handler, that does not emit anything.
  registerEmptyDiagHandler(getLLVMContext());
  // Assign the target information to the module.
  llmodule_->setDataLayout(getTargetMachine().createDataLayout());
}

void NMPLLVMIRGen::performCodeGen() {
  // Create the entry function into the LLVM module.
  llvm::Type *retTy =
      llvm::Type::getIntNTy(getLLVMContext(), getLibjitIntWidth());
  llvm::FunctionType *jitFuncTy = llvm::FunctionType::get(retTy, {}, false);
  llvmF_ = llvm::Function::Create(jitFuncTy, llvm::Function::ExternalLinkage,
                                  "_main", llmodule_.get());
  emittedLLVMFunctions_.emplace_back(llvmF_);

  // Setup the entry basic block and initialize the IR builder.
  llvm::BasicBlock *entry_bb =
      llvm::BasicBlock::Create(getLLVMContext(), "entry", llvmF_);
  builder_ = glow::make_unique<llvm::IRBuilder<>>(entry_bb);
  // Terminate the function with a return instruction.
  auto zero = builder_->getIntN(getLibjitIntWidth(), 0);
  auto *ret = builder_->CreateRet(zero);
  // Emit all the code before the retrun instruction.
  builder_->SetInsertPoint(ret);

  instrNumbering_.reset(new InstructionNumbering(*F_));
  generateFunctionDebugInfo();
  loadBaseAddresses(*builder_);
  generateLLVMIRForModule(*builder_);
}

llvm::Function *NMPLLVMIRGen::getFunction(const std::string &name) {
  auto strName = "_nmp_" + name;
  auto *F = llmodule_->getFunction(strName);
  CHECK(F) << "Unable to load the function: " << strName.c_str();
  return F;
}

void NMPLLVMIRGen::loadBaseAddresses(llvm::IRBuilder<> &builder) {
  uint32_t baseConstantWeight{NMP_BASE_DATA_ADDR};
  baseConstantWeightVarsAddr_ =
      LLVMIRGen::emitConstI32(builder, baseConstantWeight);
  uint32_t baseMutableWeight =
      baseConstantWeight + allocationsInfo_.constantWeightVarsMemSize_;
  baseMutableWeightVarsAddr_ =
      LLVMIRGen::emitConstI32(builder, baseMutableWeight);
  uint32_t baseActivations =
      baseMutableWeight + allocationsInfo_.mutableWeightVarsMemSize_;
  baseActivationsAddr_ = LLVMIRGen::emitConstI32(builder, baseActivations);
}

llvm::Value *NMPLLVMIRGen::emitElemKind(llvm::IRBuilder<> &builder,
                                        ElemKind elemKind) {
  switch (elemKind) {
  case ElemKind::Int8QTy:
    return LLVMIRGen::emitConstI16(builder, NMP_CFG_DFXP_SIZE_D08);
  case ElemKind::Int16QTy:
    return LLVMIRGen::emitConstI16(builder, NMP_CFG_DFXP_SIZE_D16);
  default:
    return nullptr;
  }
}

template <class InstructionTy>
llvm::Value *NMPLLVMIRGen::emitFusedActivationType(llvm::IRBuilder<> &builder,
                                                   const InstructionTy *I) {
  uint16_t NMP_ACT_TYPE{0};
  auto actArgsF = I->getFusedActivationArgs();
  switch (I->getFusedActivation()) {
  case FusedActivation::NONE:
    NMP_ACT_TYPE = NMP_ACT_NONE;
    break;

  case FusedActivation::RELU:
    assert(actArgsF.size() == 0 && "Invalid number of activation parameters!");
    NMP_ACT_TYPE = NMP_ACT_RELU;
    break;

  case FusedActivation::CLIP:
    // For Clip we quantize min/max using the output quantization params.
    assert(actArgsF.size() == 2 &&
           "Invalid number of parameters for fused Clip activation!");
    /* float minF = actArgsF[0]; */
    /* float maxF = actArgsF[1]; */
    /* NMP_ACT_TYPE = NMP_ACT_CLIP; */
    LOG(FATAL) << "Unsupported fused activation type (Clip)!";
    break;

  case FusedActivation::SIGMOID:
    NMP_ACT_TYPE = NMP_ACT_SIGM;
    break;

  case FusedActivation::TANH:
    NMP_ACT_TYPE = NMP_ACT_TANH;
    break;

  case FusedActivation::LEAKY_RELU:
    assert(actArgsF.size() == 1 &&
           "Invalid number of parameters for fused LeakyRelu activation!");
    /* float alpha = actArgsF[0]; */
    LOG(FATAL) << "Unsupported fused activation type (LeakyRelu)!";
    break;
  }

  return LLVMIRGen::emitConstI16(builder, NMP_ACT_TYPE);
}

llvm::Value *NMPLLVMIRGen::emitValueAddress(llvm::IRBuilder<> &builder,
                                            const glow::Value *val) {

  assert(allocationsInfo_.allocatedAddress_.count(val) &&
         "Value address was not allocated");
  assert(allocationsInfo_.valueNumbers_.count(val));
  auto &kindAndValue = allocationsInfo_.valueNumbers_[val];
  // Get the required base address.
  llvm::Value *baseAddrValue = nullptr;
  switch (kindAndValue.first) {
  case AllocationsInfo::ValueKind::Activation:
    baseAddrValue = baseActivationsAddr_;
    break;
  case AllocationsInfo::ValueKind::ConstantWeight:
    baseAddrValue = baseConstantWeightVarsAddr_;
    break;
  case AllocationsInfo::ValueKind::MutableWeight:
    baseAddrValue = baseMutableWeightVarsAddr_;
    break;
  }
  // Add offset to the base address.
  auto offsetValue = allocationsInfo_.allocatedAddress_[val];
  llvm::Value *addr =
      builder.CreateAdd(baseAddrValue, builder.getInt32(offsetValue));
  return addr;
}

void NMPLLVMIRGen::generateLLVMIRForModule(llvm::IRBuilder<> &builder) {
  createCall(builder, getFunction("init_layer"), {});

  // undocumented break after init_layer
  if (haltLayerOpt == 999)
    createCall(builder, getFunction("halt_layer"), {});

  auto *FSync = getFunction("sync_layer");
  auto *tle = LLVMIRGen::emitConstI32(builder, NMP_NUM_TLE);
  auto *tlt = LLVMIRGen::emitConstI32(builder, NMP_NUM_TLT);
  int32_t gsemma{0};
  int32_t gsync{1};
  unsigned nLayers{0};
  auto &instrs = F_->getInstrs();
  for (auto &I : instrs) {
    if (gsync) {
      createCall(builder, FSync,
                 {tle, tlt, LLVMIRGen::emitConstI32(builder, gsemma)});
      gsemma = (gsemma == 0) ? 1 : 0;
      // haltLayer is used for debug purposes. Its insert a halt_layer
      // after the layer we get to stop the code execution.
      nLayers++;
      if (haltLayerOpt != 0)
        if (nLayers == haltLayerOpt + 1)
          createCall(builder, getFunction("halt_layer"), {});
    }
    gsync = generateIRCodeForInstr(builder, &I);
  }
  if (gsync) {
    createCall(builder, FSync,
               {tle, tlt, LLVMIRGen::emitConstI32(builder, gsemma)});
  }
  createCall(builder, getFunction("halt_layer"), {});
}

template <typename T>
void NMPLLVMIRGen::generatePoolInstr(llvm::IRBuilder<> &builder,
                                     const int NMP_POOL_TYP,
                                     const glow::Instruction *I) {
  auto *PI = cast<T>(I);
  auto *src = PI->getSrc();
  assert(src->getType()->isQuantizedType() &&
         "Unsupported Type in Pool Op (MaxPool/AvgPool)");
  auto *dest = PI->getDest();
  auto *dfxpMode = emitElemKind(builder, dest->getElementType());

  // Offsets
  auto *srcPtr = emitValueAddress(builder, src);
  auto *destPtr = emitValueAddress(builder, dest);
  auto *ofm_base_idx = LLVMIRGen::emitConstI32(builder, 0);

  // Dimensions
  uint16_t src_c = src->dims()[1];
  uint16_t src_h = src->dims()[2];
  uint16_t src_w = src->dims()[3];
  uint16_t dest_h = dest->dims()[2];
  uint16_t dest_w = dest->dims()[3];
  auto *inChannels = LLVMIRGen::emitConstI16(builder, src_c);
  auto *inHeight = LLVMIRGen::emitConstI16(builder, src_h);
  auto *inWidth = LLVMIRGen::emitConstI16(builder, src_w);
  auto *outHeight = LLVMIRGen::emitConstI16(builder, dest_h);
  auto *outWidth = LLVMIRGen::emitConstI16(builder, dest_w);

  // kernels
  uint16_t kernel_h = PI->getKernels()[0];
  uint16_t kernel_w = PI->getKernels()[1];
  auto *pool_kernel_h = LLVMIRGen::emitConstI16(builder, kernel_h);
  auto *pool_kernel_w = LLVMIRGen::emitConstI16(builder, kernel_w);

  // strides
  uint16_t stride_h = PI->getStrides()[0];
  uint16_t stride_w = PI->getStrides()[1];
  auto *pool_stride_h = LLVMIRGen::emitConstI16(builder, stride_h);
  auto *pool_stride_w = LLVMIRGen::emitConstI16(builder, stride_w);

  // Pads
  uint16_t padBottom = PI->getPads()[0];
  uint16_t padLeft = PI->getPads()[1];
  uint16_t padTop = PI->getPads()[2];
  uint16_t padRight = PI->getPads()[3];

  if (!(padTop == 0 && padBottom == 0 && padLeft == 0 && padRight == 0))
    if (recomputePoolPads)
      // There is an issue with paddings that we fix below
      if ((padRight == padLeft) || (padTop == padBottom)) {
        float padding_w =
            ((float)((dest_w - 1) * stride_w) - src_w + kernel_w) / 2;
        float padding_h =
            ((float)((dest_h - 1) * stride_h) - src_h + kernel_h) / 2;
        padBottom = (uint16_t)floor(padding_h);
        padLeft = (uint16_t)ceil(padding_w);
        padTop = (uint16_t)ceil(padding_h);
        padRight = (uint16_t)floor(padding_w);
      }

  auto *pool_pad_w =
      LLVMIRGen::emitConstI16(builder, ((padRight << 8) | padLeft));
  auto *pool_pad_h =
      LLVMIRGen::emitConstI16(builder, ((padBottom << 8) | padTop));

  // types
  auto *pool_type = LLVMIRGen::emitConstI16(builder, NMP_POOL_TYP);
  uint16_t padtype = NMP_POOL_PAD_ZERO;
  if (padTop || padBottom || padLeft || padRight) {
    padtype =
        (NMP_POOL_TYP == NMP_POOL_MAX) ? NMP_POOL_PAD_NEG : NMP_POOL_PAD_AVG;
  }
  auto *pad_type = LLVMIRGen::emitConstI16(builder, padtype);

  // q-points

  int16_t factor{0};
  auto *srcTy = src->getType();
  auto *destTy = dest->getType();
  if (srcTy->getScale() == destTy->getScale()) {
    // There's an issue on Inception 2 & 3 due to RescaleQuantizedInst
    // that I can't figure out. So the factor below is a palliative
    // solution until I find out how to fix it.
    auto name = PI->getName();
    if (name == "InceptionV3_InceptionV3_Mixed_6a_Branch_2_MaxPool_1a_3x3_"
                "MaxPool__1" ||
        name == "MaxPool_inception_4e_pool_1__1")
      factor = 1;
  }
  int16_t srcQValue = log2(1 / srcTy->getScale());
  int16_t destQValue = log2(1 / destTy->getScale()) - factor;
  int16_t destILScale = srcQValue - destQValue;

  auto *pool_q_df = LLVMIRGen::emitConstI16(builder, srcQValue);
  auto *pool_q_wf = LLVMIRGen::emitConstI16(builder, 0);
  auto *pool_q_ls = LLVMIRGen::emitConstI16(builder, destILScale);
  auto *scale_q = LLVMIRGen::emitConstI16(builder, 0);

  // Load-mode
  auto *srcMode = LLVMIRGen::emitConstI16(builder, NMP_ELMSIZE_SAME);

  // Mapper's config
  uint16_t dataSize = (dest->getElementType() == ElemKind::Int16QTy) ? 2 : 1;
  uint16_t ifm_c{1};
  if (src_c >= NMP_NUM_TLTS) {
    ifm_c = (uint16_t)ceil(src_c / (float)NMP_NUM_TLTS);
  }
  uint16_t ofm_w = dest_w;
  uint16_t ofm_h = 1;
  uint16_t ifm_w = src_w + padLeft + padRight;
  for (uint16_t i = 2; i <= dest_h; i++) {
    uint16_t ifm_h = (i - 1) * stride_h + kernel_h;
    uint16_t ifm_size = ifm_h * ifm_w * dataSize;
    if (ifm_size <= NMP_MBLOB_SIZE) {
      ofm_h = i;
    } else {
      break;
    }
  }

  auto *_num_tlt = LLVMIRGen::emitConstI16(builder, NMP_NUM_TLT);
  auto *slice_ifm_c = LLVMIRGen::emitConstI16(builder, ifm_c);
  auto *slice_ofm_w = LLVMIRGen::emitConstI16(builder, ofm_w);
  auto *slice_ofm_h = LLVMIRGen::emitConstI16(builder, ofm_h);

  auto *F = getFunction("pool_layer");
  createCall(builder, F,
             {dfxpMode,      srcPtr,        destPtr,
              ofm_base_idx,  inWidth,       inHeight,   inChannels,
              outWidth,      outHeight,     pool_type,  pad_type,
              pool_kernel_h, pool_kernel_w, pool_pad_w, pool_pad_h,
              pool_stride_w, pool_stride_h, pool_q_df,  pool_q_wf,
              pool_q_ls,     scale_q,       srcMode,    _num_tlt,
              slice_ifm_c,   slice_ofm_w,   slice_ofm_h});
}

template <typename T>
void NMPLLVMIRGen::generateEltwiseInstr(llvm::IRBuilder<> &builder,
                                        const int NMP_ELTWISE_TYP,
                                        const glow::Instruction *I) {
  auto *AN = cast<T>(I);
  auto *lhs = AN->getLHS();
  assert(lhs->getType()->isQuantizedType() &&
         "Unsupported Type in Arithmetic Op");
  auto *rhs = AN->getRHS();
  auto *dest = AN->getDest();
  auto *dfxpMode = emitElemKind(builder, dest->getElementType());

  // Offsets
  auto *lhsPtr = emitValueAddress(builder, lhs);
  auto *rhsPtr = emitValueAddress(builder, rhs);
  auto *destPtr = emitValueAddress(builder, dest);

  auto *eltwise_type = LLVMIRGen::emitConstI16(builder, NMP_ELTWISE_TYP);
  auto *F = getFunction("eltwise_layer");

  // Dimension
  auto *eltsize = LLVMIRGen::emitConstI32(
      builder, lhs->dims()[1] * lhs->dims()[2] * lhs->dims()[3]);

  // Q-Points
  auto *destTy = dest->getType();
  auto *lhsTy = lhs->getType();
  auto *rhsTy = rhs->getType();
  int16_t lhsQValue = log2(1 / lhsTy->getScale());
  int16_t rhsQValue = log2(1 / rhsTy->getScale());
  int16_t destScale = lhsQValue - log2(1 / destTy->getScale());
  auto *lhsQP = LLVMIRGen::emitConstI16(builder, lhsQValue);
  auto *rhsQP = LLVMIRGen::emitConstI16(builder, rhsQValue);
  auto *destILScale = LLVMIRGen::emitConstI16(builder, destScale);

  // Load-mode
  auto *lhsLMode = LLVMIRGen::emitConstI16(builder, NMP_ELMSIZE_SAME);
  auto *rhsLMode = LLVMIRGen::emitConstI16(builder, NMP_ELMSIZE_SAME);

  // Mapper's config
  uint16_t dataSize = (dest->getElementType() == ElemKind::Int16QTy) ? 2 : 1;
  uint16_t channels = dest->dims()[1];
  uint16_t rows = dest->dims()[2];
  uint16_t cols = dest->dims()[3];
  int totalElements = channels * rows * cols;

  // Wy not just compute the ceil?
  uint16_t nElementsPerTLT = floor(totalElements / NMP_NUM_TLTS);
  nElementsPerTLT += ((totalElements % NMP_NUM_TLTS) > 0) ? 1 : 0;

  uint16_t nElementsPerTile =
      floor((nElementsPerTLT * dataSize) / (float)NMP_MBLOB_SIZE);
  nElementsPerTile =
      (nElementsPerTile == 0) ? nElementsPerTLT : (NMP_MBLOB_SIZE / dataSize);

  auto *_thread = LLVMIRGen::emitConstI32(builder, nElementsPerTLT);
  auto *_slice = LLVMIRGen::emitConstI16(builder, nElementsPerTile);
  auto *_n_tlt = LLVMIRGen::emitConstI16(builder, NMP_NUM_TLT);

  createCall(builder, F,
             {dfxpMode, lhsPtr, rhsPtr, destPtr, eltsize, eltwise_type, lhsQP,
              rhsQP, destILScale, lhsLMode, rhsLMode, _thread, _slice, _n_tlt});
}

int32_t NMPLLVMIRGen::generateIRCodeForInstr(llvm::IRBuilder<> &builder,
                                             const glow::Instruction *I) {
  setCurrentDebugLocation(builder, I);
  // Perform NMP-specific code generation
  switch (I->getKind()) {

  // Alloc and Dealloc instructions are handled by the memory allocator.
  case Kinded::Kind::AllocActivationInstKind:
  case Kinded::Kind::DeallocActivationInstKind:
  case Kinded::Kind::TensorViewInstKind:
    return 0;

  case Kinded::Kind::QuantizeInstKind: {
    auto *QI = cast<QuantizeInst>(I);
    auto *src = QI->getSrc();
    assert(src->getType()->isFPType() &&
           "Quantized Op allowed only for Input data.");
    auto *dest = QI->getDest();
    auto *destTy = dest->getType();

    auto baseActivations = NMP_BASE_DATA_ADDR +
                           allocationsInfo_.constantWeightVarsMemSize_ +
                           allocationsInfo_.mutableWeightVarsMemSize_;
    auto addr = baseActivations + allocationsInfo_.allocatedAddress_[dest];

    auto scale = destTy->getScale();
    inputAddrScale = std::make_pair(addr, scale);
    return 0;
  }

  case Kinded::Kind::DequantizeInstKind: {
    if (computeNMPSoftmaxOpt) {
      // This is the default situation. If the value is false
      // implies that return address was computed in Softmax.
      auto *DI = cast<DequantizeInst>(I);
      auto *src = DI->getSrc();

      auto baseActivations = NMP_BASE_DATA_ADDR +
                             allocationsInfo_.constantWeightVarsMemSize_ +
                             allocationsInfo_.mutableWeightVarsMemSize_;
      auto addr = baseActivations + allocationsInfo_.allocatedAddress_[src];

      auto scale = src->getType()->getScale();
      outputAddrScale = std::make_pair(addr, scale);
    }
    return 0;
  }
  case Kinded::Kind::ElementAddInstKind: {
    generateEltwiseInstr<ElementAddInst>(builder, NMP_ELTWISE_SUM, I);
    return 1;
  }

  case Kinded::Kind::ElementMaxInstKind: {
    generateEltwiseInstr<ElementMaxInst>(builder, NMP_ELTWISE_MAX, I);
    return 1;
  }

  case Kinded::Kind::ElementMulInstKind: {
    generateEltwiseInstr<ElementMulInst>(builder, NMP_ELTWISE_MUL, I);
    return 1;
  }

  case Kinded::Kind::ConvolutionInstKind: {
    auto *CI = cast<ConvolutionInst>(I);
    auto *src = CI->getSrc();
    assert(src->getType()->isQuantizedType() &&
           "Non-quantized Instruction is not supported in NMP!");
    auto *dest = CI->getDest();
    auto *filter = CI->getFilter();
    auto *bias = CI->getBias();

    // Precision
    auto *dfxpMode = emitElemKind(builder, dest->getElementType());

    // Offsets
    auto *srcPtr = emitValueAddress(builder, src);
    auto *destPtr = emitValueAddress(builder, dest);
    auto *filterPtr = emitValueAddress(builder, filter);
    auto *biasPtr = emitValueAddress(builder, bias);

    // Dimensions
    uint16_t src_c = src->dims()[1];
    uint16_t src_h = src->dims()[2];
    uint16_t src_w = src->dims()[3];
    uint16_t dest_h = dest->dims()[2];
    uint16_t dest_w = dest->dims()[3];
    auto *inChannels = LLVMIRGen::emitConstI16(builder, src_c);
    auto *inHeight = LLVMIRGen::emitConstI16(builder, src_h);
    auto *inWidth = LLVMIRGen::emitConstI16(builder, src_w);
    auto *outHeight = LLVMIRGen::emitConstI16(builder, dest_h);
    auto *outWidth = LLVMIRGen::emitConstI16(builder, dest_w);

    // Filters
    uint16_t filters = filter->dims()[0];
    auto *filterNum = LLVMIRGen::emitConstI16(builder, filters);

    // Kernels
    uint16_t kernel_h = CI->getKernels()[0];
    uint16_t kernel_w = CI->getKernels()[1];
    auto *conv_kernel_h =
        LLVMIRGen::emitConstI16(builder, (kernel_h << 8) | kernel_h);
    auto *conv_kernel_w =
        LLVMIRGen::emitConstI16(builder, (kernel_w << 8) | kernel_w);

    // Strides
    uint16_t stride_h = CI->getStrides()[0];
    uint16_t stride_w = CI->getStrides()[1];
    assert(stride_h == stride_w && "Invalid Strides.");
    auto *strides = LLVMIRGen::emitConstI16(builder, stride_h);

    // Pads
    uint16_t padBottom = CI->getPads()[0];
    uint16_t padLeft = CI->getPads()[1];
    uint16_t padTop = CI->getPads()[2];
    uint16_t padRight = CI->getPads()[3];

    if (!(padTop == 0 && padBottom == 0 && padLeft == 0 && padRight == 0))
      if (recomputeConvPads)
        // There is an issue with paddings that we fix below
        if ((padRight == padLeft) || (padTop == padBottom)) {
          float padding_w =
              ((float)((dest_w - 1) * stride_w) - src_w + kernel_w) / 2;
          float padding_h =
              ((float)((dest_h - 1) * stride_h) - src_h + kernel_h) / 2;
          padBottom = (uint16_t)floor(padding_h);
          padLeft = (uint16_t)ceil(padding_w);
          padTop = (uint16_t)ceil(padding_h);
          padRight = (uint16_t)floor(padding_w);
        }

    auto *conv_pad_w =
        LLVMIRGen::emitConstI16(builder, (padRight << 8) | padLeft);
    auto *conv_pad_h =
        LLVMIRGen::emitConstI16(builder, (padBottom << 8) | padTop);

    // Dilation
    uint16_t dilation_v = CI->getDilation()[0];
    assert(dilation_v == CI->getDilation()[1] && "Invalid dilation.");
    auto *dilation = LLVMIRGen::emitConstI16(builder, dilation_v);

    // Activation
    auto *act_type = emitFusedActivationType(builder, CI);
    auto *prelu_slope = LLVMIRGen::emitConstI16(builder, 0);

    // Compute the fixed point (Q-Value) of the input, & filter
    // Also, compute the of InnerLayerScale of the output
    // InnerLayerScale defines the number of bits that output data
    // needs to be shifted by. This value is interpreted as a signed
    // number, a positive value shifts right, a negative shifts left.
    auto *srcTy = src->getType();
    auto *destTy = dest->getType();
    auto *filterTy = filter->getType();
    auto *biasTy = bias->getType();

    int16_t srcQValue = log2(1 / srcTy->getScale());
    int16_t filterQValue = log2(1 / filterTy->getScale());
    int16_t biasQValue = log2(1 / biasTy->getScale());
    int16_t destILScale = srcQValue - log2(1 / destTy->getScale());

    auto *srcQP = LLVMIRGen::emitConstI16(builder, srcQValue);
    auto *filterQP = LLVMIRGen::emitConstI16(builder, filterQValue);
    auto *biasQP = LLVMIRGen::emitConstI16(builder, biasQValue);
    auto *outILScale = LLVMIRGen::emitConstI16(builder, destILScale);

    // Load-mode
    auto *srcMode = LLVMIRGen::emitConstI16(builder, NMP_ELMSIZE_SAME);

    // Check if the Convolution is DeepWise
    bool isdw = CI->getGroup() != 1;

    // Mapper's config
    Conv2dInfo conv = {src_c,    src_h,      src_w,    dest_h,
                       dest_w,   kernel_h,   kernel_w, filters,
                       padTop,   padBottom,  padLeft,  padRight,
                       stride_h, dilation_v, isdw,    (bias != nullptr)};

    uint16_t dataSize = (dest->getElementType() == ElemKind::Int16QTy) ? 2 : 1;
    ArchitectureInfo archInfo = {
        dataSize,       // f_size [B]
        16,             // mac_per_tlt
        1000,           // frequency [MHz]
        NMP_MBLOB_SIZE, // mblob_size [B]
        NMP_NUM_TLE,    // num_tle
        NMP_NUM_TLT,    // num_tlt_per_tle
        0.000014,       // CAS Latency [s]
        128,            // burst_size
        8533.0          // max_ddr_bw [MB/s]
    };

    TLEPartition partition = TLEPartition(archInfo, conv);
    partition.Execute();
    MapperInfo res = partition.GetResult();

    auto *_num_tle = LLVMIRGen::emitConstI16(builder, NMP_NUM_TLE);
    auto *_num_tlt = LLVMIRGen::emitConstI16(builder, NMP_NUM_TLT);
    auto *_mb_size = LLVMIRGen::emitConstI16(builder, NMP_MBLOB_SIZE);
    auto *_thread_np_y = LLVMIRGen::emitConstI16(builder, res.thread_np_y);
    auto *_thread_np_c = LLVMIRGen::emitConstI16(builder, res.thread_np_c);
    auto *_thread_conv_wgts_mb =
        LLVMIRGen::emitConstI16(builder, res.slice._filters);
    auto *_slice_ifm_c = LLVMIRGen::emitConstI16(builder, res.slice._ifm_c);
    auto *_slice_ofm_h = LLVMIRGen::emitConstI16(builder, res.slice._ofm_h);
    auto *_slice_ofm_w = LLVMIRGen::emitConstI16(builder, res.slice._ofm_w);
    auto *_slice_filters = LLVMIRGen::emitConstI16(builder, res.slice._filters);
    auto *_ofm_ch_per_tlt = LLVMIRGen::emitConstI16(builder, res.slice._ofm_c);

    auto *F = getFunction(res.fn_name);
    if (isdw) {
      createCall(builder, F,
                 {dfxpMode,     srcPtr,         destPtr,         filterPtr,
                  biasPtr,      inWidth,        inHeight,        outWidth,
                  outHeight,    conv_kernel_w,  conv_kernel_h,   filterNum,
                  conv_pad_w,   conv_pad_h,     strides,         dilation,
                  act_type,     prelu_slope,    srcQP,           filterQP,
                  biasQP,       outILScale,     srcMode,
                  _num_tle,     _num_tlt,       _thread_np_y,    _slice_ofm_w,
                  _slice_ofm_h, _slice_filters, _ofm_ch_per_tlt, _mb_size});
    } else {
      createCall(builder, F,
                 {dfxpMode, srcPtr, destPtr, filterPtr, biasPtr, inWidth,
                  inHeight, inChannels, outWidth, outHeight, conv_kernel_w,
                  conv_kernel_h, filterNum, conv_pad_w, conv_pad_h, strides,
                  dilation, act_type, prelu_slope, srcQP, filterQP, biasQP,
                  outILScale, srcMode, _num_tlt, _thread_np_y,
                  _thread_np_c, _thread_conv_wgts_mb, _slice_ofm_w, _slice_ofm_h,
                  _slice_ifm_c, _mb_size});
    }
    return 1;
  }

  case Kinded::Kind::MaxPoolInstKind: {
    generatePoolInstr<MaxPoolInst>(builder, NMP_POOL_MAX, I);
    return 1;
  }

  case Kinded::Kind::ReluInstKind: {
    auto *RI = cast<ReluInst>(I);
    auto *src = RI->getSrc();
    assert(src->getType()->isQuantizedType() && "Unsupported Type in Relu Op");
    auto *dest = RI->getDest();

    // Precision
    auto *out_dtype = emitElemKind(builder, dest->getElementType());

    auto *srcPtr = emitValueAddress(builder, src);
    auto *destPtr = emitValueAddress(builder, dest);

    // number of inputs
    uint16_t n = src->dims()[0];
    uint16_t channels = src->dims()[1];
    uint16_t rows = src->dims()[2];
    uint16_t cols = src->dims()[3];
    auto *in_w = LLVMIRGen::emitConstI32(builder, n * channels * rows * cols);

    // activation
    auto *actv = LLVMIRGen::emitConstI16(builder, NMP_ACT_RELU);

    // q-points
    auto srcTy = src->getType();
    auto destTy = dest->getType();
    int16_t srcQValue = log2(1 / srcTy->getScale());
    int16_t destILScale = srcQValue - log2(1 / destTy->getScale());
    auto *in_fxp = LLVMIRGen::emitConstI16(builder, srcQValue);
    auto *alpha_fxp = LLVMIRGen::emitConstI16(builder, 0); // check
    auto *out_ls = LLVMIRGen::emitConstI16(builder, destILScale);

    // Mapper's config

    // number of inputs to be processed in a tilelet
    int totalElements = channels * rows * cols;
    uint16_t nElementsPerTLT = floor(totalElements / NMP_NUM_TLTS);
    nElementsPerTLT += ((totalElements % NMP_NUM_TLTS) > 0) ? 1 : 0;

    // number of inputs to be processed at once in a tilelet
    uint16_t dataSize = (dest->getElementType() == ElemKind::Int16QTy) ? 2 : 1;
    uint16_t nElementsPerTile =
        floor((nElementsPerTLT * dataSize) / (float)NMP_MBLOB_SIZE);
    nElementsPerTile =
        (nElementsPerTile == 0) ? nElementsPerTLT : (NMP_MBLOB_SIZE / dataSize);

    auto *dtype_cast = LLVMIRGen::emitConstI16(builder, NMP_ELMSIZE_SAME);
    auto *part_thread_w = LLVMIRGen::emitConstI32(builder, nElementsPerTLT);
    auto *part_slice_w = LLVMIRGen::emitConstI32(builder, nElementsPerTile);
    auto *alpha = LLVMIRGen::emitConstI16(builder, 0);  // used for PRelu
    auto *ntlts = LLVMIRGen::emitConstI16(builder, NMP_NUM_TLT);

    auto *F = getFunction("activation_layer");
    createCall(builder, F,
               {out_dtype, srcPtr, destPtr, in_w, actv, in_fxp, alpha_fxp,
                out_ls, dtype_cast, part_thread_w, part_slice_w, alpha, ntlts});
    return 1;
  }

  case Kinded::Kind::ClipInstKind: {
    auto *CI = cast<ClipInst>(I);
    auto *src = CI->getSrc();
    assert(src->getType()->isQuantizedType() && "Unsupported Type in Clip op");
    auto *dest = CI->getDest();

    // Precision
    auto *out_dtype = emitElemKind(builder, dest->getElementType());

    auto *srcPtr = emitValueAddress(builder, src);
    auto *destPtr = emitValueAddress(builder, dest);

    // q-points
    auto srcTy = src->getType();
    auto destTy = dest->getType();
    int16_t srcQValue = log2(1 / srcTy->getScale());
    int16_t destILScale = srcQValue - log2(1 / destTy->getScale());
    auto *act_q_df = LLVMIRGen::emitConstI16(builder, srcQValue);
    auto *act_q_wf = LLVMIRGen::emitConstI16(builder, 0);
    auto *act_q_ls = LLVMIRGen::emitConstI16(builder, destILScale);

    // Load-mode
    auto *lMode = LLVMIRGen::emitConstI16(builder, NMP_ELMSIZE_SAME);

    // input size
    uint32_t n = src->dims()[0];
    uint32_t channels = src->dims()[1];
    uint32_t rows = src->dims()[2];
    uint32_t cols = src->dims()[3];
    uint32_t totalElements = n * channels * rows * cols;
    auto *ifm_size = LLVMIRGen::emitConstI32(builder, totalElements);

    // Mapper's config

    // number of inputs to be processed in a tilelet
    uint32_t nElementsPerTlt = totalElements / NMP_NUM_TLTS;
    nElementsPerTlt += ((totalElements % NMP_NUM_TLTS) > 0) ? 1 : 0;

    // number of inputs to be processed at once in a tilelet
    uint32_t dataSize = (dest->getElementType() == ElemKind::Int16QTy) ? 2 : 1;
    uint32_t nElementsPerTile = (nElementsPerTlt * dataSize) / NMP_MBLOB_SIZE;
    nElementsPerTile =
        (nElementsPerTile == 0) ? nElementsPerTlt : (NMP_MBLOB_SIZE / dataSize);

    auto *cnfg_ifm_thread = LLVMIRGen::emitConstI32(builder, nElementsPerTlt);
    auto *cnfg_ifm_slice = LLVMIRGen::emitConstI32(builder, nElementsPerTile);
    auto *cnfg_num_tlt = LLVMIRGen::emitConstI16(builder, NMP_NUM_TLT);

    // Clip Min & Max Values:
    // NMP only supports for now Clip == Relu6, i.e., Min==0 & Max==6
    float clipMinF = CI->getMin();
    float clipMaxF = CI->getMax();
    TensorQuantizationParams srcTQP{src->getType()->getScale(),
                                    src->getType()->getOffset()};
    int16_t clipMinQ = quantization::quantize<int16_t>(clipMinF, srcTQP);
    int16_t clipMaxQ = quantization::quantize<int16_t>(clipMaxF, srcTQP);
    /* auto *clipMin = LLVMIRGen::emitConstI16(builder, clipMinQ); */
    /* auto *clipMax = LLVMIRGen::emitConstI16(builder, clipMaxQ); */
    assert((clipMinQ == 0 && clipMaxQ == 6) && "NMP only supports Relu6");

    auto *F = getFunction("activation_layer_relu6");
    createCall(builder, F,
               {out_dtype, srcPtr, destPtr, ifm_size, act_q_df, act_q_wf,
                act_q_ls, lMode, cnfg_ifm_thread, cnfg_ifm_slice,
                cnfg_num_tlt});
    return 1;
  }

  case Kinded::Kind::AvgPoolInstKind: {
    generatePoolInstr<AvgPoolInst>(builder, NMP_POOL_AVG, I);
    return 1;
  }

  case Kinded::Kind::NMPFullyConnectedInstKind: {
    auto *FCI = cast<NMPFullyConnectedInst>(I);
    auto *src = FCI->getSrc();
    assert(src->getType()->isQuantizedType() &&
           "Unsupported Type in NMPFullyConnected op");
    auto *weights = FCI->getWeights();
    auto *bias = FCI->getBias();
    auto *dest = FCI->getDest();

    auto *dfxpMode = emitElemKind(builder, dest->getElementType());
    auto *srcPtr = emitValueAddress(builder, src);
    auto *weightsPtr = emitValueAddress(builder, weights);
    auto *biasPtr = emitValueAddress(builder, bias);
    auto *destPtr = emitValueAddress(builder, dest);

    // M[m.n] = A[m.k] * B[n.k] (B in row-major)
    uint16_t out_rows = dest->dims()[0];
    uint16_t out_cols = dest->dims()[1];
    uint16_t src_cols = src->dims()[1];
    assert(src_cols == weights->dims()[1] &&
           "Wrong inner dimension of MatMul on NMPFullyConnected op");
    auto *dest_m = NMPLLVMIRGen::emitConstI16(builder, out_rows);
    auto *dest_n = NMPLLVMIRGen::emitConstI16(builder, out_cols);
    auto *gemm_k = NMPLLVMIRGen::emitConstI16(builder, src_cols);

    // Activation
    auto *act_type = emitFusedActivationType(builder, FCI);

    // Q-points
    auto *srcTy = src->getType();
    auto *destTy = dest->getType();
    auto *weightsTy = weights->getType();
    auto *biasTy = bias->getType();

    int16_t srcQValue = log2(1 / srcTy->getScale());
    int16_t weightsQValue = log2(1 / weightsTy->getScale());
    int16_t biasQValue = log2(1 / biasTy->getScale());
    int16_t destILScale = srcQValue - log2(1 / destTy->getScale());

    auto *srcQP = LLVMIRGen::emitConstI16(builder, srcQValue);
    auto *weightsQP = LLVMIRGen::emitConstI16(builder, weightsQValue);
    auto *biasQP = LLVMIRGen::emitConstI16(builder, biasQValue);
    auto *outILScale = LLVMIRGen::emitConstI16(builder, destILScale);

    // Load-mode
    auto *lMode = LLVMIRGen::emitConstI16(builder, NMP_ELMSIZE_SAME);

    // Mapper's config
    uint16_t numIters = out_cols;
    uint16_t numRowsPerIter{1};
    uint16_t numRowsPerIterRemaining{1};
    if (out_cols > NMP_NUM_TLTS) {
      uint16_t dataSize =
          (dest->getElementType() == ElemKind::Int16QTy) ? 2 : 1;
      uint16_t maxNumberOfRows = floor(NMP_MBLOB_SIZE / (src_cols * dataSize));
      if (maxNumberOfRows > out_cols) {
        maxNumberOfRows = out_cols;
      }
      for (uint16_t i = maxNumberOfRows; i > 1; i--) {
        numRowsPerIter = i;
        numIters = floor(out_cols / numRowsPerIter);
        if (numIters >= NMP_NUM_TLTS) {
          break;
        }
      }
      numRowsPerIterRemaining = out_cols - (numRowsPerIter * numIters);
      if (numRowsPerIterRemaining != 0) {
        numIters++;
      } else {
        numRowsPerIterRemaining = numRowsPerIter;
      }
    }
    auto *_num_iters = LLVMIRGen::emitConstI32(builder, numIters);
    auto *_num_rows = LLVMIRGen::emitConstI32(builder, numRowsPerIter);
    auto *_num_rows_rem =
        LLVMIRGen::emitConstI32(builder, numRowsPerIterRemaining);
    auto *_num_tle = LLVMIRGen::emitConstI16(builder, NMP_NUM_TLE);
    auto *_num_tlt = LLVMIRGen::emitConstI16(builder, NMP_NUM_TLT);

    auto *F = getFunction("matmul_layer");
    createCall(builder, F,
               {dfxpMode, srcPtr, weightsPtr, biasPtr, destPtr, dest_m, dest_n,
                gemm_k, act_type, srcQP, weightsQP, biasQP, outILScale,
                lMode, _num_iters, _num_rows, _num_rows_rem, _num_tle, _num_tlt});
    return 1;
  }

  /* case Kinded::Kind::RescaleQuantizedInstKind: { */
  /*   auto *RQI = cast<RescaleQuantizedInst>(I); */
  /*   auto *src = RQI->getSrc(); */
  /*   assert(src->getType()->isQuantizedType() && */
  /*          "Unsupported Type in RescaleQuantized op"); */
  /*   auto *dest = RQI->getDest(); */

  /*   auto *dfxpMode = emitElemKind(builder, dest->getElementType()); */
  /*   auto *srcPtr = emitValueAddress(builder, src); */
  /*   auto *destPtr = emitValueAddress(builder, dest); */
  /*   // Use the (original) Input Address to create the alpha const == 1
  /*   // and scale it according dataSize (see below) on `model`.cpp
  /*   auto *alphaPtr = NMPLLVMIRGen::emitConstI32(builder, InputAddr); */
  /*   auto *betaPtr = NMPLLVMIRGen::emitConstI32(builder, 0); */

  /*   uint16_t src_c = src->dims()[0]; */
  /*   uint16_t src_h = src->dims()[1]; */
  /*   uint16_t src_w = src->dims()[2]; */
  /*   auto *ifm_c = NMPLLVMIRGen::emitConstI16(builder, src_c); */
  /*   auto *ifm_h = NMPLLVMIRGen::emitConstI16(builder, src_h); */
  /*   auto *ifm_w = NMPLLVMIRGen::emitConstI16(builder, src_w); */

  /*   auto *numAxis = NMPLLVMIRGen::emitConstI16(builder, 1); */
  /*   auto *axis = NMPLLVMIRGen::emitConstI16(builder, 0); */

  /*   // Activation */
  /*   auto *act_type = emitFusedActivationType(builder, RQI); */

  /*   uint16_t dataSize = (dest->getElementType() == ElemKind::Int16QTy) ? 2 : 1; */

  /*   // Q-points */
  /*   auto *srcTy = src->getType(); */
  /*   auto *destTy = dest->getType(); */
  /*   int16_t srcQValue = log2(1 / srcTy->getScale()); */
  /*   int16_t destILScale = srcQValue - log2(1 / destTy->getScale()); */
  /*   int16_t alphaScale = (dataSize == 2) ? 15 : 7; */

  /*   auto *srcQP = LLVMIRGen::emitConstI16(builder, srcQValue); */
  /*   auto *alphaQP = LLVMIRGen::emitConstI16(builder, alphaScale); */
  /*   auto *biasQP = LLVMIRGen::emitConstI16(builder, 0); */
  /*   auto *outILScale = LLVMIRGen::emitConstI16(builder, destILScale); */

  /*   // Load-mode */
  /*   auto *lMode = LLVMIRGen::emitConstI16(builder, NMP_ELMSIZE_SAME); */

  /*   // Mapper's config */
  /*   uint16_t dp_nx{1}; */
  /*   uint16_t dp_ny = NMP_NUM_TLE; */
  /*   auto *_cnfg_dp_nx = LLVMIRGen::emitConstI32(builder, dp_nx); */
  /*   auto *_cnfg_dp_ny = LLVMIRGen::emitConstI32(builder, dp_ny); */

  /*   uint16_t slice_ifm_w = src_w; */
  /*   uint16_t n_ifm_h_per_tle = floor(src_h / NMP_NUM_TLE); */
  /*   uint16_t n_ifm_h_fit_mblob = floor((ifm_w * dataSize) / NMP_MBLOB_SIZE) */
  /*   uint16_t slice_ifm_h = MIN(n_ifm_h_per_tle, n_ifm_h_fit_mblob) */

  /*   auto *_num_ifm_w = LLVMIRGen::emitConstI16(builder, slice_ifm_w); */
  /*   auto *_num_ifm_h = LLVMIRGen::emitConstI16(builder, slice_ifm_h); */
  /*   auto *_num_tlt = LLVMIRGen::emitConstI16(builder, NMP_NUM_TLT); */

  /*   auto *F = getFunction("scale_layer"); */
  /*   createCall(builder, F, */
  /*              {dfxpMode, srcPtr, destPtr, alphaPtr, betaPtr, ifm_w, ifm_h, ifm_c, */
  /*               numAxis, axis, srcQP, weightsQP, biasQP, outILScale, lMode, */
  /*               _cnfg_dp_nx, _cnfg_dp_ny, _num_ifm_w, _num_ifm_h, _num_tlt}); */
  /*   return 1; */
  /* } */

  case Kinded::Kind::SoftMaxInstKind: {
    auto *SM = cast<SoftMaxInst>(I);
    auto *src = SM->getSrc();
    assert(src->getType()->isQuantizedType() &&
           "Unsupported Type in SoftMax op");

    if (!computeNMPSoftmaxOpt) {
      // Softmax is the last operations (before dequantize).
      // If the operation will not be computed, then the
      // result is the input of the Softmax, not the input
      // in the dequantize operation.
      auto baseActivations = NMP_BASE_DATA_ADDR +
                             allocationsInfo_.constantWeightVarsMemSize_ +
                             allocationsInfo_.mutableWeightVarsMemSize_;
      auto addr = baseActivations + allocationsInfo_.allocatedAddress_[src];

      auto scale = src->getType()->getScale();
      outputAddrScale = std::make_pair(addr, scale);
      return 0;
    }

    auto *dest = SM->getDest();
    auto *srcPtr = emitValueAddress(builder, src);
    auto *destPtr = emitValueAddress(builder, dest);
    auto *dfxpMode = emitElemKind(builder, dest->getElementType());

    // q-points
    auto srcTy = src->getType();
    auto destTy = dest->getType();
    int16_t srcQValue = log2(1 / srcTy->getScale());
    int16_t destILScale = srcQValue - log2(1 / destTy->getScale());
    auto *softmax_q_df = LLVMIRGen::emitConstI16(builder, srcQValue);
    auto *softmax_q_wf = LLVMIRGen::emitConstI16(builder, 0);
    auto *softmax_q_ls = LLVMIRGen::emitConstI16(builder, destILScale);

    // Load-mode
    auto *lMode = LLVMIRGen::emitConstI16(builder, NMP_ELMSIZE_SAME);

    // input size
    uint16_t srcDim = src->dims().size();
    assert((srcDim == 2 || srcDim == 4) &&
           "Wrong number of dimmensions on SoftMax op");
    auto *ifm_dim = LLVMIRGen::emitConstI16(builder, srcDim);
    uint16_t num, channels, rows{1}, cols{1};
    if (srcDim == 4) {
      num = src->dims()[0];
      channels = src->dims()[1];
      rows = 2;
      cols = 2;
    } else {
      num = src->dims()[0];
      channels = src->dims()[1];
    }
    auto *ifm_num = LLVMIRGen::emitConstI16(builder, num);
    auto *ifm_c = LLVMIRGen::emitConstI16(builder, channels);
    auto *ifm_h = LLVMIRGen::emitConstI16(builder, rows);
    auto *ifm_w = LLVMIRGen::emitConstI16(builder, cols);

    // Mapper's config
    auto *_num_tle = LLVMIRGen::emitConstI16(builder, NMP_NUM_TLE);
    auto *_num_tlt = LLVMIRGen::emitConstI16(builder, NMP_NUM_TLT);
    auto *_mb_size = LLVMIRGen::emitConstI16(builder, NMP_MBLOB_SIZE);

    auto *F = getFunction("softmax_layer");
    createCall(builder, F,
               {dfxpMode, srcPtr, destPtr, ifm_dim, ifm_h, ifm_w, ifm_c,
                ifm_num, softmax_q_df, softmax_q_wf, softmax_q_ls, lMode,
                _num_tle, _num_tlt, _mb_size});
    return 1;
  }

  default:
    return 0;
  }
}
