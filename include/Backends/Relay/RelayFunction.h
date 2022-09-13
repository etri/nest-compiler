/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * Modifications copyright (C) 2022 <ETRI/Yongin Kwon>
 */
#ifndef GLOW_BACKENDS_RELAY_RELAYFUNCTION_H
#define GLOW_BACKENDS_RELAY_RELAYFUNCTION_H

#include "glow/Backend/Backend.h"
#include "glow/Backend/BackendUtils.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/Base/Tensor.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Quantization/Base/Base.h"

#include "llvm/ADT/ArrayRef.h"

#include <memory>
#include <unordered_map>

namespace glow {

class IRFunction;
class Value;
class Tensor;
class Constant;

// Forward declare all of the classes.
#define DEF_VALUE(CLASS, NAME) class CLASS;
#define DEF_INSTR(CLASS, NAME) class CLASS;
DEF_VALUE(WeightVar, weightvar)
DEF_INSTR(AllocActivationInst, allocactivation)
DEF_INSTR(TensorViewInst, tensorview)
DEF_INSTR(DeallocActivationInst, deallocactivation)
DEF_INSTR(CopyInst, copy)
DEF_INSTR(ConvolutionInst, convolution)
DEF_INSTR(MaxPoolInst, maxpool)
DEF_INSTR(AvgPoolInst, avgpool)
DEF_INSTR(FullyConnectedInst, fullyconnected)
DEF_INSTR(LocalResponseNormalizationInst, localresponsenormalization)
DEF_INSTR(SoftMaxInst, softmax)
DEF_INSTR(ElementAddInst, elementadd)
DEF_INSTR(ElementSubInst, elementsub)
DEF_INSTR(ElementMulInst, elementmul)
DEF_INSTR(ElementDivInst, elementdiv)
DEF_INSTR(ReluInst, relu)
DEF_INSTR(TransposeInst, transpose)
DEF_INSTR(TouchInst, touch)
DEF_INSTR(InsertTensorInst, inserttensor)
DEF_INSTR(QuantizeInst, quantize)
DEF_INSTR(DequantizeInst, dequantize)
#undef DEF_VALUE
#undef DEF_INSTR

/// Function "compiled" for execution by the Relay.
class RelayFunction final : public CompiledFunction,
                            public IRInstructionProcessingHandler {
  /// The IR to be executed.
  std::unique_ptr<IRFunction> F_;

  /// Maps Value.name to tensors for constants.
  std::unordered_map<std::string, Tensor *> constants_;

public:
  RelayFunction(std::unique_ptr<IRFunction> F, runtime::RuntimeBundle &&bundle);

  /// \name CompiledFunction interface
  ///@{
  ~RelayFunction() override;

  Error execute(ExecutionContext *context) override;

  /// Collects constants for runtime.
  void collectConstants(const Module *module) override;

  /// Add a constant to the function, this is used for loading static
  /// placeholders.
  void addConstant(std::string name, Tensor *T);

  /// Get reference to IR function.
  IRFunction *getIR() { return F_.get(); }

  /// Read trace events out of this func and write them into /p context
  void translateTraceEvents(ExecutionContext *context) const override;

  /// \returns the backend used to compile this function.
  virtual std::string getCompileBackendName() const override { return "Relay"; }
  ///@}
};

/// An RelayFunction bound to a specific invocation.
class BoundRelayFunction : public IRInstructionProcessingHandler {
  /// Maps values to Tensors, that are owned by this class.
  std::unordered_map<const Value *, Tensor *> tensors_;

  /// Maps values to Tensors, that are *not* owned by this class.
  std::unordered_map<const Value *, Tensor *> externalTensors_;

  /// A reference to the constant map from the owning RelayFunction.
  const std::unordered_map<std::string, Tensor *> &constants_;

public:
  explicit BoundRelayFunction(
      const std::unordered_map<std::string, Tensor *> &constants)
      : constants_(constants) {}

  ~BoundRelayFunction();

  Error execute(IRFunction *F, ExecutionContext *context);

  /// \returns a pointer to the tensor that is saved under \p v.
  Tensor *getTensor(const Value *v) const;

  /// \returns a typed handle to the tensor that is stored at \p v.
  template <class ElemTy = float>
  Handle<ElemTy> getWeightHandle(Value *v) const {
    return getTensor(v)->getHandle<ElemTy>();
  }

private:
  /// Allocate a tensor to back the value \p v. Do not allocate anything if a
  /// tensor is already allocated for \p v.
  /// \returns a tensor for \p v.
  Tensor *getOrCreateTensor(const Value *v);

  /// Allocate an unowned tensor to back the value \p v. The source tensor of
  /// the unowned tensor is provided by \p src.
  /// \returns a tensor for \p v.
  Tensor *getOrCreateUnownedTensor(const Value *v, const Value *src,
                                   llvm::ArrayRef<dim_t> offsets);

  /// If a tensor is allocated for \p v then delete it.
  void deleteTensor(const Value *v);

  /// @name BoundRelayFunction methods. This is a list of method
  /// declerations that are used by the Relay to dispatch different
  /// instructions.
  ///@{

#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME) void fwd##CLASS(const CLASS *I);
  DEF_VALUE(WeightVar, weightvar)
  DEF_INSTR(AllocActivationInst, allocactivation)
  DEF_INSTR(TensorViewInst, tensorview)
  DEF_INSTR(DeallocActivationInst, deallocactivation)
  DEF_INSTR(CopyInst, copy)
  DEF_INSTR(ConvolutionInst, convolution)
  DEF_INSTR(MaxPoolInst, maxpool)
  DEF_INSTR(AvgPoolInst, avgpool)
  DEF_INSTR(FullyConnectedInst, fullyconnected)
  DEF_INSTR(LocalResponseNormalizationInst, localresponsenormalization)
  DEF_INSTR(SoftMaxInst, softmax)
  DEF_INSTR(ElementAddInst, elementadd)
  DEF_INSTR(ElementSubInst, elementsub)
  DEF_INSTR(ElementMulInst, elementmul)
  DEF_INSTR(ElementDivInst, elementdiv)
  DEF_INSTR(ReluInst, relu)
  DEF_INSTR(TransposeInst, transpose)
  DEF_INSTR(TouchInst, touch)
  DEF_INSTR(InsertTensorInst, inserttensor)
  DEF_INSTR(QuantizeInst, quantize)
  DEF_INSTR(DequantizeInst, dequantize)
#undef DEF_VALUE
#undef DEF_INSTR

  ///@}
};

} // end namespace glow

#endif // GLOW_BACKENDS_RELAY_RELAYFUNCTION_H
