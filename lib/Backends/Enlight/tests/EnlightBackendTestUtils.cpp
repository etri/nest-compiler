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

#include "EnlightBackendTestUtils.h"

#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "llvm/Support/CommandLine.h"

#include <future>
#include "Enlight.h"

namespace glow {

using llvm::cast;

namespace {
// Helpers for creating and intializing placeholders from tensors.
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

} // namespace

void inferFCNet(Tensor *inputs, Tensor *filter, Tensor *bias, Tensor *out,
                llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Placeholder *inputP;
  Placeholder *filterP;
  Placeholder *biasP;
  Placeholder *outP;
  TypeRef OT;
  inputP = createPlaceholder(mod, bindings, inputs, "inputP");
  filterP = createPlaceholder(mod, bindings, filter, "filterP");
  biasP = createPlaceholder(mod, bindings, bias, "biasP");
  outP = createPlaceholder(mod, bindings, out, "outP");
  OT = F->getParent()->uniqueType(out->getElementType(), out->dims());

  auto *fc = F->createFullyConnected("fc", inputP, filterP, biasP, OT, 1);

  auto *result = F->createSave("ret", fc, outP);
  F->dumpDAG("fc.dot");
  auto *resultTensor = bindings.get(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings, {inputP, filterP, biasP},
                          {inputs, filter, bias});
  EE.run(bindings);
  out->assign(resultTensor);
}

void inferFCReluNet(Tensor *inputs, Tensor *filter, Tensor *bias, Tensor *out,
                llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Placeholder *inputP;
  Placeholder *filterP;
  Placeholder *biasP;
  Placeholder *outP;
  TypeRef OT;
  inputP = createPlaceholder(mod, bindings, inputs, "inputP");
  filterP = createPlaceholder(mod, bindings, filter, "filterP");
  biasP = createPlaceholder(mod, bindings, bias, "biasP");
  outP = createPlaceholder(mod, bindings, out, "outP");
  OT = F->getParent()->uniqueType(out->getElementType(), out->dims());

  auto *fc = F->createFullyConnected("fc", inputP, filterP, biasP, OT, 1);
  auto *relu = F->createRELU("relu", fc);
  auto *result = F->createSave("ret", relu, outP);

  F->dumpDAG("fc.dot");
  auto *resultTensor = bindings.get(result->getPlaceholder());
  std::unique_ptr<Backend> backend(createBackend("Enlight"));
  CompilationContext cctx;
  auto error = ::glow::optimizeFunction(F, *backend, cctx);
  F->dumpDAG("fc2.dot");
  EE.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings, {inputP, filterP, biasP},
                          {inputs, filter, bias});
  EE.run(bindings);
  out->assign(resultTensor);
}

void inferFCFCNet(Tensor *inputs, Tensor *filter0, Tensor *bias0, Tensor *filter1, Tensor *bias1, Tensor *out0, Tensor *out1,
                llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Placeholder *inputP;
  Placeholder *filter0P;
  Placeholder *bias0P;
  Placeholder *filter1P;
  Placeholder *bias1P;
  Placeholder *outP;
  TypeRef OT0, OT1;
  inputP = createPlaceholder(mod, bindings, inputs, "inputP");
  filter0P = createPlaceholder(mod, bindings, filter0, "filter0P");
  bias0P = createPlaceholder(mod, bindings, bias0, "bias0P");
  filter1P = createPlaceholder(mod, bindings, filter1, "filter1P");
  bias1P = createPlaceholder(mod, bindings, bias1, "bias1P");
  outP = createPlaceholder(mod, bindings, out1, "outP");
  OT0 = F->getParent()->uniqueType(out0->getElementType(), out0->dims());
  OT1 = F->getParent()->uniqueType(out1->getElementType(), out1->dims());

  auto *fc0 = F->createFullyConnected("fc0", inputP, filter0P, bias0P, OT0, 1);
  auto *fc1 = F->createFullyConnected("fc1", fc0, filter1P, bias1P, OT1, 1);
  auto *result = F->createSave("ret", fc1, outP);
  F->dumpDAG("fc.dot");
  auto *resultTensor = bindings.get(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings, {inputP, filter0P, bias0P, filter1P, bias1P},
                          {inputs, filter0, bias0, filter1, bias1});
  EE.run(bindings);
  out1->assign(resultTensor);
}

} // namespace glow
