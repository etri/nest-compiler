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
 * @file: NMPBundleSaver.h
 * @brief description: Save NMP model object & header files to bundle folder.
 * @date: 11 18, 2021
 */

#ifndef GLOW_NMPIRCODEGEN_BUNDLESAVER_H
#define GLOW_NMPIRCODEGEN_BUNDLESAVER_H

#include "NMPLLVMIRGen.h"
#include "glow/IR/IR.h"
#include "glow/LLVMIRCodeGen/AllocationsInfo.h"
#include "glow/LLVMIRCodeGen/BundleSaver.h"
#include "glow/LLVMIRCodeGen/LLVMBackend.h"
#include "glow/LLVMIRCodeGen/LLVMIRGen.h"

namespace glow {
class NMPBackend;

class NMPBundleSaver : public BundleSaver {
public:
  NMPBundleSaver(const NMPBackend &nmpBackend, llvm::StringRef outputDir,
                 llvm::StringRef bundleName);

  ~NMPBundleSaver() override = default;

  /// Produce a bundle for NMP.
  void produceBundle() override;

protected:
  /// Emit the entry function for the saved function \p savedF.
  void emitBundleEntryFunction(SavedIRFunction &savedF) override;

  /// Save header file for the bundle.
  void saveHeader(llvm::StringRef headerFileName) override;

};

} // namespace glow

#endif // GLOW_NMPIRCODEGEN_BUNDLESAVER_H
