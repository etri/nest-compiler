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
 * @file: NMPBundleSaver.cpp
 * @brief description: Save NMP model object & header files to bundle folder.
 * @date: 11 18, 2021
 */

#include "glow/LLVMIRCodeGen/CommandLine.h"

#include "NMPBackend.h"
#include "NMPBundleSaver.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Debug.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <glog/logging.h>

#define DEBUG_TYPE "jit"

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

// NMP Header file string template.
static const char *headerFileNMP = R"RAW(%s
#ifndef _GLOW_BUNDLE_%s_H
#define _GLOW_BUNDLE_%s_H

#include <stdint.h>

// ---------------------------------------------------------------
//                       Common definitions
// ---------------------------------------------------------------
#ifndef _GLOW_BUNDLE_COMMON_DEFS
#define _GLOW_BUNDLE_COMMON_DEFS

// Glow bundle error code for correct execution.
#define GLOW_SUCCESS 0

#endif

// ---------------------------------------------------------------
//                       LNE definitions
// ---------------------------------------------------------------
#ifndef _GLOW_BUNDLE_LNE_DEFS
#define _GLOW_BUNDLE_LNE_DEFS

#define	LNE_DEVICE_NODE	        "/dev/dq1_lne"
#define LNE_IOCTL_MAGIC         'L'
#define LNE_IOCTL_CMD_START     _IO(LNE_IOCTL_MAGIC, 0)
#define LNE_IOCTL_CMD_STOP      _IO(LNE_IOCTL_MAGIC, 1)
#define LNE_IOCTL_CMD_RESET     _IO(LNE_IOCTL_MAGIC, 2)

#define LNE_TEXT_ADDRESS        0x200
#define LNE_DATA_ADDRESS        0x01000000

#endif

// ---------------------------------------------------------------
//                          Bundle API
// ---------------------------------------------------------------
%s
%s
#endif
)RAW";

// Function to print the NMP header file using the template.
static void printNMPHeader(llvm::StringRef headerFileName,
                           llvm::StringRef bundleName,
                           llvm::StringRef modelInfo,
                           llvm::StringRef modelApi) {
  std::error_code EC;
  llvm::raw_fd_ostream headerFile(headerFileName, EC,
                                  llvm::sys::fs::OpenFlags::F_Text);
  CHECK(!EC) << "Could not open header file!";
  std::string header;
  header += "// Bundle API auto-generated header file. Do not edit!\n";
#ifdef GLOW_VERSION
  header += "// Glow Tools version: " + std::string(GLOW_VERSION) + "\n";
#endif
  headerFile << strFormat(headerFileNMP, header.c_str(),
                          bundleName.upper().data(), bundleName.upper().data(),
                          modelInfo.data(), modelApi.data());
  headerFile.close();
}

// Utility function to serialize a binary file to text file as a C array.
static void serializeNMPBinToText(llvm::StringRef binFileName,
                                  llvm::StringRef txtFileName) {
  FILE *inpFile = fopen(binFileName.str().c_str(), "rb");
  CHECK(inpFile) << "Could not open binary input file: " << binFileName.str();
  FILE *outFile = fopen(txtFileName.str().c_str(), "w");
  CHECK(outFile) << "Could not open text output file: " << txtFileName.str();
  const size_t numBytesPerLine = 20;
  for (size_t i = 0;; i++) {
    int ch = fgetc(inpFile);
    if (ch == EOF) {
      break;
    }
    fprintf(outFile, " 0X%02X,", ch);
    if ((i % numBytesPerLine) == (numBytesPerLine - 1)) {
      fprintf(outFile, "\n");
    }
  }
  fprintf(outFile, "\n");
  fclose(inpFile);
  fclose(outFile);
}

NMPBundleSaver::NMPBundleSaver(const NMPBackend &nmpBackend,
                               llvm::StringRef outputDir,
                               llvm::StringRef bundleName)
    : BundleSaver(nmpBackend, outputDir, bundleName) {}

void NMPBundleSaver::produceBundle() {
  DCHECK(!isSaved_) << "produceBundle can be invoked only once";
  isSaved_ = true;
  // Emit entry functions.
  for (auto &savedFunction : savedIRFunctions_) {
    emitBundleEntryFunction(savedFunction);
  }
  // Finish code generation.
  irgen_->finishCodeGen();
  setIRFunction("<noname>", nullptr);
  // Emit symbol table and bundle config only for dynamic API
  if (bundleAPI_ == BundleApiType::Dynamic) {
    // Emit the symbol table for weight variables.
    emitSymbolTable();
    // Emit the config for the bundle.
    emitBundleConfig();
  }

  auto &M = irgen_->getModule();
  auto outputDir = irgen_->getOutputDir();
  auto bundleName = irgen_->getBundleName();
  auto savedBundleName = irgen_->getSavedBundleName().empty()
                             ? bundleName
                             : irgen_->getSavedBundleName();
  std::string extension = (llvmCompiler.empty()) ? ".o" : ".bc";
  std::string bundleCodeOutput;
  bundleCodeOutput = (outputDir + "/" + savedBundleName + extension).str();
  auto bundleWeightsBinOut =
      (outputDir + "/" + savedBundleName + ".weights.bin").str();
  auto bundleHeaderOutput = (outputDir + "/" + savedBundleName + ".h").str();
  DEBUG_GLOW(llvm::dbgs() << "Producing a bundle:\n"
                          << "saved bundle name: " << savedBundleName << "\n"
                          << "bundle name: " << bundleName << "\n"
                          << "bundle code: " << bundleCodeOutput << "\n"
                          << "bundle weights:" << bundleWeightsBinOut << "\n"
                          << "header file: " << bundleHeaderOutput << "\n");
  llvm::StringRef fileName = bundleCodeOutput;
  std::error_code EC;
  llvm::raw_fd_ostream outputFile(fileName, EC, llvm::sys::fs::F_None);
  CHECK(!EC) << "Could not open the output file for saving the bundle "
                "code with file name: "
             << fileName.str();
  if (fileName.endswith(".bc")) {
    // Emit the bitcode file.
    llvm::WriteBitcodeToFile(M, outputFile);
    outputFile.flush();
    if (!llvmCompiler.empty()) {
      // Compile bitcode using an external LLVM compiler.
      // The code is optimized twice with the external opt tool.
      std::string cmd = llvmCompiler;
      for (auto option : llvmCompilerOptions) {
        cmd += " " + option + " ";
      }
      cmd += " " + bundleCodeOutput;
      std::string bundleObjectCodeOutputOpt;
      if (!llvmOpt.empty()) {
        bundleObjectCodeOutputOpt =
            " -emit-llvm -o " +
            (outputDir + "/" + savedBundleName + ".beforeopt.bc").str();
      } else {
        bundleObjectCodeOutputOpt =
            " -o " + (outputDir + "/" + savedBundleName + ".o").str();
      }

      cmd += bundleObjectCodeOutputOpt;
      CHECK(!system(cmd.c_str()))
          << "Error running external LLVM compiler: " << cmd;

      // Running opt tool to optimize a second time.
      // TODO: Only run the appropriate passes as needed.
      if (!llvmOpt.empty()) {
        cmd.clear();
        cmd = llvmOpt;
        cmd +=
            " " + (outputDir + "/" + savedBundleName + ".beforeopt.bc").str();
        cmd +=
            " -O3 -o " + (outputDir + "/" + savedBundleName + ".opt.bc").str();
        CHECK(!system(cmd.c_str()))
            << "Error running external opt compiler: " << cmd;

        cmd.clear();
        cmd = llvmCompiler;
        for (auto option : llvmCompilerOptions) {
          cmd += " " + option + " ";
        }
        cmd += " " + (outputDir + "/" + savedBundleName + ".opt.bc").str();
        cmd += " -o " + (outputDir + "/" + savedBundleName + ".o").str();
        CHECK(!system(cmd.c_str()))
            << "Error running external LLVM compiler: " << cmd;
      }
    }
  } else if (fileName.endswith(".o")) {
    // Emit the object file.
    llvm::legacy::PassManager PM;
    auto &TM = irgen_->getTargetMachine();

#if LLVM_VERSION_MAJOR < 10
    TM.addPassesToEmitFile(
        PM, outputFile, nullptr,
        llvm::TargetMachine::CodeGenFileType::CGFT_ObjectFile);
#else
    TM.addPassesToEmitFile(PM, outputFile, nullptr, llvm::CGFT_ObjectFile);
#endif

    PM.run(M);
  }
  outputFile.close();
  // Create bundle archive with additional object files.
  createBundleArchive(fileName, irgen_->getObjectRegistry(),
                      irgen_->getBundleObjects());
  // Output weights.
  saveWeights(bundleWeightsBinOut);
  // Header file.
  saveHeader(bundleHeaderOutput);
  // Save weights also in text format for Static API.
  if (bundleAPI_ == BundleApiType::Static) {
    auto bundleWeightsTxtOut =
        (outputDir + "/" + savedBundleName + ".weights.txt").str();
    serializeNMPBinToText(bundleWeightsBinOut, bundleWeightsTxtOut);
  }
}

// Emit the entry function for the bundle.
// It simply calls the main entry of the module
void NMPBundleSaver::emitBundleEntryFunction(
    BundleSaver::SavedIRFunction &savedF) {
  // The bundle entry point has the following API:
  // int entry();
  llvm::Type *retTy = llvm::Type::getIntNTy(irgen_->getLLVMContext(),
                                            irgen_->getLibjitIntWidth());
  llvm::FunctionType *bundleFuncTy = llvm::FunctionType::get(retTy, {}, false);
  // NMP needs a main function, so We will ignore the entryName
  auto *func =
      llvm::Function::Create(bundleFuncTy, llvm::Function::ExternalLinkage,
                             "main" /* savedF.entryName */, &irgen_->getModule());
  llvm::BasicBlock *entry_bb =
      llvm::BasicBlock::Create(irgen_->getLLVMContext(), "entry", func);
  llvm::IRBuilder<> builder(entry_bb);
  // Add a provisional terminator to make the function well-formed.
  auto *zero = builder.getIntN(irgen_->getLibjitIntWidth(), 0);
  auto *ret = builder.CreateRet(zero);
  builder.SetInsertPoint(ret);

  // Invoke the main inference entry
  auto *entryF = savedF.llvmF;
  entryF->setLinkage(llvm::Function::InternalLinkage);
  auto *result = irgen_->createCall(builder, entryF, {});
  // Terminate the function.
  builder.CreateRet(result);
  // Remove the provisional terminator.
  ret->eraseFromParent();
  // Create the debug info for the bundle entry point function.
  irgen_->generateFunctionDebugInfo(func);
}

void NMPBundleSaver::saveHeader(llvm::StringRef headerFileName) {
  auto bundleName = irgen_->getBundleName();
  auto bundleNameUpper = llvm::StringRef(bundleName).upper();
  auto constMemSize = irgen_->getAllocationsInfo().constantWeightVarsMemSize_;
  auto mutableMemSize = irgen_->getAllocationsInfo().mutableWeightVarsMemSize_;
  auto activationsMemSize = irgen_->getAllocationsInfo().activationsMemSize_;
  auto memAlignSize = TensorAlignment;
  auto totMemSize = constMemSize + mutableMemSize + activationsMemSize;

  // Format model description.
  std::string modelInfo = strFormat("// Model name: \"%s\"\n"
                                    "// Total data size: %#lx (bytes)\n",
                                    bundleName.data(), totMemSize);
  // Print placeholders (mandatory).
  modelInfo += "// Placeholders:\n";
  auto placeholders = findPlaceholders();
  for (auto &v : placeholders) {
    auto *w = cast<WeightVar>(getWeightForNode(v));
    // Get placeholder properties.
    auto name = w->getName();
    auto type = w->getType();
    auto typeName = type->toString();
    auto sizeElem = type->size();
    auto sizeByte = type->getSizeInBytes();
    auto offset = allocationsInfo_.allocatedAddress_[w];
    modelInfo += strFormat("//\n"
                           "//   Name: \"%s\"\n"
                           "//   Type: %s\n"
                           "//   Size: %" PRIuDIM " (elements)\n"
                           "//   Size: %zu (bytes)\n"
                           "//   Offset: %lu (bytes)\n",
                           name.data(), typeName.c_str(), sizeElem, sizeByte,
                           (unsigned long)offset);
  }
  // Print constants (optional).
  if (bundleAPIVerbose) {
    modelInfo += "//\n"
                 "// Constants:\n";
    auto constantWeights = findConstantWeights();
    for (auto &weightInfo : constantWeights) {
      auto *w = weightInfo.first;
      // Get constant properties.
      auto name = w->getName();
      auto type = w->getType();
      auto typeName = type->toString();
      auto sizeElem = type->size();
      auto sizeByte = type->getSizeInBytes();
      auto offset = allocationsInfo_.allocatedAddress_[w];
      modelInfo += strFormat("//\n"
                             "//   Name: \"%s\"\n"
                             "//   Type: %s\n"
                             "//   Size: %" PRIuDIM " (elements)\n"
                             "//   Size: %zu (bytes)\n"
                             "//   Offset: %lu (bytes)\n",
                             name.data(), typeName.c_str(), sizeElem, sizeByte,
                             (unsigned long)offset);
    }
  }
  modelInfo += "//";

  std::string modelApi = "\n";
  // Get the names and offsets of the input and output data.
  NMPLLVMIRGen *nmpgen = static_cast<NMPLLVMIRGen*>(irgen_.get());
  auto input = nmpgen->getInputData();
  auto output = nmpgen->getOutputData();
  // Print address offsets & q-points.
  modelApi += "// Placeholder address offsets within mutable buffer (bytes).\n";
  modelApi += strFormat("#define %s_%s  %#x\n", bundleNameUpper.data(), "input_offset",
                        input.first - NMP_BASE_ADDR);
  modelApi += strFormat("#define %s_%s  %#x\n", bundleNameUpper.data(), "output_offset",
                        output.first - NMP_BASE_ADDR);
  modelApi += strFormat("#define %s_%s  %2d\n", bundleNameUpper.data(),
                        "input_qpoint", (int)log2(1 / input.second));
  modelApi += strFormat("#define %s_%s  %2d\n", bundleNameUpper.data(),
                        "output_qpoint", (int)log2(1 / output.second));
  modelApi += "\n";

  // Print memory sizes and memory alignment.
  modelApi +=
      strFormat("// Memory sizes (bytes).\n"
                "#define %s_CONSTANT_MEM_SIZE     %#lx\n"
                "#define %s_MUTABLE_MEM_SIZE      %#lx\n"
                "#define %s_ACTIVATIONS_MEM_SIZE  %#lx\n"
                "\n"
                "// Memory alignment (bytes).\n"
                "#define %s_MEM_ALIGN  %d\n"
                "\n",
                bundleNameUpper.data(), constMemSize, bundleNameUpper.data(),
                mutableMemSize, bundleNameUpper.data(), activationsMemSize,
                bundleNameUpper.data(), memAlignSize);

  // Print header file.
  printNMPHeader(headerFileName, bundleName, modelInfo, modelApi);
}
