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

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <dirent.h>
#include <fstream>
#include <string>

//#include "CCodeGenerator.h"
#include "Loader.h"
#include "ModelPartitionTuner.h"
#include "glow/Partitioner/PartitionerUtils.h"
//#include "glow/Partitioner/NestPartitioner.h"
#include "glow/Runtime/RuntimeTypes.h"

using namespace glow;
using namespace std;

llvm::cl::OptionCategory compilerCat("Compiler Options");

namespace {
llvm::cl::opt<int> executionType(
    "exe-type",
    llvm::cl::desc("\"0: partition plan loading & code generation, 1: profiling, 2: simple partitioning, 3: NEST partitioning\""),
    llvm::cl::init(0), llvm::cl::cat(compilerCat));

llvm::cl::opt<std::string> profilePath("profile-path",
                                       llvm::cl::desc("Profile path"),
                                       llvm::cl::value_desc("./"),
                                       llvm::cl::cat(compilerCat));

llvm::cl::opt<std::string>
    partitionPlanFile("partition-plan", llvm::cl::desc("Partition plan file"),
                      llvm::cl::value_desc("partitionPlan.yaml"),
                      llvm::cl::cat(compilerCat));

llvm::cl::opt<int> profileMode(
    "profile-mode",
    llvm::cl::desc("\"0: partition plan loading & code generation, 1: profiling, 2: simple partitioning, 3: NEST partitioning\""),
    llvm::cl::init(0), llvm::cl::cat(compilerCat));


llvm::cl::opt<int> partitionExe(
        "partition-exe",
        llvm::cl::desc("\"1: generating an execution file for each partition\""),
        llvm::cl::init(0), llvm::cl::cat(compilerCat));
}

//int glow::partitionProfileMode() { return profileMode; }
//size_t exeType = 3;
int main(int argc, char **argv) {
  std::cout << "= model-partition-tuner = " << std::endl;
  // Parse command line parameters. All the options will be available as part of
  // the loader object.
  parseCommandLine(argc, argv);

  // Emit bundle flag should be true.
  //  CHECK(emittingBundle())
  //      << "Bundle output directory not provided. Use the -emit-bundle option!";

  Loader loader;
  loader.loadModel();

  //  std::cout << "profile path = " << profilePath << std::endl;
  // std::cout << "execution type = " << executionType << std::endl;
  //    std::cout << "profileMode = " << profileMode << std::endl;
  if(executionType == 0) {
    std::cout << "partition plan name = " << partitionPlanFile << std::endl;

  }
  std::string funcName = loader.getFunction()->getName();
//  std::cout << "funcName = " << funcName << std::endl;
  if (funcName.find('/') != std::string::npos) {
    std::string newFuncName =
        funcName.substr(funcName.find_last_of("/") + 1);
//    std::cout << "modified func name = " << newFuncName << std::endl;
    loader.getFunction()->setName(newFuncName);
  }

  CompilationContext cctx = loader.getCompilationContext();
  //0: generate compiled partition
  loader.compileForNestPartition(cctx, executionType, profilePath, partitionPlanFile, profileMode, partitionExe);

  return 0;
}
