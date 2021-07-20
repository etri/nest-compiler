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

llvm::cl::opt<int> ospartitionmaintype(
            "os-partition-main-type",
            llvm::cl::desc("\"0: main code local 1: main code nest-os\""),
            llvm::cl::init(0), llvm::cl::cat(compilerCat));


llvm::cl::opt<int> partitionExe(
        "partition-exe",
        llvm::cl::desc("\"1: generating an execution file for each partition\""),
        llvm::cl::init(0), llvm::cl::cat(compilerCat));
}
int os_partition_main_type;
//int glow::partitionProfileMode() { return profileMode; }
//size_t exeType = 3;
int main(int argc, char **argv) {
  std::cout << "= nestos-paritioner = " << std::endl;

  //-------Profiler-----
  // Generate the execution time of each node on available PEs
  // Generate the execution time of fused nodes on available PEs
  //------------

  //---------Tuner-------
  //------------
  // Load a PE description file (config file 1)
  // Load a configuration file specifying fusalbe operations per all PEs (config file 2)
  //------------

  //------------
  // Load profile into the partitionProfileInfo (config file 3)
  //------------

  //------------
  // Find optimal allocation
  // - We consider that all the execution time of an operation and fused operation is previously examined on all available PEs
  //------------

  // Parse command line parameters. All the options will be available as part of
  // the loader object.
  parseCommandLine(argc, argv);

  // Emit bundle flag should be true.
  //  CHECK(emittingBundle())
  //      << "Bundle output directory not provided. Use the -emit-bundle option!";

  Loader loader;  //create loader
  loader.loadModel();

    std::cout << "profile path = " << profilePath << std::endl;
    std::cout << "execution type = " << executionType << std::endl;
    std::cout << "profileMode = " << profileMode << std::endl;
    std::cout << "ospartitionmaintype = " << ospartitionmaintype << std::endl;
    os_partition_main_type = ospartitionmaintype;
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
