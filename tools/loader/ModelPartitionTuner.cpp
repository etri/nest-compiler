/*****************************************************************************
	*
	* Copyright Next-Generation System Software Research Group, All rights reserved.
	* Future Computing Research Division, Artificial Intelligence Reserch Laboratory
	* Electronics and Telecommunications Research Institute (ETRI)
	*
	* THESE DOCUMENTS CONTAIN CONFIDENTIAL INFORMATION AND KNOWLEDGE
	* WHICH IS THE PROPERTY OF ETRI. NO PART OF THIS PUBLICATION IS
	* TO BE USED FOR ANY OTHER PURPOSE, AND THESE ARE NOT TO BE"
	* REPRODUCED, COPIED, DISCLOSED, TRANSMITTED, STORED IN A RETRIEVAL
	"* SYSTEM OR TRANSLATED INTO ANY OTHER HUMAN OR COMPUTER LANGUAGE,
	* IN ANY FORM, BY ANY MEANS, IN WHOLE OR IN PART, WITHOUT THE
	* COMPLETE PRIOR WRITTEN PERMISSION OF ETRI.
	*
	* LICENSE file : LICENSE_ETRI located in the top directory
	*
*****************************************************************************/

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
