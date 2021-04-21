#include "Relay.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"
#include <cstring>
#include <fstream>
#include <vector>

using namespace glow;


void Relay::save(Function *F, llvm::StringRef outputDir,
               llvm::StringRef bundleName,
               llvm::StringRef mainEntryName) const {
  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());
  for (const auto &I : IR->getInstrs()) {
    std::cout<<"SAVE INSTRUCTION FOR "<<I.getKindName()<<std::endl;
  }

  return;
}


