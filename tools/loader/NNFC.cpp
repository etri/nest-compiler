//
// Created by jemin lee on 26/05/2020.
//

#include "Loader.h"
#include "include/glow/Graph/Graph.h"
#include "lib/Graph/Graph.cpp"

using namespace glow;

int main(int argc, char **argv) {

  // Parse command line parameters. All the options will be available as part of
  // the loader object.
  parseCommandLine(argc, argv);

  // Initialize the loader object.
  Loader loader;

  // Emit bundle flag should be true.
  CHECK(emittingBundle())
      << "Bundle output directory not provided. Use the -emit-bundle option!";

  // Load the model.
  loader.loadModel();

  Function *F_= loader.getFunction();

  F_->dumpDAG("model.stcnn");

  FunctionDottyPrinter DP;

  DP.visitGraph(F_);

  //DP.dumpAll(llvm::outs());


  // Compile the model with default options.
  //CompilationContext cctx = loader.getCompilationContext();
  //loader.compile(cctx);

  return 0;
}