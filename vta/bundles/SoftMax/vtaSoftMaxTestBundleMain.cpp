#include "vtaSoftMaxTestBundle.h"
#include <vector>
#include <string>
#include <assert.h>
#include <inttypes.h>
#include <cstring>
#include <fstream>
#include <iostream>

#include "vta/runtime.h"

//===----------------------------------------------------------------------===//
//                   Image processing helpers
//===----------------------------------------------------------------------===//
std::vector<std::string> inputImageFilenames;

/// Loads and normalizes all PNGs into a tensor memory block \p resultT in the
/// NCHW 3x224x224 format.
static void loadImagesAndPreprocess(const std::vector<std::string> &filenames,
                                    int8_t *&resultT, size_t *resultDims) {
  assert(filenames.size() == 1 &&
         "There must be ONLY one filename in filenames");
  std::pair<float, float> range = std::make_pair(0., 1.0);
  unsigned numImages = filenames.size();
  assert(numImages==1);

  // N x H X W X C
  resultDims[0] = 1;
  resultDims[1] = 1000;
  resultDims[2] = 1;
  resultDims[3] = 1;
  size_t resultSizeInBytes =
          1 * 1000 * sizeof(float);

  std::ifstream fis( filenames[0], std::ios::in | std::ios::binary);
  fis.seekg(0, std::ios::end);
  long size = fis.tellg();
  assert(size==resultSizeInBytes);

  fis.seekg(0, std::ios::beg);
  resultT = static_cast<int8_t *>(malloc(size));

  //int8_t* buf_i = new int8_t[size];
  fis.read((char *)resultT, size);
  fis.close();

  printf("Loaded input size in bytes is: %lu\n", size);
}

/// Parse images file names into a vector.
void parseCommandLineOptions(int argc, char **argv) {
  int arg = 1;
  while (arg < argc) {
    inputImageFilenames.push_back(argv[arg++]);
  }
}

//===----------------------------------------------------------------------===//
//                 Wrapper code for executing a bundle
//===----------------------------------------------------------------------===//
/// Find in the bundle's symbol table a weight variable whose name starts with
/// \p name.
const SymbolTableEntry *getWeightVar(const BundleConfig &config,
                                     const char *name) {
  for (unsigned i = 0, e = config.numSymbols; i < e; ++i) {
    if (!strncmp(config.symbolTable[i].name, name, strlen(name))) {
      return &config.symbolTable[i];
    }
  }
  return nullptr;
}

/// Find in the bundle's symbol table a mutable weight variable whose name
/// starts with \p name.
const SymbolTableEntry &getMutableWeightVar(const BundleConfig &config,
                                            const char *name) {
  const SymbolTableEntry *mutableWeightVar = getWeightVar(config, name);
  if (!mutableWeightVar) {
    printf("Expected to find variable '%s'\n", name);
  }
  assert(mutableWeightVar && "Expected to find a mutable weight variable");
  assert(mutableWeightVar->kind != 0 &&
         "Weight variable is expected to be mutable");
  return *mutableWeightVar;
}

/// Allocate an aligned block of memory.
void *alignedAlloc(const BundleConfig &config, size_t size) {
  void *ptr;
  // Properly align the memory region.
  int res = posix_memalign(&ptr, config.alignment, size);
  assert(res == 0 && "posix_memalign failed");
  assert((size_t)ptr % config.alignment == 0 && "Wrong alignment");
  memset(ptr, 0, size);
  (void)res;
  return ptr;
}

/// Initialize the constant weights memory block by loading the weights from the
/// weights file.
static int8_t *initConstantWeights(const char *weightsFileName,
                                   const BundleConfig &config) {
  // Load weights.
  FILE *weightsFile = fopen(weightsFileName, "rb");
  if (!weightsFile) {
    fprintf(stderr, "Could not open the weights file: %s\n", weightsFileName);
    exit(1);
  }
  fseek(weightsFile, 0, SEEK_END);
  size_t fileSize = ftell(weightsFile);
  fseek(weightsFile, 0, SEEK_SET);
  int8_t *baseConstantWeightVarsAddr =
          static_cast<int8_t *>(alignedAlloc(config, fileSize));
  printf("Allocated weights of size: %lu\n", fileSize);
  printf("Expected weights of size: %" PRIu64 "\n",
         config.constantWeightVarsMemSize);
  assert(fileSize == config.constantWeightVarsMemSize &&
         "Wrong weights file size");
  int result = fread(baseConstantWeightVarsAddr, fileSize, 1, weightsFile);
  if (result != 1) {
    perror("Could not read the weights file");
  } else {
    printf("Loaded weights of size: %lu from the file %s\n", fileSize,
           weightsFileName);
  }
  fclose(weightsFile);
  return baseConstantWeightVarsAddr;
}

/// The assumed layout of the area for mutable WeightVars is:
/// data | gpu_0/data | results
static int8_t *allocateMutableWeightVars(const BundleConfig &config) {
  auto *weights = static_cast<int8_t *>(
          alignedAlloc(config, config.mutableWeightVarsMemSize));
  printf("Allocated mutable weight variables of size: %" PRIu64 "\n",
         config.mutableWeightVarsMemSize);
  return weights;
}

/// Dump the result of the inference by looking at the results vector and
/// finding the index of the max element.
static void dumpInferenceResults(const BundleConfig &config,
                                 int8_t *mutableWeightVars) {
  const SymbolTableEntry &outputWeights =
          getMutableWeightVar(config, "outP");
  int8_t *results = mutableWeightVars + outputWeights.offset;

  std::ofstream fos("output.bin", std::ios::out | std::ios::binary);
  fos.write(reinterpret_cast<char*>(results), 1000*4);
  fos.close();
}

/// The assumed layout of the area for mutable WeightVars is:
/// data | gpu_0/data | results
static int8_t *initMutableWeightVars(const BundleConfig &config) {
  int8_t *mutableWeightVarsAddr = allocateMutableWeightVars(config);
  size_t inputDims[4];
  int8_t *inputT{nullptr};
  loadImagesAndPreprocess(inputImageFilenames, inputT, inputDims);
  // Copy image data into the gpu_0/data input variable in the
  // mutableWeightVars area.
  size_t imageDataSizeInBytes =
          inputDims[0] * inputDims[1] * inputDims[2] * inputDims[3] * sizeof(float);
  printf("Copying image data into mutable weight vars: %lu bytes\n",
         imageDataSizeInBytes);
  const SymbolTableEntry &inputGPUDataVar =
          getMutableWeightVar(config, "input");
  memcpy(mutableWeightVarsAddr + inputGPUDataVar.offset, inputT,
         imageDataSizeInBytes);
  return mutableWeightVarsAddr;
}

static int8_t *initActivations(const BundleConfig &config) {
  return static_cast<int8_t *>(
          alignedAlloc(config, config.activationsMemSize));
}


int main(int argc, char **argv) {
  parseCommandLineOptions(argc, argv);
  // Allocate and initialize constant and mutable weights.
  int8_t *constantWeightVarsAddr =
          initConstantWeights("vtaSoftMaxTestBundle.weights.bin", vtaSoftMaxTestBundle_config);
  int8_t *mutableWeightVarsAddr = initMutableWeightVars(vtaSoftMaxTestBundle_config);
  int8_t *activationsAddr = initActivations(vtaSoftMaxTestBundle_config);

  // Perform the computation.
  int errCode =
          vtaSoftMaxTestMainEntry(constantWeightVarsAddr, mutableWeightVarsAddr, activationsAddr);
  if (errCode != GLOW_SUCCESS) {
    printf("Error running bundle: error code %d\n", errCode);
  }

  // Report the results.
  dumpInferenceResults(vtaSoftMaxTestBundle_config, mutableWeightVarsAddr);

  // Free all resources.
  free(activationsAddr);
  free(constantWeightVarsAddr);
  free(mutableWeightVarsAddr);
}