/*****************************************************************************
*
* Copyright Next-Generation System Software Research Group, All rights reserved.
* Future Computing Research Division, Artificial Intelligence Reserch Laboratory
* Electronics and Telecommunications Research Institute (ETRI)
*
* THESE DOCUMENTS CONTAIN CONFIDENTIAL INFORMATION AND KNOWLEDGE
* WHICH IS THE PROPERTY OF ETRI. NO PART OF THIS PUBLICATION IS
* TO BE USED FOR ANY OTHER PURPOSE, AND THESE ARE NOT TO BE
* REPRODUCED, COPIED, DISCLOSED, TRANSMITTED, STORED IN A RETRIEVAL
* SYSTEM OR TRANSLATED INTO ANY OTHER HUMAN OR COMPUTER LANGUAGE,
* IN ANY FORM, BY ANY MEANS, IN WHOLE OR IN PART, WITHOUT THE
* COMPLETE PRIOR WRITTEN PERMISSION OF ETRI.
*
* LICENSE file : LICENSE_ETRI located in the top directory
*
*****************************************************************************/

#include "multievtaConvTestMainEntry0.h"
#include "multievtaConvTestMainEntry1.h"
#include "multievtaConvTestMainEntry2.h"
#include "multievtaConvTestMainEntry3.h"
#include <vector>
#include <string>
#include <assert.h>
#include <inttypes.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include "VTARuntime.h"
#include "vta/runtime.h"
#include <pthread.h>

#define DEFAULT_HEIGHT 14
#define DEFAULT_WIDTH 14
#define DEFAULT_CHANNEL 32
#define DEFAULT_OHEIGHT 14
#define DEFAULT_OWIDTH 14
#define DEFAULT_OCHANNEL 32

//===----------------------------------------------------------------------===//
//                   Image processing helpers
//===----------------------------------------------------------------------===//
std::vector<std::string> inputImageFilenames;

/// Loads and normalizes all PNGs into a tensor memory block \p resultT in the
/// NCHW 3x224x224 format.
static void loadImagesAndPreprocess(char* inputfilename,
                                    int8_t *&resultT, size_t *resultDims) {
  // N x H X W X C
  resultDims[0] = 1;
  resultDims[1] = DEFAULT_HEIGHT;
  resultDims[2] = DEFAULT_WIDTH;
  resultDims[3] = DEFAULT_CHANNEL;
  size_t resultSizeInBytes =
          1 * DEFAULT_CHANNEL * DEFAULT_HEIGHT * DEFAULT_WIDTH * sizeof(int8_t);

  std::ifstream fis( inputfilename, std::ios::in | std::ios::binary);
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
                                 int8_t *mutableWeightVars,
                                 char* outputname) {
  const SymbolTableEntry &outputWeights =
          getMutableWeightVar(config, "outP");
  int8_t *results = mutableWeightVars + outputWeights.offset;
  int batch_size = 1;
  int filter_size = DEFAULT_OCHANNEL;
  int out_h = DEFAULT_OHEIGHT;
  int out_w = DEFAULT_OWIDTH;

  int8_t resized_out[batch_size][filter_size][out_h][out_w];

  for (int b = 0; b < batch_size; b++){
    for (int f = 0; f < filter_size; f++){
      for (int i = 0; i<out_h; i++){	//h
        for (int j = 0; j < out_w; j++){	//w

          int out_ch = f / filter_size;
          int oo = f % filter_size;

          int8_t v = *(results+(out_ch*out_h*out_w*filter_size + i*out_w*filter_size + j*filter_size + oo));
          resized_out[b][f][i][j] = v;
        }

      }
    }
  }



  std::ofstream fos(outputname, std::ios::out | std::ios::binary);
  int write_cnt = 0;
  int16_t data16 = 0;
  for (int b = 0; b < batch_size; b++){
    for (int i = 0; i<out_h; i++){	//h
      for (int j = 0; j < out_w; j++){	//w
        for (int f = 0; f < filter_size; f++){


          if(write_cnt%2 == 0)
          {
            data16 = 0xff & resized_out[b][f][i][j];
          }
          else{
            data16 = data16 | resized_out[b][f][i][j]<<8;
            fos.write((const char *)&data16, 2);
          }
          write_cnt ++;
        }

      }
    }
  }
  fos.close();
}

/// The assumed layout of the area for mutable WeightVars is:
/// data | gpu_0/data | results
static int8_t *initMutableWeightVars(char* inputfilename, const BundleConfig &config) {
  int8_t *mutableWeightVarsAddr = allocateMutableWeightVars(config);
  size_t inputDims[4];
  int8_t *inputT{nullptr};
  loadImagesAndPreprocess(inputfilename, inputT, inputDims);
  // Copy image data into the gpu_0/data input variable in the
  // mutableWeightVars area.
  size_t imageDataSizeInBytes =
          inputDims[0] * inputDims[1] * inputDims[2] * inputDims[3] * sizeof(int8_t);
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


void *evta0(void *arg) {
  //parseCommandLineOptions(argc, argv);
  initVTARuntime();
  // Allocate and initialize constant and mutable weights.
  int8_t *constantWeightVarsAddr =
          initConstantWeights("multievtaConvTestBundle0.weights.bin", multievtaConvTestBundle0_config);
  int8_t *mutableWeightVarsAddr = initMutableWeightVars("multievtaConvTestInput0.bin", multievtaConvTestBundle0_config);
  int8_t *activationsAddr = initActivations(multievtaConvTestBundle0_config);

  multievtaConvTestMainEntry0_load_module((uint8_t*)constantWeightVarsAddr);

  // Perform the computation.
  for(int i = 0; i< 10000; i++) {
    int errCode =
        multievtaConvTestMainEntry0((uint8_t *) constantWeightVarsAddr,
                                    (uint8_t *) mutableWeightVarsAddr,
                                    (uint8_t *) activationsAddr);

    if (errCode != GLOW_SUCCESS) {
      printf("Error running bundle: error code %d\n", errCode);
    }
  }
  // Report the results.
  dumpInferenceResults(multievtaConvTestBundle0_config, mutableWeightVarsAddr, "output.bin");

  multievtaConvTestMainEntry0_destroy_module();
  // Free all resources.
  free(activationsAddr);
  free(constantWeightVarsAddr);
  free(mutableWeightVarsAddr);
  destroyVTARuntime();
  return 0;
}

void *evta1(void *arg) {
  //parseCommandLineOptions(argc, argv);
  initVTARuntime1();
  // Allocate and initialize constant and mutable weights.
  int8_t *constantWeightVarsAddr =
      initConstantWeights("multievtaConvTestBundle1.weights.bin", multievtaConvTestBundle1_config);
  int8_t *mutableWeightVarsAddr = initMutableWeightVars("multievtaConvTestInput1.bin", multievtaConvTestBundle1_config);
  int8_t *activationsAddr = initActivations(multievtaConvTestBundle1_config);

  multievtaConvTestMainEntry1_load_module((uint8_t*)constantWeightVarsAddr);

  // Perform the computation.
  for(int i = 0; i< 10000; i++) {
    int errCode =
        multievtaConvTestMainEntry1((uint8_t *) constantWeightVarsAddr,
                                    (uint8_t *) mutableWeightVarsAddr,
                                    (uint8_t *) activationsAddr);

    if (errCode != GLOW_SUCCESS) {
      printf("Error running bundle: error code %d\n", errCode);
    }
  }
  // Report the results.
  dumpInferenceResults(multievtaConvTestBundle1_config, mutableWeightVarsAddr, "multievtaConvTestOutput1.bin");

  multievtaConvTestMainEntry1_destroy_module();
  // Free all resources.
  free(activationsAddr);
  free(constantWeightVarsAddr);
  free(mutableWeightVarsAddr);
  destroyVTARuntime1();
  return 0;
}

void *evta2(void *arg) {
  //parseCommandLineOptions(argc, argv);
  initVTARuntime2();
  // Allocate and initialize constant and mutable weights.
  int8_t *constantWeightVarsAddr =
      initConstantWeights("multievtaConvTestBundle2.weights.bin", multievtaConvTestBundle2_config);
  int8_t *mutableWeightVarsAddr = initMutableWeightVars("multievtaConvTestInput2.bin", multievtaConvTestBundle2_config);
  int8_t *activationsAddr = initActivations(multievtaConvTestBundle2_config);

  multievtaConvTestMainEntry2_load_module((uint8_t*)constantWeightVarsAddr);

  // Perform the computation.
  for(int i = 0; i< 10000; i++) {
    int errCode =
        multievtaConvTestMainEntry2((uint8_t *) constantWeightVarsAddr,
                                    (uint8_t *) mutableWeightVarsAddr,
                                    (uint8_t *) activationsAddr);
    if (errCode != GLOW_SUCCESS) {
      printf("Error running bundle: error code %d\n", errCode);
    }
  }
  // Report the results.
  dumpInferenceResults(multievtaConvTestBundle2_config, mutableWeightVarsAddr, "multievtaConvTestOutput2.bin");

  multievtaConvTestMainEntry2_destroy_module();
  // Free all resources.
  free(activationsAddr);
  free(constantWeightVarsAddr);
  free(mutableWeightVarsAddr);
  destroyVTARuntime2();
  return 0;
}

void *evta3(void *arg) {
  //parseCommandLineOptions(argc, argv);
  initVTARuntime3();
  // Allocate and initialize constant and mutable weights.
  int8_t *constantWeightVarsAddr =
      initConstantWeights("multievtaConvTestBundle3.weights.bin", multievtaConvTestBundle3_config);
  int8_t *mutableWeightVarsAddr = initMutableWeightVars("multievtaConvTestInput3.bin", multievtaConvTestBundle3_config);
  int8_t *activationsAddr = initActivations(multievtaConvTestBundle3_config);

  multievtaConvTestMainEntry3_load_module((uint8_t*)constantWeightVarsAddr);

  // Perform the computation.
  for(int i = 0; i< 10000; i++) {
    int errCode =
        multievtaConvTestMainEntry3((uint8_t *) constantWeightVarsAddr,
                                    (uint8_t *) mutableWeightVarsAddr,
                                    (uint8_t *) activationsAddr);
    if (errCode != GLOW_SUCCESS) {
      printf("Error running bundle: error code %d\n", errCode);
    }
  }
  // Report the results.
  dumpInferenceResults(multievtaConvTestBundle3_config, mutableWeightVarsAddr, "multievtaConvTestOutput3.bin");

  multievtaConvTestMainEntry3_destroy_module();
  // Free all resources.
  free(activationsAddr);
  free(constantWeightVarsAddr);
  free(mutableWeightVarsAddr);
  destroyVTARuntime3();
  return 0;
}

int main(){
  pthread_t thread[4];

  pthread_create(&thread[0], NULL, evta0, (void*) nullptr);
  pthread_create(&thread[1], NULL, evta1, (void*) nullptr);
  pthread_create(&thread[2], NULL, evta2, (void*) nullptr);
  pthread_create(&thread[3], NULL, evta3, (void*) nullptr);
  int status;
  pthread_join(thread[0], (void **)&status);
  pthread_join(thread[1], (void **)&status);
  pthread_join(thread[2], (void **)&status);
  pthread_join(thread[3], (void **)&status);
  return 0;
}