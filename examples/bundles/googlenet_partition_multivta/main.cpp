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

//#define VTAMAIN 0

#ifdef VTAMAIN
#include "VTARuntime.h"
#endif

#include <assert.h>
//#include <inttypes.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

//#include "resnet18v2.h"
//#include "mxnet_exported_resnet18.h"
#include "p0.h"
#include "p1.h"
#include "p10.h"
#include "p11.h"
#include "p12.h"
#include "p13.h"
#include "p14.h"
#include "p15.h"
#include "p16.h"
#include "p17.h"
#include "p18.h"
#include "p19.h"
#include "p2.h"
#include "p20.h"
#include "p21.h"
#include "p22.h"
#include "p23.h"
#include "p24.h"
#include "p25.h"
#include "p26.h"
#include "p27.h"
#include "p28.h"
#include "p29.h"
#include "p3.h"
#include "p30.h"
#include "p31.h"
#include "p32.h"
#include "p33.h"
#include "p34.h"
#include "p35.h"
#include "p4.h"
#include "p5.h"
#include "p6.h"
#include "p7.h"
#include "p8.h"
#include "p9.h"

/// This is an example demonstrating how to use auto-generated bundles and
/// create standalone executables that can perform neural network computations.
/// This example loads and runs the compiled resnet50 network model.
/// This example is using the dynamic bundle API (default).

#define DEFAULT_HEIGHT 224
#define DEFAULT_WIDTH 224

//===----------------------------------------------------------------------===//
//                   Image processing helpers
//===----------------------------------------------------------------------===//
std::vector<std::string> inputImageFilenames;

/// \returns the index of the element at x,y,z,w.
size_t getXYZW(const size_t *dims, size_t x, size_t y, size_t z, size_t w) {
  return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
         (z * dims[3]) + w;
}

/// \returns the index of the element at x,y,z.
size_t getXYZ(const size_t *dims, size_t x, size_t y, size_t z) {
  return (x * dims[1] * dims[2]) + (y * dims[2]) + z;
}

/// Reads a PNG image from a file into a newly allocated memory block \p imageT
/// representing a WxHxNxC tensor and returns it. The client is responsible for
/// freeing the memory block.
bool readPngImage(const char *filename, std::pair<float, float> range,
                  float *&imageT, size_t *imageDims) {
  unsigned char header[8];
  // open file and test for it being a png.
  FILE *fp = fopen(filename, "rb");
  // Can't open the file.
  if (!fp) {
    return true;
  }

  // Validate signature.
  size_t fread_ret = fread(header, 1, 8, fp);
  if (fread_ret != 8) {
    return true;
  }
  if (png_sig_cmp(header, 0, 8)) {
    return true;
  }

  // Initialize stuff.
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    return true;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    return true;
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);

  size_t width = png_get_image_width(png_ptr, info_ptr);
  size_t height = png_get_image_height(png_ptr, info_ptr);
  int color_type = png_get_color_type(png_ptr, info_ptr);
  int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  const bool isGray = color_type == PNG_COLOR_TYPE_GRAY;
  const size_t numChannels = isGray ? 1 : 3;

  (void)bit_depth;
  assert(bit_depth == 8 && "Invalid image");
  assert((color_type == PNG_COLOR_TYPE_RGB_ALPHA ||
          color_type == PNG_COLOR_TYPE_RGB || isGray) &&
         "Invalid image");
  bool hasAlpha = (color_type == PNG_COLOR_TYPE_RGB_ALPHA);

  int number_of_passes = png_set_interlace_handling(png_ptr);
  (void)number_of_passes;
  assert(number_of_passes == 1 && "Invalid image");

  png_read_update_info(png_ptr, info_ptr);

  // Error during image read.
  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  auto *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
  for (size_t y = 0; y < height; y++) {
    row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }

  png_read_image(png_ptr, row_pointers);
  png_read_end(png_ptr, info_ptr);

  imageDims[0] = width;
  imageDims[1] = height;
  imageDims[2] = numChannels;
  imageT = static_cast<float *>(
      calloc(1, width * height * numChannels * sizeof(float)));

  float scale = ((range.second - range.first) / 255.0);
  float bias = range.first;

  for (size_t row_n = 0; row_n < height; row_n++) {
    png_byte *row = row_pointers[row_n];
    for (size_t col_n = 0; col_n < width; col_n++) {
      png_byte *ptr =
          &(row[col_n * (hasAlpha ? (numChannels + 1) : numChannels)]);
      if (isGray) {
        imageT[getXYZ(imageDims, row_n, col_n, 0)] =
            float(ptr[0]) * scale + bias;
      } else {
        imageT[getXYZ(imageDims, row_n, col_n, 2)] =
            float(ptr[0]) * scale + bias;
        imageT[getXYZ(imageDims, row_n, col_n, 1)] =
            float(ptr[1]) * scale + bias;
        imageT[getXYZ(imageDims, row_n, col_n, 0)] =
            float(ptr[2]) * scale + bias;
      }
    }
  }

  for (size_t y = 0; y < height; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
  fclose(fp);
  printf("Loaded image: %s\n", filename);

  return false;
}

/// Loads and normalizes all PNGs into a tensor memory block \p resultT in the
/// NCHW 3x224x224 format.
static void loadImagesAndPreprocess(const std::vector<std::string> &filenames,
                                    float *&resultT, size_t *resultDims) {
  assert(filenames.size() > 0 &&
         "There must be at least one filename in filenames");

  // for normalization range setting

  // for resnet18v2.onnx
  // std::pair<float, float> range = std::make_pair(0., 1.0);

  // for mxnet_exported_resnet18.onnx
  std::pair<float, float> range = std::make_pair(0., 255.0);

  unsigned numImages = filenames.size();
  // N x C x H x W
  resultDims[0] = numImages;
  resultDims[1] = 3;
  resultDims[2] = DEFAULT_HEIGHT;
  resultDims[3] = DEFAULT_WIDTH;
  size_t resultSizeInBytes =
      numImages * 3 * DEFAULT_HEIGHT * DEFAULT_WIDTH * sizeof(float);
  resultT = static_cast<float *>(malloc(resultSizeInBytes));
  // We iterate over all the png files, reading them all into our result tensor
  // for processing
  for (unsigned n = 0; n < numImages; n++) {
    float *imageT{nullptr};
    size_t dims[3];
    bool loadSuccess = !readPngImage(filenames[n].c_str(), range, imageT, dims);
    assert(loadSuccess && "Error reading input image.");
    (void)loadSuccess;

    assert((dims[0] == DEFAULT_HEIGHT && dims[1] == DEFAULT_WIDTH) &&
           "All images must have the same Height and Width");

    // Convert to BGR, as this is what NN is expecting.
    for (unsigned z = 0; z < 3; z++) {
      for (unsigned y = 0; y < dims[1]; y++) {
        for (unsigned x = 0; x < dims[0]; x++) {
          resultT[getXYZW(resultDims, n, 2 - z, x, y)] =
              imageT[getXYZ(dims, x, y, z)];
        }
      }
    }
  }
  /// printf("Loaded images size in bytes is: %lu\n", resultSizeInBytes);
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
static uint8_t *initConstantWeights(const char *weightsFileName,
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
  uint8_t *baseConstantWeightVarsAddr =
      static_cast<uint8_t *>(alignedAlloc(config, fileSize));
  //  printf("Allocated weights of size: %lu\n", fileSize);
  //  printf("Expected weights of size: %" PRIu64 "\n",
  //         config.constantWeightVarsMemSize);
  assert(fileSize == config.constantWeightVarsMemSize &&
         "Wrong weights file size");
  int result = fread(baseConstantWeightVarsAddr, fileSize, 1, weightsFile);
  if (result != 1) {
    perror("Could not read the weights file");
  } else {
    // printf("Loaded weights of size: %lu from the file %s\n", fileSize,
    // weightsFileName);
  }
  fclose(weightsFile);
  return baseConstantWeightVarsAddr;
}

/// The assumed layout of the area for mutable WeightVars is:
/// data | gpu_0/data | results
static uint8_t *allocateMutableWeightVars(const BundleConfig &config) {
  auto *weights = static_cast<uint8_t *>(
      alignedAlloc(config, config.mutableWeightVarsMemSize));
  //  printf("Allocated mutable weight variables of size: %" PRIu64 "\n",
  //         config.mutableWeightVarsMemSize);
  return weights;
}

#include <math.h>
void softmax(float *in, float *result) {
  float max = -INFINITY;
  //    for(size_t i = 0; i < N; i++) {
  //      max[i] = -INFINITY;
  //    }

  //    for (size_t i = 0; i < N; i++) {
  for (size_t j = 0; j < 1000; j++) {
    if (in[j] > max) {
      max = in[j];
    }
  }
  //  }

  float sum = 0.0;
  // memset(sum, 0x00, sizeof(float));
  // for (size_t i = 0; i < N; i++) {
  for (size_t j = 0; j < 1000; j++) {
    sum += expf(in[j] - max);
  }
  //}

  // for (size_t i = 0; i < N; i++) {
  for (size_t j = 0; j < 1000; j++) {
    result[j] = expf(in[j] - max) / sum;
  }
  //}
}

/// Dump the result of the inference by looking at the results vector and
/// finding the index of the max element.
static int dumpInferenceResults(const BundleConfig &config,
                                uint8_t *mutableWeightVars, char *resultName) {
  const SymbolTableEntry &outputWeights =
      getMutableWeightVar(config, resultName);
  int maxIdx = 0;
  float maxValue = 0;
  float *results = (float *)(mutableWeightVars + outputWeights.offset);
  int size = (int)outputWeights.size;
  float softmax_result[1000];
  softmax(results, &softmax_result[0]);
  for (int i = 0; i < size; ++i) {
    if (softmax_result[i] > maxValue) {
      maxValue = softmax_result[i];
      maxIdx = i;
    }
  }
  printf("Result: %u\n", maxIdx);
  printf("Confidence: %f\n", maxValue);

  return maxIdx;
}

static uint8_t *copyMutableWeightVarsWithAlloc(const BundleConfig &config,
                                               const char *name,
                                               float *inputT) {

  uint8_t *mutableWeightVarsAddr = allocateMutableWeightVars(config);

  const SymbolTableEntry &inputDataVar = getMutableWeightVar(config, name);
  //  printf("[copyMutableWeightVars] Copying data into mutable weight vars: %lu
  //  bytes\n",
  //         inputDataVar.size*4);
  //  printf( "inputDataVar.offset = %d\n", inputDataVar.offset);
  memcpy(mutableWeightVarsAddr + inputDataVar.offset, inputT,
         inputDataVar.size * 4);

  return mutableWeightVarsAddr;
}

static uint8_t *
copyMutableWeightVarsWithoutAlloc(const BundleConfig &config,
                                  uint8_t *mutableWeightVarsAddr,
                                  const char *name, float *inputT) {

  //  uint8_t *mutableWeightVarsAddr = allocateMutableWeightVars(config);

  const SymbolTableEntry &inputDataVar = getMutableWeightVar(config, name);
  //  printf("[copyMutableWeightVars] Copying data into mutable weight vars: %lu
  //  bytes\n",
  //         inputDataVar.size*4);
  //  printf( "inputDataVar.offset = %d\n", inputDataVar.offset);
  memcpy(mutableWeightVarsAddr + inputDataVar.offset, inputT,
         inputDataVar.size * 4);

  return mutableWeightVarsAddr;
}

void transpose(float *&in, float *&out) {

  float(*inPtr)[1][3][DEFAULT_HEIGHT][DEFAULT_WIDTH];
  float(*outPtr)[1][DEFAULT_HEIGHT][DEFAULT_WIDTH][3];

  inPtr = (float(*)[1][3][DEFAULT_HEIGHT][DEFAULT_WIDTH])in;
  outPtr = (float(*)[1][DEFAULT_HEIGHT][DEFAULT_WIDTH][3])out;

  for (int j = 0; j < 3; j++) {
    for (int k = 0; k < DEFAULT_HEIGHT; k++) {
      for (int l = 0; l < DEFAULT_WIDTH; l++) {
        (*outPtr)[0][k][l][j] = (*inPtr)[0][j][k][l];
      }
    }
  }
}

#define NHWC 1

/// The assumed layout of the area for mutable WeightVars is:
/// data | gpu_0/data | results
static uint8_t *initMutableWeightVars(const BundleConfig &config,
                                      char *varName) {
  uint8_t *mutableWeightVarsAddr = allocateMutableWeightVars(config);
  size_t inputDims[4];
  float *inputT{nullptr};
  loadImagesAndPreprocess(inputImageFilenames, inputT, inputDims);
  // Copy image data into the gpu_0/data input variable in the
  // mutableWeightVars area.
  size_t imageDataSizeInBytes =
      inputDims[0] * inputDims[1] * inputDims[2] * inputDims[3] * sizeof(float);
  //  printf("Copying image data into mutable weight vars: %lu bytes\n",
  //         imageDataSizeInBytes);
  const SymbolTableEntry &inputGPUDataVar =
      getMutableWeightVar(config, varName);

  if (NHWC == 1) {
    float *tmp =
        (float *)malloc(sizeof(float[1][DEFAULT_HEIGHT][DEFAULT_WIDTH][3]));
    memset(tmp, 0, imageDataSizeInBytes);
    transpose(inputT, tmp);
    memcpy(inputT, tmp, imageDataSizeInBytes);
    free(tmp);
  }

  memcpy(mutableWeightVarsAddr + inputGPUDataVar.offset, inputT,
         imageDataSizeInBytes);
  free(inputT);
  return mutableWeightVarsAddr;
}

static uint8_t *initActivations(const BundleConfig &config) {
  return static_cast<uint8_t *>(
      alignedAlloc(config, config.activationsMemSize));
}

static float *getInferenceResultVar(const BundleConfig &config,
                                    uint8_t *mutableWeightVars,
                                    const char *name) {
  const SymbolTableEntry &outputWeights = getMutableWeightVar(config, name);
  float *results = (float *)(mutableWeightVars + outputWeights.offset);
  return results;
}

#include <sys/timeb.h>
int main(int argc, char **argv) {
  parseCommandLineOptions(argc, argv);
#ifdef VTAMAIN
  initVTARuntime();
#endif

  //======== multiple partitions =======//
  uint8_t *constantWeightVarsAddr0 =
      initConstantWeights("p0.weights.bin", p0_config);
  uint8_t *mutableWeightVarsAddr0 = initMutableWeightVars(p0_config, "data");
  uint8_t *activationsAddr0 = initActivations(p0_config);

  timeb t_start;
  ftime(&t_start);

  // Perform the computation.
  //  int errCode0 =
  p0(constantWeightVarsAddr0, mutableWeightVarsAddr0, activationsAddr0);
  //  if (errCode0 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode0);
  //  }

  float *resultVar0 = getInferenceResultVar(p0_config, mutableWeightVarsAddr0,
                                            "resnetv10_pool0_fwd__1");

  uint8_t *constantWeightVarsAddr1 =
      initConstantWeights("p1.weights.bin", p1_config);
  uint8_t *mutableWeightVarsAddr1 = copyMutableWeightVarsWithAlloc(
      p1_config, "resnetv10_pool0_fwd__1", resultVar0);
  uint8_t *activationsAddr1 = initActivations(p1_config);

  // Perform the computation.
  //  int errCode1 =
  p1(constantWeightVarsAddr1, mutableWeightVarsAddr1, activationsAddr1);
  //  if (errCode1 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode1);
  //  }

  float *resultVar1 = getInferenceResultVar(p1_config, mutableWeightVarsAddr1,
                                            "resnetv10_stage1_conv0_fwd__2");
  //  uint8_t *constantWeightVarsAddr2 =
  //      initConstantWeights("p2.weights.bin", p2_config);
  uint8_t *mutableWeightVarsAddr2 = copyMutableWeightVarsWithAlloc(
      p2_config, "resnetv10_stage1_conv0_fwd__2", resultVar1);
  uint8_t *activationsAddr2 = initActivations(p2_config);

  // Perform the computation.
  //  int errCode2 =
  p2(nullptr, mutableWeightVarsAddr2, activationsAddr2);
  //  if (errCode2 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode2);
  //  }

  float *resultVar2 = getInferenceResultVar(p2_config, mutableWeightVarsAddr2,
                                            "resnetv10_stage1_relu0_fwd__1");
  uint8_t *constantWeightVarsAddr3 =
      initConstantWeights("p3.weights.bin", p3_config);
  uint8_t *mutableWeightVarsAddr3 = copyMutableWeightVarsWithAlloc(
      p3_config, "resnetv10_stage1_relu0_fwd__1", resultVar2);
  uint8_t *activationsAddr3 = initActivations(p3_config);

  // Perform the computation.
  //  int errCode3 =
  p3(constantWeightVarsAddr3, mutableWeightVarsAddr3, activationsAddr3);
  //  if (errCode3 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode3);
  //  }

  float *resultVar3 = getInferenceResultVar(p3_config, mutableWeightVarsAddr3,
                                            "resnetv10_stage1_conv1_fwd__2");
  //  uint8_t *constantWeightVarsAddr4 =
  //      initConstantWeights("p4.weights.bin", p4_config);
  uint8_t *mutableWeightVarsAddr4 = copyMutableWeightVarsWithAlloc(
      p4_config, "resnetv10_stage1_conv1_fwd__2", resultVar3);
  copyMutableWeightVarsWithoutAlloc(p4_config, mutableWeightVarsAddr4,
                                    "resnetv10_pool0_fwd__1", resultVar0);

  uint8_t *activationsAddr4 = initActivations(p4_config);

  // Perform the computation.
  //  int errCode4 =
  p4(nullptr, mutableWeightVarsAddr4, activationsAddr4);
  //  if (errCode4 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode4);
  //  }

  float *resultVar4 = getInferenceResultVar(p4_config, mutableWeightVarsAddr4,
                                            "resnetv10_stage1_activation0__1");
  uint8_t *constantWeightVarsAddr5 =
      initConstantWeights("p5.weights.bin", p5_config);
  uint8_t *mutableWeightVarsAddr5 = copyMutableWeightVarsWithAlloc(
      p5_config, "resnetv10_stage1_activation0__1", resultVar4);
  uint8_t *activationsAddr5 = initActivations(p5_config);

  // Perform the computation.
  //  int errCode5 =
  p5(constantWeightVarsAddr5, mutableWeightVarsAddr5, activationsAddr5);
  //  if (errCode5 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode5);
  //  }

  float *resultVar5 = getInferenceResultVar(p5_config, mutableWeightVarsAddr5,
                                            "resnetv10_stage1_conv2_fwd__2");
  // uint8_t *constantWeightVarsAddr6 =
  //    initConstantWeights("p6.weights.bin", p6_config);
  uint8_t *mutableWeightVarsAddr6 = copyMutableWeightVarsWithAlloc(
      p6_config, "resnetv10_stage1_conv2_fwd__2", resultVar5);
  uint8_t *activationsAddr6 = initActivations(p6_config);

  // Perform the computation.
  //  int errCode6 =
  p6(nullptr, mutableWeightVarsAddr6, activationsAddr6);
  //  if (errCode6 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode6);
  //  }

  float *resultVar6 = getInferenceResultVar(p6_config, mutableWeightVarsAddr6,
                                            "resnetv10_stage1_relu1_fwd__1");
  uint8_t *constantWeightVarsAddr7 =
      initConstantWeights("p7.weights.bin", p7_config);
  uint8_t *mutableWeightVarsAddr7 = copyMutableWeightVarsWithAlloc(
      p7_config, "resnetv10_stage1_relu1_fwd__1", resultVar6);
  uint8_t *activationsAddr7 = initActivations(p7_config);

  // Perform the computation.
  //  int errCode7 =
  p7(constantWeightVarsAddr7, mutableWeightVarsAddr7, activationsAddr7);
  //  if (errCode7 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode7);
  //  }

  float *resultVar7 = getInferenceResultVar(p7_config, mutableWeightVarsAddr7,
                                            "resnetv10_stage1_conv3_fwd__2");
  // uint8_t *constantWeightVarsAddr8 =
  //     initConstantWeights("p8.weights.bin", p8_config);
  uint8_t *mutableWeightVarsAddr8 = copyMutableWeightVarsWithAlloc(
      p8_config, "resnetv10_stage1_conv3_fwd__2", resultVar7);
  copyMutableWeightVarsWithoutAlloc(p8_config, mutableWeightVarsAddr8,
                                    "resnetv10_stage1_activation0__1",
                                    resultVar4);
  uint8_t *activationsAddr8 = initActivations(p8_config);

  // Perform the computation.
  //  int errCode8 =
  p8(nullptr, mutableWeightVarsAddr8, activationsAddr8);
  //  if (errCode8 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode8);
  //  }

  float *resultVar8 = getInferenceResultVar(p8_config, mutableWeightVarsAddr8,
                                            "resnetv10_stage1_activation1__1");
  uint8_t *constantWeightVarsAddr9 =
      initConstantWeights("p9.weights.bin", p9_config);
  uint8_t *mutableWeightVarsAddr9 = copyMutableWeightVarsWithAlloc(
      p9_config, "resnetv10_stage1_activation1__1", resultVar8);
  uint8_t *activationsAddr9 = initActivations(p9_config);

  // Perform the computation.
  //  int errCode9 =
  p9(constantWeightVarsAddr9, mutableWeightVarsAddr9, activationsAddr9);
  //  if (errCode9 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode9);
  //  }

  float *resultVar9 = getInferenceResultVar(p9_config, mutableWeightVarsAddr9,
                                            "resnetv10_stage2_conv0_fwd__2");
  //  uint8_t *constantWeightVarsAddr10 =
  //      initConstantWeights("p10.weights.bin", p10_config);
  uint8_t *mutableWeightVarsAddr10 = copyMutableWeightVarsWithAlloc(
      p10_config, "resnetv10_stage2_conv0_fwd__2", resultVar9);
  uint8_t *activationsAddr10 = initActivations(p10_config);

  // Perform the computation.
  //  int errCode10 =
  p10(nullptr, mutableWeightVarsAddr10, activationsAddr10);
  //  if (errCode10 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode10);
  //  }

  uint8_t *constantWeightVarsAddr11 =
      initConstantWeights("p11.weights.bin", p11_config);
  uint8_t *mutableWeightVarsAddr11 = copyMutableWeightVarsWithAlloc(
      p11_config, "resnetv10_stage1_activation1__1", resultVar8);
  uint8_t *activationsAddr11 = initActivations(p11_config);

  // Perform the computation.
  //  int errCode11 =
  p11(constantWeightVarsAddr11, mutableWeightVarsAddr11, activationsAddr11);
  //  if (errCode11 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode11);
  //  }

  float *resultVar10 = getInferenceResultVar(
      p10_config, mutableWeightVarsAddr10, "resnetv10_stage2_relu0_fwd__1");
  uint8_t *constantWeightVarsAddr12 =
      initConstantWeights("p12.weights.bin", p12_config);
  uint8_t *mutableWeightVarsAddr12 = copyMutableWeightVarsWithAlloc(
      p12_config, "resnetv10_stage2_relu0_fwd__1", resultVar10);
  uint8_t *activationsAddr12 = initActivations(p12_config);

  // Perform the computation.
  //  int errCode12 =
  p12(constantWeightVarsAddr12, mutableWeightVarsAddr12, activationsAddr12);
  //  if (errCode12 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode12);
  //  }

  float *resultVar11 = getInferenceResultVar(
      p11_config, mutableWeightVarsAddr11, "resnetv10_stage2_conv2_fwd__2");
  float *resultVar12 = getInferenceResultVar(
      p12_config, mutableWeightVarsAddr12, "resnetv10_stage2_conv1_fwd__2");
  // uint8_t *constantWeightVarsAddr13 =
  //     initConstantWeights("p13.weights.bin", p13_config);
  uint8_t *mutableWeightVarsAddr13 = copyMutableWeightVarsWithAlloc(
      p13_config, "resnetv10_stage2_conv2_fwd__2", resultVar11);
  copyMutableWeightVarsWithoutAlloc(p13_config, mutableWeightVarsAddr13,
                                    "resnetv10_stage2_conv1_fwd__2",
                                    resultVar12);
  uint8_t *activationsAddr13 = initActivations(p13_config);

  // Perform the computation.
  //  int errCode13 =
  p13(nullptr, mutableWeightVarsAddr13, activationsAddr13);
  //  if (errCode13 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode13);
  //  }

  float *resultVar13 = getInferenceResultVar(
      p13_config, mutableWeightVarsAddr13, "resnetv10_stage2_activation0__1");
  uint8_t *constantWeightVarsAddr14 =
      initConstantWeights("p14.weights.bin", p14_config);
  uint8_t *mutableWeightVarsAddr14 = copyMutableWeightVarsWithAlloc(
      p14_config, "resnetv10_stage2_activation0__1", resultVar13);
  uint8_t *activationsAddr14 = initActivations(p14_config);

  // Perform the computation.
  //  int errCode14 =
  p14(constantWeightVarsAddr14, mutableWeightVarsAddr14, activationsAddr14);
  //  if (errCode14 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode14);
  //  }

  float *resultVar14 = getInferenceResultVar(
      p14_config, mutableWeightVarsAddr14, "resnetv10_stage2_conv3_fwd__2");
  // uint8_t *constantWeightVarsAddr16 =
  //    initConstantWeights("p16.weights.bin", p16_config);
  uint8_t *mutableWeightVarsAddr15 = copyMutableWeightVarsWithAlloc(
      p15_config, "resnetv10_stage2_conv3_fwd__2", resultVar14);
  uint8_t *activationsAddr15 = initActivations(p15_config);

  // Perform the computation.
  //  int errCode15 =
  p15(nullptr, mutableWeightVarsAddr15, activationsAddr15);
  //  if (errCode15 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode15);
  //  }

  float *resultVar15 = getInferenceResultVar(
      p16_config, mutableWeightVarsAddr15, "resnetv10_stage2_relu1_fwd__1");
  uint8_t *constantWeightVarsAddr16 =
      initConstantWeights("p16.weights.bin", p16_config);
  uint8_t *mutableWeightVarsAddr16 = copyMutableWeightVarsWithAlloc(
      p16_config, "resnetv10_stage2_relu1_fwd__1", resultVar15);
  uint8_t *activationsAddr16 = initActivations(p16_config);

  // Perform the computation.
  //  int errCode16 =
  p16(constantWeightVarsAddr16, mutableWeightVarsAddr16, activationsAddr16);
  //  if (errCode16 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode16);
  //  }

  float *resultVar16 = getInferenceResultVar(
      p16_config, mutableWeightVarsAddr16, "resnetv10_stage2_conv4_fwd__2");
  //  uint8_t *constantWeightVarsAddr18 =
  //      initConstantWeights("p18.weights.bin", p18_config);
  uint8_t *mutableWeightVarsAddr17 = copyMutableWeightVarsWithAlloc(
      p17_config, "resnetv10_stage2_conv4_fwd__2", resultVar16);
  copyMutableWeightVarsWithoutAlloc(p17_config, mutableWeightVarsAddr17,
                                    "resnetv10_stage2_activation0__1",
                                    resultVar13);
  uint8_t *activationsAddr17 = initActivations(p17_config);

  // Perform the computation.
  //  int errCode17 =
  p17(nullptr, mutableWeightVarsAddr17, activationsAddr17);
  //  if (errCode17 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode17);
  //  }

  float *resultVar17 = getInferenceResultVar(
      p18_config, mutableWeightVarsAddr17, "resnetv10_stage2_activation1__1");
  uint8_t *constantWeightVarsAddr18 =
      initConstantWeights("p18.weights.bin", p18_config);
  uint8_t *mutableWeightVarsAddr18 = copyMutableWeightVarsWithAlloc(
      p18_config, "resnetv10_stage2_activation1__1", resultVar17);
  uint8_t *activationsAddr18 = initActivations(p18_config);

  // Perform the computation.
  //  int errCode18 =
  p18(constantWeightVarsAddr18, mutableWeightVarsAddr18, activationsAddr18);
  //  if (errCode18 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode18);
  //  }

  float *resultVar18 = getInferenceResultVar(
      p18_config, mutableWeightVarsAddr18, "resnetv10_stage3_conv0_fwd__2");
  //   uint8_t *constantWeightVarsAddr19 =
  //       initConstantWeights("p19.weights.bin", p19_config);
  uint8_t *mutableWeightVarsAddr19 = copyMutableWeightVarsWithAlloc(
      p19_config, "resnetv10_stage3_conv0_fwd__2", resultVar18);
  uint8_t *activationsAddr19 = initActivations(p19_config);

  // Perform the computation.
  //  int errCode19 =
  p19(nullptr, mutableWeightVarsAddr19, activationsAddr19);
  //  if (errCode19 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode19);
  //  }

  // float* resultVar19 = getInferenceResultVar(p19_config,
  // mutableWeightVarsAddr19, "tmp__19");
  uint8_t *constantWeightVarsAddr20 =
      initConstantWeights("p20.weights.bin", p20_config);
  uint8_t *mutableWeightVarsAddr20 = copyMutableWeightVarsWithAlloc(
      p20_config, "resnetv10_stage2_activation1__1", resultVar17);
  uint8_t *activationsAddr20 = initActivations(p20_config);

  // Perform the computation.
  //  int errCode20 =
  p20(constantWeightVarsAddr20, mutableWeightVarsAddr20, activationsAddr20);
  //  if (errCode20 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode20);
  //  }

  float *resultVar19 = getInferenceResultVar(
      p19_config, mutableWeightVarsAddr19, "resnetv10_stage3_relu0_fwd__1");
  uint8_t *constantWeightVarsAddr21 =
      initConstantWeights("p21.weights.bin", p21_config);
  uint8_t *mutableWeightVarsAddr21 = copyMutableWeightVarsWithAlloc(
      p21_config, "resnetv10_stage3_relu0_fwd__1", resultVar19);
  uint8_t *activationsAddr21 = initActivations(p21_config);

  // Perform the computation.
  //  int errCode21 =
  p21(constantWeightVarsAddr21, mutableWeightVarsAddr21, activationsAddr21);
  //  if (errCode21 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode21);
  //  }

  float *resultVar21 = getInferenceResultVar(
      p21_config, mutableWeightVarsAddr21, "resnetv10_stage3_conv1_fwd__2");
  float *resultVar20 = getInferenceResultVar(
      p20_config, mutableWeightVarsAddr20, "resnetv10_stage3_conv2_fwd__2");
  //  uint8_t *constantWeightVarsAddr22 =
  //      initConstantWeights("p22.weights.bin", p22_config);
  uint8_t *mutableWeightVarsAddr22 = copyMutableWeightVarsWithAlloc(
      p22_config, "resnetv10_stage3_conv1_fwd__2", resultVar21);
  copyMutableWeightVarsWithoutAlloc(p22_config, mutableWeightVarsAddr22,
                                    "resnetv10_stage3_conv2_fwd__2",
                                    resultVar20);
  uint8_t *activationsAddr22 = initActivations(p22_config);

  // Perform the computation.
  //  int errCode22 =
  p22(nullptr, mutableWeightVarsAddr22, activationsAddr22);
  //  if (errCode22 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode22);
  //  }

  float *resultVar22 = getInferenceResultVar(
      p22_config, mutableWeightVarsAddr22, "resnetv10_stage3_activation0__1");
  uint8_t *constantWeightVarsAddr23 =
      initConstantWeights("p23.weights.bin", p23_config);
  uint8_t *mutableWeightVarsAddr23 = copyMutableWeightVarsWithAlloc(
      p23_config, "resnetv10_stage3_activation0__1", resultVar22);
  uint8_t *activationsAddr23 = initActivations(p23_config);

  // Perform the computation.
  //  int errCode23 =
  p23(constantWeightVarsAddr23, mutableWeightVarsAddr23, activationsAddr23);
  //  if (errCode23 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode23);
  //  }

  float *resultVar23 = getInferenceResultVar(
      p23_config, mutableWeightVarsAddr23, "resnetv10_stage3_conv3_fwd__2");
  //   uint8_t *constantWeightVarsAddr24 =
  //       initConstantWeights("p24.weights.bin", p24_config);
  uint8_t *mutableWeightVarsAddr24 = copyMutableWeightVarsWithAlloc(
      p24_config, "resnetv10_stage3_conv3_fwd__2", resultVar23);
  uint8_t *activationsAddr24 = initActivations(p24_config);

  // Perform the computation.
  //  int errCode24 =
  p24(nullptr, mutableWeightVarsAddr24, activationsAddr24);
  //  if (errCode24 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode24);
  //  }

  float *resultVar24 = getInferenceResultVar(
      p24_config, mutableWeightVarsAddr24, "resnetv10_stage3_relu1_fwd__1");
  uint8_t *constantWeightVarsAddr25 =
      initConstantWeights("p25.weights.bin", p25_config);
  uint8_t *mutableWeightVarsAddr25 = copyMutableWeightVarsWithAlloc(
      p25_config, "resnetv10_stage3_relu1_fwd__1", resultVar24);
  uint8_t *activationsAddr25 = initActivations(p25_config);

  // Perform the computation.
  //  int errCode25 =
  p25(constantWeightVarsAddr25, mutableWeightVarsAddr25, activationsAddr25);
  //  if (errCode25 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode25);
  //  }

  float *resultVar25 = getInferenceResultVar(
      p25_config, mutableWeightVarsAddr25, "resnetv10_stage3_conv4_fwd__2");
  //  uint8_t *constantWeightVarsAddr26 =
  //      initConstantWeights("p26.weights.bin", p26_config);
  uint8_t *mutableWeightVarsAddr26 = copyMutableWeightVarsWithAlloc(
      p26_config, "resnetv10_stage3_conv4_fwd__2", resultVar25);
  copyMutableWeightVarsWithoutAlloc(p26_config, mutableWeightVarsAddr26,
                                    "resnetv10_stage3_activation0__1",
                                    resultVar22);
  uint8_t *activationsAddr26 = initActivations(p26_config);

  // Perform the computation.
  //  int errCode26 =
  p26(nullptr, mutableWeightVarsAddr26, activationsAddr26);
  //  if (errCode26 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode26);
  //  }

  float *resultVar26 = getInferenceResultVar(
      p26_config, mutableWeightVarsAddr26, "resnetv10_stage3_activation1__1");
  uint8_t *constantWeightVarsAddr27 =
      initConstantWeights("p27.weights.bin", p27_config);
  uint8_t *mutableWeightVarsAddr27 = copyMutableWeightVarsWithAlloc(
      p27_config, "resnetv10_stage3_activation1__1", resultVar26);
  uint8_t *activationsAddr27 = initActivations(p27_config);

  // Perform the computation.
  //  int errCode27 =
  p27(constantWeightVarsAddr27, mutableWeightVarsAddr27, activationsAddr27);
  //  if (errCode27 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode27);
  //  }

  float *resultVar27 = getInferenceResultVar(
      p27_config, mutableWeightVarsAddr27, "resnetv10_stage4_conv0_fwd__2");
  //   uint8_t *constantWeightVarsAddr28 =//
  //       initConstantWeights("p28.weights.bin", p28_config);
  uint8_t *mutableWeightVarsAddr28 = copyMutableWeightVarsWithAlloc(
      p28_config, "resnetv10_stage4_conv0_fwd__2", resultVar27);
  uint8_t *activationsAddr28 = initActivations(p28_config);

  // Perform the computation.
  //  int errCode28 =
  p28(nullptr, mutableWeightVarsAddr28, activationsAddr28);
  //  if (errCode28 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode28);
  //  }

  float *resultVar28 = getInferenceResultVar(
      p28_config, mutableWeightVarsAddr28, "resnetv10_stage4_relu0_fwd__1");
  uint8_t *constantWeightVarsAddr29 =
      initConstantWeights("p29.weights.bin", p29_config);
  uint8_t *mutableWeightVarsAddr29 = copyMutableWeightVarsWithAlloc(
      p29_config, "resnetv10_stage4_relu0_fwd__1", resultVar28);
  copyMutableWeightVarsWithoutAlloc(p29_config, mutableWeightVarsAddr29,
                                    "resnetv10_stage3_activation1__1",
                                    resultVar26);
  uint8_t *activationsAddr29 = initActivations(p29_config);

  // Perform the computation.
  //  int errCode29 =
  p29(constantWeightVarsAddr29, mutableWeightVarsAddr29, activationsAddr29);
  //  if (errCode29 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode29);
  //  }

  float *resultVar29_1 = getInferenceResultVar(
      p29_config, mutableWeightVarsAddr29, "resnetv10_stage4_conv2_fwd__2");
  float *resultVar29_2 = getInferenceResultVar(
      p29_config, mutableWeightVarsAddr29, "resnetv10_stage4_conv1_fwd__2");
  //  uint8_t *constantWeightVarsAddr30 =
  //      initConstantWeights("p30.weights.bin", p30_config);
  uint8_t *mutableWeightVarsAddr30 = copyMutableWeightVarsWithAlloc(
      p30_config, "resnetv10_stage4_conv2_fwd__2", resultVar29_1);
  copyMutableWeightVarsWithoutAlloc(p30_config, mutableWeightVarsAddr30,
                                    "resnetv10_stage4_conv1_fwd__2",
                                    resultVar29_2);
  uint8_t *activationsAddr30 = initActivations(p30_config);

  // Perform the computation.
  //  int errCode30 =
  p30(nullptr, mutableWeightVarsAddr30, activationsAddr30);
  //  if (errCode30 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode30);
  //  }

  float *resultVar30 = getInferenceResultVar(
      p30_config, mutableWeightVarsAddr30, "resnetv10_stage4_activation0__1");
  uint8_t *constantWeightVarsAddr31 =
      initConstantWeights("p31.weights.bin", p31_config);
  uint8_t *mutableWeightVarsAddr31 = copyMutableWeightVarsWithAlloc(
      p31_config, "resnetv10_stage4_activation0__1", resultVar30);
  uint8_t *activationsAddr31 = initActivations(p31_config);

  // Perform the computation.
  //  int errCode31 =
  p31(constantWeightVarsAddr31, mutableWeightVarsAddr31, activationsAddr31);
  //  if (errCode31 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode31);
  //  }

  float *resultVar31 = getInferenceResultVar(
      p31_config, mutableWeightVarsAddr31, "resnetv10_stage4_conv3_fwd__2");
  //  uint8_t *constantWeightVarsAddr32 =
  //      initConstantWeights("p32.weights.bin", p32_config);
  uint8_t *mutableWeightVarsAddr32 = copyMutableWeightVarsWithAlloc(
      p32_config, "resnetv10_stage4_conv3_fwd__2", resultVar31);
  uint8_t *activationsAddr32 = initActivations(p32_config);

  // Perform the computation.
  //  int errCode32 =
  p32(nullptr, mutableWeightVarsAddr32, activationsAddr32);
  //  if (errCode32 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode32);
  //  }

  float *resultVar32 = getInferenceResultVar(
      p32_config, mutableWeightVarsAddr32, "resnetv10_stage4_relu1_fwd__1");
  uint8_t *constantWeightVarsAddr33 =
      initConstantWeights("p33.weights.bin", p33_config);
  uint8_t *mutableWeightVarsAddr33 = copyMutableWeightVarsWithAlloc(
      p33_config, "resnetv10_stage4_relu1_fwd__1", resultVar32);
  uint8_t *activationsAddr33 = initActivations(p33_config);

  // Perform the computation.
  //  int errCode33 =
  p33(constantWeightVarsAddr33, mutableWeightVarsAddr33, activationsAddr33);
  //  if (errCode33 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode33);
  //  }

  float *resultVar33 = getInferenceResultVar(
      p33_config, mutableWeightVarsAddr33, "resnetv10_stage4_conv4_fwd__2");
  uint8_t *constantWeightVarsAddr34 =
      initConstantWeights("p34.weights.bin", p34_config);
  uint8_t *mutableWeightVarsAddr34 = copyMutableWeightVarsWithAlloc(
      p34_config, "resnetv10_stage4_conv4_fwd__2", resultVar33);
  copyMutableWeightVarsWithoutAlloc(p34_config, mutableWeightVarsAddr34,
                                    "resnetv10_stage4_activation0__1",
                                    resultVar30);
  uint8_t *activationsAddr34 = initActivations(p34_config);

  // Perform the computation.
  //  int errCode34 =
  p34(constantWeightVarsAddr34, mutableWeightVarsAddr34, activationsAddr34);
  //  if (errCode34 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode34);
  //  }

  timeb t_now;
  ftime(&t_now);
  float inftime = (float)(t_now.time - t_start.time) * 1000.0f +
                  (float)(t_now.millitm - t_start.millitm);

  printf("\n====> [Inference time] %.4lf ms.\n", inftime);

  printf("\n======= Result ========\n");
  int maxIdx = dumpInferenceResults(p34_config, mutableWeightVarsAddr34,
                                    "resnetv10_dense0_fwd");
  //
  //  float* resultVar34 = getInferenceResultVar(p34_config,
  //  mutableWeightVarsAddr34, "resnetv10_dense0_fwd");
  // // uint8_t *constantWeightVarsAddr35 =
  // //     initConstantWeights("p35.weights.bin", p35_config);
  //  uint8_t *mutableWeightVarsAddr35 =
  //  copyMutableWeightVarsWithAlloc(p35_config, "resnetv10_dense0_fwd",
  //  resultVar34); uint8_t *activationsAddr35 = initActivations(p35_config);
  //
  //  // Perform the computation.
  //  int errCode35 =
  //      p35(nullptr, mutableWeightVarsAddr35, activationsAddr35);
  //  if (errCode35 != GLOW_SUCCESS) {
  //    printf("Error running bundle: error code %d\n", errCode35);
  //  }
  //
  //  printf("====== output of the final node of the neural net ========\n");
  //  dumpInferenceResults(p35_config, mutableWeightVarsAddr35,
  //  "resnetv10_dense0_fwd__1");
  //

  //   Free all resources.
  free(activationsAddr0);
  free(constantWeightVarsAddr0);
  free(mutableWeightVarsAddr0);

  free(activationsAddr1);
  free(constantWeightVarsAddr1);
  free(mutableWeightVarsAddr1);

  free(activationsAddr2);
  //  free(constantWeightVarsAddr2);
  free(mutableWeightVarsAddr2);

  free(activationsAddr3);
  free(constantWeightVarsAddr3);
  free(mutableWeightVarsAddr3);

  free(activationsAddr4);
  // free(constantWeightVarsAddr4);
  free(mutableWeightVarsAddr4);

  free(activationsAddr5);
  free(constantWeightVarsAddr5);
  free(mutableWeightVarsAddr5);

  free(activationsAddr6);
  // free(constantWeightVarsAddr6);
  free(mutableWeightVarsAddr6);

  free(activationsAddr7);
  free(constantWeightVarsAddr7);
  free(mutableWeightVarsAddr7);

  free(activationsAddr8);
  //  free(constantWeightVarsAddr8);
  free(mutableWeightVarsAddr8);

  free(activationsAddr9);
  free(constantWeightVarsAddr9);
  free(mutableWeightVarsAddr9);

  free(activationsAddr10);
  //  free(constantWeightVarsAddr10);
  free(mutableWeightVarsAddr10);

  free(activationsAddr11);
  free(constantWeightVarsAddr11);
  free(mutableWeightVarsAddr11);

  free(activationsAddr12);
  free(constantWeightVarsAddr12);
  free(mutableWeightVarsAddr12);

  free(activationsAddr13);
  //  free(constantWeightVarsAddr13);
  free(mutableWeightVarsAddr13);

  free(activationsAddr14);
  free(constantWeightVarsAddr14);
  free(mutableWeightVarsAddr14);

  free(activationsAddr15);
  //  free(constantWeightVarsAddr15);
  free(mutableWeightVarsAddr15);

  free(activationsAddr16);
  free(constantWeightVarsAddr16);
  free(mutableWeightVarsAddr16);

  free(activationsAddr17);
  //  free(constantWeightVarsAddr17);
  free(mutableWeightVarsAddr17);

  free(activationsAddr18);
  free(constantWeightVarsAddr18);
  free(mutableWeightVarsAddr18);

  free(activationsAddr19);
  //  free(constantWeightVarsAddr19);
  free(mutableWeightVarsAddr19);

  free(activationsAddr20);
  free(constantWeightVarsAddr20);
  free(mutableWeightVarsAddr20);

  free(activationsAddr21);
  free(constantWeightVarsAddr21);
  free(mutableWeightVarsAddr21);

  free(activationsAddr22);
  //  free(constantWeightVarsAddr22);
  free(mutableWeightVarsAddr22);

  free(activationsAddr23);
  free(constantWeightVarsAddr23);
  free(mutableWeightVarsAddr23);

  free(activationsAddr24);
  //  free(constantWeightVarsAddr24);
  free(mutableWeightVarsAddr24);

  free(activationsAddr25);
  free(constantWeightVarsAddr25);
  free(mutableWeightVarsAddr25);

  free(activationsAddr26);
  //  free(constantWeightVarsAddr26);
  free(mutableWeightVarsAddr26);

  free(activationsAddr27);
  free(constantWeightVarsAddr27);
  free(mutableWeightVarsAddr27);

  free(activationsAddr28);
  //  free(constantWeightVarsAddr28);
  free(mutableWeightVarsAddr28);

  free(activationsAddr29);
  free(constantWeightVarsAddr29);
  free(mutableWeightVarsAddr29);

  free(activationsAddr30);
  //  free(constantWeightVarsAddr30);
  free(mutableWeightVarsAddr30);

  free(activationsAddr31);
  free(constantWeightVarsAddr31);
  free(mutableWeightVarsAddr31);

  free(activationsAddr32);
  //  free(constantWeightVarsAddr32);
  free(mutableWeightVarsAddr32);

  free(activationsAddr33);
  free(constantWeightVarsAddr33);
  free(mutableWeightVarsAddr33);

  free(activationsAddr34);
  free(constantWeightVarsAddr34);
  free(mutableWeightVarsAddr34);

  //  free(activationsAddr35);
  // free(constantWeightVarsAddr35);
  //  free(mutableWeightVarsAddr35);

#ifdef VTAMAIN
  destroyVTARuntime();
#endif

  return !(maxIdx == 281);
}
