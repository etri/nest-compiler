/*****************************************************************************
 *
 * Copyright Next-Generation System Software Research Group, All rights
 *reserved. Future Computing Research Division, Artificial Intelligence Reserch
 *Laboratory Electronics and Telecommunications Research Institute (ETRI)
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

/*
//
// Created by luiin on 20. 10. 20..
//

#ifndef GLOW_MAIN_TEMPLATE_H
#define GLOW_MAIN_TEMPLATE_H

#include <assert.h>
#include <inttypes.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "main_header_test.h"

/// This is an example demonstrating how to use auto-generated bundles and
/// create standalone executables that can perform neural network computations.
/// This example loads and runs the compiled resnet50 network model.
/// This example is using the dynamic bundle API (default).

#define DEFAULT_HEIGHT 224
#define DEFAULT_WIDTH 224

//===----------------------------------------------------------------------===//
//                   Image processing helpers
//===----------------------------------------------------------------------===//
class main_template {
  std::vector<std::string> inputImageFilenames;

  size_t getXYZW(const size_t *dims, size_t x, size_t y, size_t z, size_t w);
  size_t getXYZ(const size_t *dims, size_t x, size_t y, size_t z);
  bool readPngImage(const char *filename, std::pair<float, float> range,
                    float *&imageT, size_t *imageDims);
  static void loadImagesAndPreprocess(const std::vector<std::string> &filenames,
                                      float *&resultT, size_t *resultDims);
  void parseCommandLineOptions(int argc, char **argv);
  const SymbolTableEntry *getWeightVar(const BundleConfig &config,
                                       const char *name);
  const SymbolTableEntry &getMutableWeightVar(const BundleConfig &config,
                                              const char *name);
  void *alignedAlloc(const BundleConfig &config, size_t size);
  static uint8_t *initConstantWeights(const char *weightsFileName,
                                      const BundleConfig &config);
  static uint8_t *allocateMutableWeightVars(const BundleConfig &config);
  static void dumpInferenceResults(const BundleConfig &config,
                                   uint8_t *mutableWeightVars, char*
resultName); static uint8_t *copyMutableWeightVars(const BundleConfig &config,
const char *name, float* inputT); static uint8_t
*copyMutableWeightVarsWithAlloc(const BundleConfig &config, const char *name,
float* inputT); static uint8_t *copyMutableWeightVarsWithoutAlloc(const
BundleConfig &config, uint8_t *mutableWeightVarsAddr, const char *name, float*
inputT); void transpose(float* &in, float* &out);

  #define NHWC 1

  static uint8_t *initMutableWeightVars(const BundleConfig &config, char*
varName); static uint8_t *initActivations(const BundleConfig &config); static
float* getInferenceResultVar(const BundleConfig &config, uint8_t
*mutableWeightVars, const char* name);

};

#endif // GLOW_MAIN_TEMPLATE_H
*/

//
// Created by luiin on 20. 10. 20..
//

#include <assert.h>
#include <inttypes.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include <fstream>

#include "main_header_test.h"

#define DEFAULT_HEIGHT 224
#define DEFAULT_WIDTH 224

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
        //        imageT[getXYZ(imageDims, row_n, col_n, 0)] =
        //            float(ptr[0]) * scale + bias;
        //        imageT[getXYZ(imageDims, row_n, col_n, 1)] =
        //            float(ptr[1]) * scale + bias;
        //        imageT[getXYZ(imageDims, row_n, col_n, 2)] =
        //            float(ptr[2]) * scale + bias;
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

//#define NHWC 0
//#define RANGE 1

#define NHWC 1
#define RANGE 255

/// Loads and normalizes all PNGs into a tensor memory block \p resultT in the
/// NCHW 3x224x224 format.
static void loadImagesAndPreprocess(const std::vector<std::string> &filenames,
                                    float *&resultT, size_t *resultDims) {
  assert(filenames.size() > 0 &&
         "There must be at least one filename in filenames");

  // for normalization range setting

  // for resnet18v2.onnx
  std::pair<float, float> range = std::make_pair(0., 1.0);

  if (RANGE == 255) {
    // for mxnet_exported_resnet18.onnx
    range = std::make_pair(0., 255.0);
  }

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
  printf("Loaded images size in bytes is: %lu\n", resultSizeInBytes);
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
  // printf("%s", weightsFileName);
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
    // perror("Could not read the weights file");
  } else {
    printf("Loaded weights of size: %lu from the file %s\n", fileSize,
           weightsFileName);
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

/// Dump the result of the inference by looking at the results vector and
/// finding the index of the max element.
// static void dumpInferenceResults(const BundleConfig &config,
//                                 uint8_t *mutableWeightVars, char* resultName)
//                                 {
//  const SymbolTableEntry &outputWeights =
//      getMutableWeightVar(config, resultName);
//  int maxIdx = 0;
//  float maxValue = 0;
//  float *results = (float *)(mutableWeightVars + outputWeights.offset);
//  for (int i = 0; i < (int)outputWeights.size; ++i) {
//    if (results[i] > maxValue) {
//      maxValue = results[i];
//      maxIdx = i;
//    }
//  }
//  printf("Result: %u\n", maxIdx);
//  printf("Confidence: %f\n", maxValue);
//}

#include <math.h>
void softmax(float *in, float *result) {
  float max = -INFINITY;
  for (size_t j = 0; j < 1000; j++) {
    if (in[j] > max) {
      max = in[j];
    }
  }

  float sum = 0.0;
  for (size_t j = 0; j < 1000; j++) {
    sum += expf(in[j] - max);
  }

  for (size_t j = 0; j < 1000; j++) {
    result[j] = expf(in[j] - max) / sum;
  }
}

/// Dump the result of the inference by looking at the results vector and
/// finding the index of the max element.
static void dumpInferenceResults(const BundleConfig &config,
                                 uint8_t *mutableWeightVars, char *resultName) {
  const SymbolTableEntry &outputWeights =
      getMutableWeightVar(config, resultName);
  int maxIdx = 0;
  float maxValue = 0;
  float *results = (float *)(mutableWeightVars + outputWeights.offset);
  int size = (int)outputWeights.size;
  float softmax_result[1000];
  softmax(results, softmax_result);
  for (int i = 0; i < size; ++i) {
    if (softmax_result[i] > maxValue) {
      maxValue = softmax_result[i];
      maxIdx = i;
    }
  }
  printf("Result: %u\n", maxIdx);
  printf("Confidence: %f\n", maxValue);
}

static uint8_t *copyMutableWeightVars(const BundleConfig &config,
                                      const char *name, float *inputT) {

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

void createYamlFile();
void setProfilingInfo_delayTime(float time, int i);
float getProfilingInfo_delay_time(int i);
