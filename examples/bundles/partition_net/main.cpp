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
#include <assert.h>
#include <inttypes.h>
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
#include "p2.h"
#include "p3.h"

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
        imageT[getXYZ(imageDims, row_n, col_n, 0)] =
            float(ptr[0]) * scale + bias;
        imageT[getXYZ(imageDims, row_n, col_n, 1)] =
            float(ptr[1]) * scale + bias;
        imageT[getXYZ(imageDims, row_n, col_n, 2)] =
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

  //for normalization range setting

  //for resnet18v2.onnx
  //std::pair<float, float> range = std::make_pair(0., 1.0);

  //for mxnet_exported_resnet18.onnx
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
static void dumpInferenceResults(const BundleConfig &config,
                                 uint8_t *mutableWeightVars, char* resultName) {
  const SymbolTableEntry &outputWeights =
      getMutableWeightVar(config, resultName);
  int maxIdx = 0;
  float maxValue = 0;
  float *results = (float *)(mutableWeightVars + outputWeights.offset);
  for (int i = 0; i < (int)outputWeights.size; ++i) {
    if (results[i] > maxValue) {
      maxValue = results[i];
      maxIdx = i;
    }
  }
  printf("Result: %u\n", maxIdx);
  printf("Confidence: %f\n", maxValue);
}

static uint8_t *copyMutableWeightVarsWithAlloc(const BundleConfig &config, const char *name, float* inputT) {

  uint8_t *mutableWeightVarsAddr = allocateMutableWeightVars(config);

  const SymbolTableEntry &inputDataVar = getMutableWeightVar(config, name);
//  printf("[copyMutableWeightVars] Copying data into mutable weight vars: %lu bytes\n",
//         inputDataVar.size*4);
//  printf( "inputDataVar.offset = %d\n", inputDataVar.offset);
  memcpy(mutableWeightVarsAddr + inputDataVar.offset, inputT,
         inputDataVar.size*4);

  return mutableWeightVarsAddr;
}

static uint8_t *copyMutableWeightVarsWithoutAlloc(const BundleConfig &config, uint8_t *mutableWeightVarsAddr, const char *name, float* inputT) {

//  uint8_t *mutableWeightVarsAddr = allocateMutableWeightVars(config);

  const SymbolTableEntry &inputDataVar = getMutableWeightVar(config, name);
//  printf("[copyMutableWeightVars] Copying data into mutable weight vars: %lu bytes\n",
//         inputDataVar.size*4);
//  printf( "inputDataVar.offset = %d\n", inputDataVar.offset);
  memcpy(mutableWeightVarsAddr + inputDataVar.offset, inputT,
         inputDataVar.size*4);

  return mutableWeightVarsAddr;
}

void transpose(float* &in, float* &out) {

  float (*inPtr)[1][3][DEFAULT_HEIGHT][DEFAULT_WIDTH];
  float (*outPtr)[1][DEFAULT_HEIGHT][DEFAULT_WIDTH][3];

  inPtr = (float (*)[1][3][DEFAULT_HEIGHT][DEFAULT_WIDTH])in;
  outPtr = (float (*)[1][DEFAULT_HEIGHT][DEFAULT_WIDTH][3])out;


  for (int j=0; j<3; j++){
    for (int k=0; k<DEFAULT_HEIGHT; k++){
      for (int l=0; l<DEFAULT_WIDTH; l++){
        (*outPtr)[0][k][l][j] = (*inPtr)[0][j][k][l];
      }
    }
  }

}

//static uint8_t *copyMutableWeightVars(const BundleConfig &config, const char *name, float* inputT) {
//
//  uint8_t *mutableWeightVarsAddr = allocateMutableWeightVars(config);
//
//  const SymbolTableEntry &inputDataVar = getMutableWeightVar(config, name);
////  printf("[copyMutableWeightVars] Copying data into mutable weight vars: %lu bytes\n",
////         inputDataVar.size*4);
////  printf( "inputDataVar.offset = %d\n", inputDataVar.offset);
//  memcpy(mutableWeightVarsAddr + inputDataVar.offset, inputT,
//         inputDataVar.size*4);
//
//  return mutableWeightVarsAddr;
//}


#define NHWC 1

/// The assumed layout of the area for mutable WeightVars is:
/// data | gpu_0/data | results
static uint8_t *initMutableWeightVars(const BundleConfig &config, char* varName) {
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

  if(NHWC == 1) {
    float* tmp = (float *)malloc(sizeof(float[1][DEFAULT_HEIGHT][DEFAULT_WIDTH][3]));
    memset(tmp, 0, imageDataSizeInBytes);
    transpose(inputT, tmp);
    memcpy(inputT, tmp, imageDataSizeInBytes);
    free(tmp);

  }

  memcpy(mutableWeightVarsAddr + inputGPUDataVar.offset, inputT,
         imageDataSizeInBytes);
  return mutableWeightVarsAddr;
}

static uint8_t *initActivations(const BundleConfig &config) {
  return static_cast<uint8_t *>(
      alignedAlloc(config, config.activationsMemSize));
}


static float* getInferenceResultVar(const BundleConfig &config,
                                    uint8_t *mutableWeightVars, const char* name) {
  const SymbolTableEntry &outputWeights = getMutableWeightVar(config, name);
  float *results = (float *)(mutableWeightVars + outputWeights.offset);
  return results;
}


int main(int argc, char **argv) {
  parseCommandLineOptions(argc, argv);

//  //======== single partition =======//
//  uint8_t *constantWeightVarsAddr =
//      initConstantWeights("resnet18v2.weights.bin", resnet18v2_config);
//  uint8_t *mutableWeightVarsAddr = initMutableWeightVars(resnet18v2_config, "data");
//  uint8_t *activationsAddr = initActivations(resnet18v2_config);
//
////  uint8_t *constantWeightVarsAddr =
////      initConstantWeights("mxnet_exported_resnet18.weights.bin", mxnet_exported_resnet18_config);
////  uint8_t *mutableWeightVarsAddr = initMutableWeightVars(mxnet_exported_resnet18_config, "data");
////  uint8_t *activationsAddr = initActivations(mxnet_exported_resnet18_config);
//
//  // Perform the computation.
//  int errCode =
////      mxnet_exported_resnet18(constantWeightVarsAddr, mutableWeightVarsAddr, activationsAddr);
//    resnet18v2(constantWeightVarsAddr, mutableWeightVarsAddr, activationsAddr);
//
//  if (errCode != GLOW_SUCCESS) {
//    printf("Error running bundle: error code %d\n", errCode);
//  }
//
//  printf("====== the value of a non-partitioned bundle ========\n");
//  // Report the results.
//  dumpInferenceResults(resnet18v2_config, mutableWeightVarsAddr, "resnetv22_dense0_fwd__1");
////  dumpInferenceResults(mxnet_exported_resnet18_config, mutableWeightVarsAddr, "resnetv10_dense0_fwd__1");
//
//  // Free all resources.
//  free(activationsAddr);
//  free(constantWeightVarsAddr);
//  free(mutableWeightVarsAddr);

  //======== multiple partitions =======//
  uint8_t *constantWeightVarsAddr0 =
      initConstantWeights("p0.weights.bin", p0_config);
  uint8_t *mutableWeightVarsAddr0 = initMutableWeightVars(p0_config, "data");
  uint8_t *activationsAddr0 = initActivations(p0_config);

  // Perform the computation.
  int errCode0 =
      p0(constantWeightVarsAddr0, mutableWeightVarsAddr0, activationsAddr0);
  if (errCode0 != GLOW_SUCCESS) {
    printf("Error running bundle: error code %d\n", errCode0);
  }

  float* resultVar0 = getInferenceResultVar(p0_config, mutableWeightVarsAddr0, "resnetv10_pool0_fwd__1");

  uint8_t *constantWeightVarsAddr1 =
      initConstantWeights("p1.weights.bin", p1_config);
  uint8_t *mutableWeightVarsAddr1 = copyMutableWeightVarsWithAlloc(p1_config, "resnetv10_pool0_fwd__1", resultVar0);
  uint8_t *activationsAddr1 = initActivations(p1_config);

  // Perform the computation.
  int errCode1 =
      p1(constantWeightVarsAddr1, mutableWeightVarsAddr1, activationsAddr1);
  if (errCode1 != GLOW_SUCCESS) {
    printf("Error running bundle: error code %d\n", errCode1);
  }

//   Report the results.
//  printf("====== the value of the final node ========\n");
//  dumpInferenceResults(p1_config, mutableWeightVarsAddr1, "resnetv10_dense0_fwd");

  float* resultVar1 = getInferenceResultVar(p1_config, mutableWeightVarsAddr1, "resnetv10_stage1_conv0_fwd__2");
  uint8_t *constantWeightVarsAddr2 =
      initConstantWeights("p2.weights.bin", p2_config);
  uint8_t *mutableWeightVarsAddr2 = copyMutableWeightVarsWithAlloc(p2_config, "resnetv10_pool0_fwd__1", resultVar0);
  copyMutableWeightVarsWithoutAlloc(p2_config, mutableWeightVarsAddr2, "resnetv10_stage1_conv0_fwd__2", resultVar1);
  uint8_t *activationsAddr2 = initActivations(p2_config);

  // Perform the computation.
  int errCode2 =
      p2(constantWeightVarsAddr2, mutableWeightVarsAddr2, activationsAddr2);
  if (errCode2 != GLOW_SUCCESS) {
    printf("Error running bundle: error code %d\n", errCode2);
  }


  // Report the results.
  printf("====== final result of this neural net ========\n");
 // dumpInferenceResults(p2_config, mutableWeightVarsAddr2, "resnetv22_dense0_fwd__1"); //for resnet18v2.onnx
  dumpInferenceResults(p2_config, mutableWeightVarsAddr2, "resnetv10_dense0_fwd"); //for mxnet_exported_resnet18.onnx

//   Free all resources.
  free(activationsAddr0);
  free(constantWeightVarsAddr0);
  free(mutableWeightVarsAddr0);

  free(activationsAddr1);
  free(constantWeightVarsAddr1);
  free(mutableWeightVarsAddr1);

  free(activationsAddr2);
  free(constantWeightVarsAddr2);
  free(mutableWeightVarsAddr2);

}
