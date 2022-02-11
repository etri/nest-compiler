/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * Modifications copyright (C) 2022 <ETRI/Yongin Kwon>
 */

/*!
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */


#include "mxnet_exported_resnet18.h"
//#include <dlpack/dlpack.h>
//#include <tvm/runtime/module.h>
//#include <tvm/runtime/packed_func.h>
//#include <tvm/runtime/registry.h>


#include <cstring>
#include <cstdio>
#include <png.h>
#include <assert.h>
#include <math.h>
#include <iostream>
#include <limits>
#include <sys/time.h>




// \returns the index of the element at x,y,z,w.
size_t getXYZW(const size_t *dims, size_t x, size_t y, size_t z, size_t w) {
  return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
         (z * dims[3]) + w;
}

// \returns the index of the element at x,y,z.
size_t getXYZ(const size_t *dims, size_t x, size_t y, size_t z) {
  return (x * dims[1] * dims[2]) + (y * dims[2]) + z;
}

// Reads a PNG image from a file into a newly allocated memory block \p imageT
// representing a WxHxNxC tensor and returns it. The client is responsible for
// freeing the memory block.
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
  printf("%p w:%d h:%d c:%d\n",imageT,width,height,numChannels);

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


static int read_all(const char* file_description, const char* file_path, char** out_params,
                    size_t* params_size) {
  FILE* fp = fopen(file_path, "rb");
  if (fp == NULL) {
    return 2;
  }

  int error = 0;
  error = fseek(fp, 0, SEEK_END);
  if (error < 0) {
    return error;
  }

  long file_size = ftell(fp);
  if (file_size < 0) {
    return (int)file_size;
  } else if (file_size == 0 || file_size > 1000000000) {  // file size should be in (0, 20MB].
    char buf[128];
    snprintf(buf, sizeof(buf), "determing file size: %s,%d", file_path,file_size);
    perror(buf);
    return 2;
  }

  if (params_size != NULL) {
    *params_size = file_size;
  }

  printf("size:%d\n", *params_size);

  error = fseek(fp, 0, SEEK_SET);
  if (error < 0) {
    return error;
  }

  *out_params = (char*)malloc((unsigned long)file_size);
  if (fread(*out_params, file_size, 1, fp) != 1) {
    free(*out_params);
    *out_params = NULL;

    char buf[128];
    snprintf(buf, sizeof(buf), "reading: %s", file_path);
    perror(buf);
    return 2;
  }

  error = fclose(fp);
  if (error != 0) {
    free(*out_params);
    *out_params = NULL;
  }

  return 0;
}


#define NHWC 1

void transpose(float* &in, float* &out) {

  float (*inPtr)[1][3][224][224];
  float (*outPtr)[1][224][224][3];

  inPtr = (float (*)[1][3][224][224])in;
  outPtr = (float (*)[1][224][224][3])out;


  for (int j=0; j<3; j++){
    for (int k=0; k<224; k++){
      for (int l=0; l<224; l++){
        (*outPtr)[0][k][l][j] = (*inPtr)[0][j][k][l];
        }
      }
    }

}




int main(int argc, char** argv) {

  float *imgT{nullptr};

  std::pair<float, float> range = std::make_pair(0., 255.0);
  size_t loadDims[3];
  auto ttt = argv[1];
  bool loadSuccess = !readPngImage(argv[1], range, imgT, loadDims);
  assert(loadSuccess && "Error reading input image.");
  (void)loadSuccess;

  // png는 n h w c
  size_t inputDims[4];
  inputDims[0] = 1;
  inputDims[1] = 3;
  inputDims[2] = 224;
  inputDims[3] = 224;

  float *inputT;
  inputT = static_cast<float *>(malloc(1*3*224*224*sizeof(float)));

  // Convert to BGR, as this is what NN is expecting.
  for (unsigned z = 0; z < 3; z++) {
    for (unsigned y = 0; y < loadDims[1]; y++) {
      for (unsigned x = 0; x < loadDims[0]; x++) {
        inputT[getXYZW(inputDims,0, 2 - z, x, y)] = imgT[getXYZ(loadDims, x, y, z)];
      }
    }
  }

  // nhwc -> nchw
  if(NHWC == 1) {
    float* tmp = (float *)malloc(sizeof(float[1][224][224][3]));
    int imgSize= 1*224*224*3*sizeof(float);
    memset(tmp, 0,     imgSize);
    transpose(inputT, tmp);
    memcpy(inputT, tmp, imgSize);
    free(tmp);

  }  

  

    //set input
  uint8_t *mutableMem = (uint8_t*)malloc(mxnet_exported_resnet18_config.mutableWeightVarsMemSize);
  memcpy(mutableMem, inputT, sizeof(float)*1*224*224*3);
  
  mxnet_exported_resnet18_load_module(0);
  timeval t1, t2;
  gettimeofday(&t1, NULL);
  mxnet_exported_resnet18(0, mutableMem,0);
  gettimeofday(&t2, NULL);
  float output_storage[1000];
  memcpy(output_storage, mutableMem+mxnet_exported_resnet18_config.symbolTable[1].offset, mxnet_exported_resnet18_config.symbolTable[1].size * sizeof(float) );

  float max_iter = -std::numeric_limits<float>::max();
  int32_t max_index = -1;
  for (auto i = 0; i < 1000; ++i) {
    if (output_storage[i] > max_iter) {
      max_iter = output_storage[i];
      max_index = i;
    }
  }

  printf("Result: %u\n", max_index);
  printf("Confidence: %f\n", max_iter);
  double result;
  result = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)/1000.0;
  std::cout<<"Inference time: "<<result<<"ms\n";
  
  return 0;



}
