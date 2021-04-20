/**
 * Copyright (c) 2017-present, Facebook, Inc.
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
/*
#include "glow/Base/Image.h"
#include "glow/Base/Tensor.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"

using namespace glow;
*/

#include "Image.h"
#include <png.h>
#include <iostream>

#include "stdlib.h"


//namespace glow {

/*
llvm::cl::OptionCategory imageCat("Image Processing Options");

ImageNormalizationMode imageNormMode;
static llvm::cl::opt<ImageNormalizationMode, true> imageNormModeF(
    "image-mode", llvm::cl::desc("Specify the image mode:"),
    llvm::cl::cat(imageCat), llvm::cl::location(imageNormMode),
    llvm::cl::values(clEnumValN(ImageNormalizationMode::kneg1to1, "neg1to1",
                                "Values are in the range: -1 and 1"),
                     clEnumValN(ImageNormalizationMode::k0to1, "0to1",
                                "Values are in the range: 0 and 1"),
                     clEnumValN(ImageNormalizationMode::k0to255, "0to255",
                                "Values are in the range: 0 and 255"),
                     clEnumValN(ImageNormalizationMode::kneg128to127,
                                "neg128to127",
                                "Values are in the range: -128 .. 127")),
    llvm::cl::init(ImageNormalizationMode::k0to255));
static llvm::cl::alias imageNormModeA("i",
                                      llvm::cl::desc("Alias for -image-mode"),
                                      llvm::cl::aliasopt(imageNormModeF),
                                      llvm::cl::cat(imageCat));

ImageChannelOrder imageChannelOrder;
static llvm::cl::opt<ImageChannelOrder, true> imageChannelOrderF(
    "image-channel-order", llvm::cl::desc("Specify the image channel order"),
    llvm::cl::Optional, llvm::cl::cat(imageCat),
    llvm::cl::location(imageChannelOrder),
    llvm::cl::values(clEnumValN(ImageChannelOrder::BGR, "BGR", "Use BGR"),
                     clEnumValN(ImageChannelOrder::RGB, "RGB", "Use RGB")),
    llvm::cl::init(ImageChannelOrder::BGR));

ImageLayout imageLayout;
static llvm::cl::opt<ImageLayout, true>
    imageLayoutF("image-layout",
                 llvm::cl::desc("Specify which image layout to use"),
                 llvm::cl::Optional, llvm::cl::cat(imageCat),
                 llvm::cl::location(imageLayout),
                 llvm::cl::values(clEnumValN(ImageLayout::NCHW, "NCHW",
                                             "Use NCHW image layout"),
                                  clEnumValN(ImageLayout::NHWC, "NHWC",
                                             "Use NHWC image layout")),
                 llvm::cl::init(ImageLayout::NCHW));
static llvm::cl::alias imageLayoutA("l",
                                    llvm::cl::desc("Alias for -image-layout"),
                                    llvm::cl::aliasopt(imageLayoutF),
                                    llvm::cl::cat(imageCat));

bool useImagenetNormalization;
static llvm::cl::opt<bool, true> useImagenetNormalizationF(
    "use-imagenet-normalization",
    llvm::cl::desc("Use Imagenet Normalization. This works in combination "
                   "with the Image Mode normalization."),
    llvm::cl::cat(imageCat), llvm::cl::location(useImagenetNormalization),
    llvm::cl::init(false));

llvm::cl::list<float> meanValues(
    "mean",
    llvm::cl::desc("Mean values m1,m2,m3..."
                   "Count must be equal to number of input channels."),
    llvm::cl::value_desc("float"), llvm::cl::ZeroOrMore,
    llvm::cl::CommaSeparated, llvm::cl::cat(imageCat));

llvm::cl::list<float> stddevValues(
    "stddev",
    llvm::cl::desc("Standard deviation values s1,s2,s3..."
                   "Count must be equal to number of input channels."),
    llvm::cl::value_desc("float"), llvm::cl::ZeroOrMore,
    llvm::cl::CommaSeparated, llvm::cl::cat(imageCat));
} // namespace glow

/// Convert the normalization to numeric floating poing ranges.
std::pair<float, float> glow::normModeToRange(ImageNormalizationMode mode) {
  switch (mode) {
  case ImageNormalizationMode::kneg1to1:
    return {-1., 1.};
  case ImageNormalizationMode::k0to1:
    return {0., 1.0};
  case ImageNormalizationMode::k0to255:
    return {0., 255.0};
  case ImageNormalizationMode::kneg128to127:
    return {-128., 127.};
  default:
    GLOW_ASSERT(false && "Image format not defined.");
  }
}
*/
std::tuple<size_t, size_t, bool> getPngInfo(const char *filename) {
  // open file and test for it being a png.
  FILE *fp = fopen(filename, "rb");
  if(fp == 0) {
	  cout << "Can't open image file." << endl;
  }
  //GLOW_ASSERT(fp && "Can't open image file.");

  unsigned char header[8];
  size_t fread_ret = fread(header, 1, 8, fp);
  if(fread_ret != 8 && png_sig_cmp(header, 0, 8) != 0) {
	  cout << "Invalid image file signature." << endl;
  }
  //GLOW_ASSERT(fread_ret == 8 && !png_sig_cmp(header, 0, 8) &&
  //            "Invalid image file signature.");

  // Initialize stuff.
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if(png_ptr == 0)
	  cout << "Image initialization failed." << endl;
 // GLOW_ASSERT(png_ptr && "Image initialization failed.");

  png_infop info_ptr = png_create_info_struct(png_ptr);
  //GLOW_ASSERT(info_ptr && "Could not get png info.");
  if(info_ptr == 0)
	  cout << "Could not get png info." << endl;

  int sjmpGetPtr = setjmp(png_jmpbuf(png_ptr));
  //GLOW_ASSERT(!sjmpGetPtr && "Failed getting png_ptr.");
  if(sjmpGetPtr != 0)
	  cout << "Failed getting png_ptr." << endl;

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);

  size_t height = png_get_image_height(png_ptr, info_ptr);
  size_t width = png_get_image_width(png_ptr, info_ptr);
  png_byte color_type = png_get_color_type(png_ptr, info_ptr);

  const bool isGray = color_type == PNG_COLOR_TYPE_GRAY;

  png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
  fclose(fp);

  return std::make_tuple(height, width, isGray);
}


/*
bool glow::readPngImage(Tensor *T, const char *filename,
                        std::pair<float, float> range,
                        llvm::ArrayRef<float> mean,
                        llvm::ArrayRef<float> stddev) {
*/

//bool readPngImage(float data[1][224][224][3], const char *filename,
//                 std::pair<float, float> range,
//                  vector<float> mean,
//                  vector<float> stddev)
bool readPngImage(float data[1][3][224][224], const char *filename,
                  std::pair<float, float> range,
                  vector<float> mean,
                  vector<float> stddev)
{
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
    fclose(fp);
    return true;
  }
  if (png_sig_cmp(header, 0, 8)) {
    fclose(fp);
    return true;
  }

  // Initialize stuff.
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    fclose(fp);
    return true;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
    fclose(fp);
    return true;
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
    fclose(fp);
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

  if(bit_depth != 8)
	  cout << "Invalid image" << endl;

  if(!(color_type == PNG_COLOR_TYPE_RGB_ALPHA ||
          color_type == PNG_COLOR_TYPE_RGB || isGray))
	  cout << "Invalid image" << endl;

  (void)bit_depth;
  if(bit_depth != 8) {
	  cout << "invalid image" << endl;
	  return true;
  }
  if(color_type != PNG_COLOR_TYPE_RGB_ALPHA &&
          color_type != PNG_COLOR_TYPE_RGB && !isGray) {
	  cout << "invalid image" << endl;
  	  return true;
	}

  bool hasAlpha = (color_type == PNG_COLOR_TYPE_RGB_ALPHA);

  int number_of_passes = png_set_interlace_handling(png_ptr);

  if(number_of_passes != 1)
	  cout << "Invalid image" << endl;

//  (void)number_of_passes;
  if(number_of_passes != 1) {
	  cout << "Invalid image" << endl;
	  return true;
  }

  png_read_update_info(png_ptr, info_ptr);

  // Error during image read.
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
    fclose(fp);
    return true;
  }

  auto *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
  for (size_t y = 0; y < height; y++) {
    row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }

  png_read_image(png_ptr, row_pointers);
  png_read_end(png_ptr, info_ptr);



  //T->reset(ElemKind::FloatTy, {height, width, numChannels});
  //auto H = T->getHandle<>();

  float scale = ((range.second - range.first) / 255.0);
  float bias = range.first;

  for (size_t row_n = 0; row_n < height; row_n++) {
    png_byte *row = row_pointers[row_n];
    for (size_t col_n = 0; col_n < width; col_n++) {
      png_byte *ptr =
          &(row[col_n * (hasAlpha ? (numChannels + 1) : numChannels)]);
      for (size_t i = 0; i < numChannels; i++) {
        float val = float(ptr[i]);
        val = (val - mean[i]) / stddev[i];
        //H.at({row_n, col_n, i}) = val * scale + bias;

        //msyu
        data[0][i][row_n][col_n] = val * scale + bias;
      }
    }
  }

  for (size_t y = 0; y < height; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
  fclose(fp);

  return false;
}

/*
bool glow::writePngImage(Tensor *T, const char *filename,
                         std::pair<float, float> range,
                         llvm::ArrayRef<float> mean,
                         llvm::ArrayRef<float> stddev) {
*/
bool writePngImage(float data[1][1][672][672], const char *filename,
                   std::pair<float, float> range,
                   vector<float> mean,
                   vector<float> stddev, vector<size_t> odim)
{
  /* create file */
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    return true;
  }

  /* initialize stuff */
  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

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

  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  //auto H = T->getHandle<>();

  //auto odim = H.dims();
  constexpr size_t numChannels = 3;


  if(odim[2] != numChannels)
	  cout << "Currently only supports saving RGB images without alpha." << endl;
  /*
  assert(odim[2] == numChannels &&
         "Currently only supports saving RGB images without alpha.");
	*/


  size_t width = odim[0];
  size_t height = odim[1];
  int color_type = PNG_COLOR_TYPE_RGB_ALPHA;
  int bit_depth = 8;

  png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
               PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  auto *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
  for (size_t y = 0; y < height; y++) {
    row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }

  float scale = ((range.second - range.first) / 255.0);
  float bias = range.first;

  for (size_t y = 0; y < height; y++) {
    png_byte *row = row_pointers[y];
    for (size_t x = 0; x < width; x++) {
      png_byte *ptr = &(row[x * 4]);
      for (size_t i = 0; i < numChannels; i++) {
        //float val = (H.at({y, x, i}) - bias) / scale;
    	  //msyy
    	   float val = (data[0][y][x][i] - bias) / scale;

        val = (val * stddev[i]) + mean[i];
        ptr[i] = val;
      }
      ptr[3] = 0xff;
    }
  }

  png_write_image(png_ptr, row_pointers);

  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  png_write_end(png_ptr, nullptr);

  /* cleanup heap allocation */
  for (size_t y = 0; y < height; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);

  return false;
}
//}
