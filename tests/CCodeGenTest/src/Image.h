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
#ifndef IMAGE_H
#define IMAGE_H


//"Please install libpng"

#include <tuple>
#include <vector>

using namespace std;
//namespace glow {

/// Pixel value ranges.
enum class ImageNormalizationMode {
  kneg1to1,     // Values are in the range: -1 and 1.
  k0to1,        // Values are in the range: 0 and 1.
  k0to255,      // Values are in the range: 0 and 255.
  kneg128to127, // Values are in the range: -128 .. 127
};

/// Layout of image dimensions (batch, channels, height, width).
enum class ImageLayout {
  NCHW,
  NHWC,
};

/// Order of color channels (red, green, blue).
enum class ImageChannelOrder {
  BGR,
  RGB,
};

/// -image-mode flag.
extern ImageNormalizationMode imageNormMode;

/// -image-channel-order flag.
extern ImageChannelOrder imageChannelOrder;

/// -image-layout flag.
extern ImageLayout imageLayout;

/// -use-imagenet-normalization flag.
extern bool useImagenetNormalization;

/// These are standard normalization factors for imagenet, adjusted for
/// normalizing values in the 0to255 range instead of 0to1, as seen at:
/// https://github.com/pytorch/examples/blob/master/imagenet/main.py
//static const float imagenetNormMean[] = {0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0};
static const vector<float> imagenetNormMean = {0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0};
//static const float imagenetNormStd[] = {0.229, 0.224, 0.225};
static const vector<float> imagenetNormStd = {0.229, 0.224, 0.225};

/// Default values for mean and stddev.
#define MAX_TENSOR_DIM 6
static const std::vector<float> zeroMean(MAX_TENSOR_DIM, 0.f);
static const std::vector<float> oneStd(MAX_TENSOR_DIM, 1.f);
static const std::pair<float, float> zeroOne(0.0, 1.0);

/// \returns the floating-point range corresponding to enum value \p mode.
std::pair<float, float> normModeToRange(ImageNormalizationMode mode);

/// Reads a png image header from png file \p filename and \returns a tuple
/// containing height, width, and a bool if it is grayscale or not.
std::tuple<size_t, size_t, bool> getPngInfo(const char *filename);

/// Reads a png image. \returns True if an error occurred. The values of the
/// image are in the range \p range.
/*
bool readPngImage(Tensor *T, const char *filename,
                  std::pair<float, float> range,
                  llvm::ArrayRef<float> mean = zeroMean,
                  llvm::ArrayRef<float> stddev = oneStd);
*/

bool readPngImage(float data[1][3][224][224], const char *filename,
                  std::pair<float, float> range = zeroOne,
                  vector<float> mean = zeroMean,
                  vector<float> stddev = oneStd);
/// Writes a png image. \returns True if an error occurred. The values of the
/// image are in the range \p range.
/*
bool writePngImage(Tensor *T, const char *filename,
                   std::pair<float, float> range,
                   llvm::ArrayRef<float> mean = zeroMean,
                   llvm::ArrayRef<float> stddev = oneStd);
*/

bool writePngImage(float data[1][1][672][672], const char *filename,
                   std::pair<float, float> range = zeroOne,
                   vector<float> mean = zeroMean,
                   vector<float> stddev = oneStd,  vector<size_t> odim = {1,1,672,672});// vector<size_t> odim = {224,224,3,1});


//} // namespace glow

#endif // IMAGE_H
