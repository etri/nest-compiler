#!/bin/bash
input=$1
trap "exit" INT

# 3090
declare -a validation="/ssd/imagenet/val_processed"
declare -a model="/ssd/nestc-data/glow_data/resnet50/model.onnx"

# 2080ti
declare -a model="/home/jemin/development/dataset/glow_models/caffe2_resnet50.onnx"

# onnx(caffe2) resnet50 quantization option
if [ $input -eq 1 ]
then
    python3 ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=$validation \
--image-classifier-cmd="../../cmake-build-release/bin/image-classifier
-m=$model
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-minibatch=0 -"
fi

if [ $input -eq 2 ]
then
    python3.6 ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/caffe2_resnet50.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/goldfish_resnet50_bin1000.yaml
-quantization-schema=symmetric
-minibatch=0 -"
fi

# Verify variance depending on Python version.
if [ $input -eq 3 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/caffe2_resnet50.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/goldfish_resnet50_bin1000.yaml
-quantization-schema=symmetric_with_power2_scale
-minibatch=0 -"
fi

if [ $input -eq 4 ]
then
    python3.6 ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/caffe2_resnet50.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/goldfish_resnet50_bin1000.yaml
-quantization-schema=asymmetric
-quantization-calibration=KL
-minibatch=0 -"
fi

if [ $input -eq 5 ]
then
    python3.6 ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/caffe2_resnet50.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/goldfish_resnet50_bin1000.yaml
-quantization-schema=symmetric
-quantization-calibration=KL
-minibatch=0 -"
fi

if [ $input -eq 6 ]
then
    python3.6 ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/caffe2_resnet50.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/goldfish_resnet50_bin1000.yaml
-quantization-schema=symmetric_with_uint8
-quantization-calibration=KL
-minibatch=0 -"
fi

if [ $input -eq 7 ]
then
    python3.6 ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/caffe2_resnet50.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/goldfish_resnet50_bin1000.yaml
-quantization-schema=symmetric_with_power2_scale
-quantization-calibration=KL
-minibatch=0 -"
fi

if [ $input -eq 8 ]
then
  python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier
  -m=/home/jemin/development/dataset/glow_models/caffe2_resnet50.onnx
  -model-input-name=gpu_0/data_0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=OpenCL
  -topk=5
  -quantization-schema=asymmetric
  -quantization-calibration=none
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/goldfish_resnet50_bin1000.yaml
  -minibatch=0 -"
fi

if [ $input -eq 9 ]
then
  python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier
  -m=/home/jemin/development/dataset/glow_models/caffe2_resnet50.onnx
  -model-input-name=gpu_0/data_0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=OpenCL
  -topk=5
  -quantization-schema=symmetric
  -quantization-calibration=none
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/goldfish_resnet50_bin1000.yaml
  -minibatch=0 -"
fi