#!/bin/bash

input=$1
trap "exit" INT

# Declare an array of string with type
# "asymmetric" "symmetric" "symmetric_with_uint8" "symmetric_with_power2_scale"
# inputs = "goldfish.txt"
declare -a inputs=("input_goldfish.txt" "input_1000.txt" "input_10000.txt")
declare -a profiles=("goldfish_resnet18_bin1000.yaml" "1000_resnet18_bin1000.yaml" "10000_resnet18_bin1000.yaml")
# "goldfish_mxnet_resnet18_bin1000.yaml"
# -enable-channelwise
# -quantization-calibration=KL

##  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
# for testing: /home/jemin/development/dataset/small_imagenet
# VGG
#for ((i=0; i<4; i++))
#do
#  echo "${inputs[i]}"
#      /home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier-vtainterpreter-fusion \
#      -input-image-list-file=/home/jemin/development/nest_compiler/utils/"${inputs[i]}" \
#      -m=/home/jemin/hdd/models/vgg19/model.onnx \
#      -model-input-name=data_0 \
#      -image-mode=neg128to127 \
#      -topk=5 \
#      -minibatch=1 \
#      -dump-profile="${profiles[i]}" \
#      -num-histogram-bins=1000
#done

# RESNET18
for ((i=0; i<4; i++))
do
  echo "${inputs[i]}"
      /home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier \
      -input-image-list-file=/home/jemin/quantization/representative_image_list/"${inputs[i]}" \
      -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx \
      -model-input-name=data \
      -image-layout=NHWC \
      -image-mode=0to255 \
      -image-channel-order=RGB \
      -compute-softmax \
      -topk=5 \
      -minibatch=1 \
      -dump-profile="${profiles[i]}" \
      -num-histogram-bins=1000
done


## Shuffle, 1,000
#if [ $input -eq 1 ]
#then
#/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier \
#  -input-image-list-file=/home/jemin/development/nest_compiler/utils/input.txt \
#  -m=/home/jemin/development/dataset/glow_models/shufflenet.onnx \
#  -model-input-name=gpu_0/data_0 \
#  -image-mode=0to1 \
#  -use-imagenet-normalization \
#  -minibatch=1 \
#  -dump-profile=1000_shufflenet_bin1000.yaml \
#  -num-histogram-bins=1000
#fi
#
## squeezeNet, 1,0000
#if [ $input -eq 2 ]
#then
#/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier \
#  -input-image-list-file=/home/jemin/development/nest_compiler/utils/input.txt \
#  -m=/home/jemin/hdd/models/squeezenet/model.onnx \
#  -model-input-name=data_0 \
#  -image-mode=neg128to127 \
#  -use-imagenet-normalization \
#  -minibatch=1 \
#  -dump-profile=1000_squeezenet_bin1000.yaml \
#  -num-histogram-bins=1000
#fi
#
## squeezeNet, 1,0000
#if [ $input -eq 3 ]
#then
#/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier \
#  -input-image-list-file=/home/jemin/development/nest_compiler/utils/input_10000.txt \
#  -m=/home/jemin/hdd/models/squeezenet/model.onnx \
#  -model-input-name=data_0 \
#  -image-mode=neg128to127 \
#  -use-imagenet-normalization \
#  -minibatch=1 \
#  -dump-profile=10000_squeezenet_bin1000.yaml \
#  -num-histogram-bins=1000
#fi
#
## googlenet-v4, 1,000
#if [ $input -eq 4 ]
#then
#/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier \
#  -input-image-list-file=/home/jemin/development/nest_compiler/utils/input299_1000.txt \
#  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx \
#  -model-input-name=input:0 \
#  -image-mode=0to1 \
#  -image-layout=NHWC \
#  -label-offset=1 \
#  -use-imagenet-normalization \
#  -minibatch=1 \
#  -dump-profile=1000_googlenet_v4_slim_bin1000.yaml \
#  -num-histogram-bins=1000
#fi

