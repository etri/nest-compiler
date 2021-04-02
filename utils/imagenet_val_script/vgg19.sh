#!/bin/bash
input=$1
trap "exit" INT

# Declare an array of string with type
# "asymmetric" "symmetric" "symmetric_with_uint8" "symmetric_with_power2_scale"
declare -a quantizations=("asymmetric" "symmetric" "symmetric_with_uint8" "symmetric_with_power2_scale")
declare -a executors=("image-classifier" "image-classifier_googlenet_mixed_1")
declare -a clips=("none" "KL")
declare -a profiles=("goldfish_vgg19_bin1000" "1000_vgg19_bin1000" "10000_vgg19_bin1000")

declare -a validation="/ssd/imagenet/val_processed"
declare -a model="/ssd/nestc-data/glow_data/vgg19/model.onnx"
# -enable-channelwise
# -quantization-calibration=KL

##  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
# for testing: /home/jemin/development/dataset/small_imagenet

# top1: 69.54 (gluon vgg19 == 74)
if [ $input -eq 1 ]
then
  python ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
    -m=/home/jemin/hdd/models/vgg19/model.onnx
    -model-input-name=data_0
    -image-mode=neg128to127
    -use-imagenet-normalization
    -backend=OpenCL
    -topk=5
    -minibatch=0 -"
fi

# top1: 69.99
# top1: 3090 PC, 70.01
if [ $input -eq 2 ]
then
  python3 ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --validation-images-dir=$validation \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier
    -m=$model
    -model-input-name=data_0
    -image-mode=neg128to127
    -backend=CPU
    -topk=5
    -minibatch=0 -"
fi

# top1: 66.37% 255없는 버전으로 실행시.
# top1: 69.53%, 255를 넣고 실행. Imagenet-normalization
# stddev에도255 추가시 0% 정확도
if [ $input -eq 3 ]
then
  python ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --ignore-exist-file-check \
    --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
    -m=/home/jemin/hdd/models/vgg19/model.onnx
    -model-input-name=data_0
    -image-mode=neg128to127
    -mean=123.675,116.28,103.53
    -stddev=0.229,0.224,0.225
    -backend=OpenCL
    -topk=5
    -minibatch=0 -"
fi

# 3090 quantization
if [ $input -eq 4 ]
then
  python3 ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=$validation \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier
  -m=$model
  -model-input-name=data_0
  -image-mode=neg128to127
  -backend=CPU
  -topk=5
  -quantization-schema=symmetric
  -quantization-calibration=none
  -load-profile=/ssd/nestc-data/quantization/yaml_09/vgg19/goldfish_vgg19_bin1000.yaml
  -minibatch=0 -"
fi