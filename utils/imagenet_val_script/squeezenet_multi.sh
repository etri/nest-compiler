#!/bin/bash

input=$1
trap "exit" INT

# Declare an array of string with type
# "asymmetric" "symmetric" "symmetric_with_uint8" "symmetric_with_power2_scale"
declare -a quantizations=("asymmetric" "symmetric" "symmetric_with_uint8" "symmetric_with_power2_scale")
declare -a executors=("image-classifier" "image-classifier_googlenet_mixed_1")
declare -a clips=("none" "KL")
declare -a profiles=("10000_squeezenet_bin1000")

# -enable-channelwise
# -quantization-calibration=KL

##  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
# for testing: /home/jemin/development/dataset/small_imagenet
for profile in "${profiles[@]}"; do
  for clip in "${clips[@]}"; do
    for quant in "${quantizations[@]}"; do
      echo "test option: " $profile $clip $quant
      python ../imagenet_topk_accuracy_driver_py3.py --verbose \
      --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
      --image-classifier-cmd="../../cmake-build-release/bin/image-classifier
      -m=/home/jemin/hdd/models/squeezenet/model.onnx
      -model-input-name=data_0
      -image-mode=neg128to127
      -use-imagenet-normalization
      -backend=CPU
      -topk=5
      -quantization-schema=$quant
      -quantization-calibration=$clip
      -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/$profile.yaml
      -enable-channelwise
      -minibatch=0 -"
    done
  done
done