#!/bin/bash
input=$1
trap "exit" INT

# Declare an array of string with type
# "asymmetric" "symmetric" "symmetric_with_uint8" "symmetric_with_power2_scale"
declare -a quantizations=("asymmetric" "symmetric" "symmetric_with_uint8" "symmetric_with_power2_scale")
declare -a executors=("image-classifier" "image-classifier_googlenet_mixed_1")
declare -a clips=("none" "KL")
declare -a profiles=("goldfish_resnet50_bin1000" "1000_resnet50_bin1000")
# "goldfish_resnet50_bin1000" "1000_resnet50_bin1000" "10000_resnet50_bin1000"

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
      --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
      -m=/home/jemin/development/dataset/glow_models/caffe2_resnet50.onnx
      -model-input-name=gpu_0/data_0
      -image-mode=0to1
      -use-imagenet-normalization
      -backend=CPU
      -topk=5
      -quantization-schema=$quant
      -quantization-calibration=$clip
      -load-profile=/home/jemin/quantization/yaml_09/resnet50/$profile.yaml
      -enable-channelwise
      -minibatch=0 -"
    done
  done

done
