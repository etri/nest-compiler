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
declare -a profile_path="/ssd/nestc-data/quantization/yaml_09/vgg19/"

# -enable-channelwise
# -quantization-calibration=KL

##  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
# for testing: /home/jemin/development/dataset/small_imagenet
for profile in "${profiles[@]}"; do
    for clip in "${clips[@]}"; do
      for quant in "${quantizations[@]}"; do
      echo "test option: " $profile $clip $quant
      python3 ../imagenet_topk_accuracy_driver_py3.py --verbose \
      --validation-images-dir=$validation \
      --image-classifier-cmd="../../cmake-build-release/bin/image-classifier
      -m=$model
      -model-input-name=data_0
      -image-mode=neg128to127
      -backend=CPU
      -topk=5
      -quantization-schema=$quant
      -quantization-calibration=$clip
      -load-profile=$profile_path$profile.yaml
      -minibatch=0 -"
    done
  done
done
