#!/bin/bash
input=$1
trap "exit" INT

# Declare an array of string with type
# "asymmetric" "symmetric" "symmetric_with_uint8" "symmetric_with_power2_scale"
declare -a quantizations=("asymmetric" "symmetric" "symmetric_with_uint8" "symmetric_with_power2_scale")
declare -a executors=("image-classifier" "image-classifier_googlenet_mixed_1")
declare -a clips=("none" "KL")
declare -a profiles=("goldfish_mxnet_resnet18_bin1000" "1000_mxnet_resnet18_bin1000" "10000_mxnet_resnet18_bin1000")

# -enable-channelwise
# -quantization-calibration=KL

##  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
# for testing: /home/jemin/development/dataset/small_imagenet
# image-classifier: image-classifier-vtainterpreter-fusion, image-classifier-vtainterpreter-nonfusion

#--ignore-exist-file-check

# ----------- for parallels  ----------- #
for profile in "${profiles[@]}"; do
  for clip in "${clips[@]}"; do
      for quant in "${quantizations[@]}"; do
      echo "test option: " $profile $clip
      python ../imagenet_topk_accuracy_driver_py3.py --verbose \
      --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
      --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-nonfusion
      -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
      -model-input-name=data
      -image-mode=0to255
      -image-layout=NHWC
      -image-channel-order=RGB
      -compute-softmax
      -backend=OpenCL
      -topk=5
      -quantization-schema=$quant
      -quantization-calibration=$clip
      -load-profile=/home/jemin/quantization/yaml_09/resnet18/$profile.yaml
      -keep-original-precision-for-nodes=Div,Sub
      -minibatch=0 -"
    done
  done
done

# ----------- for parallels  ----------- #
for profile in "${profiles[@]}"; do
  for clip in "${clips[@]}"; do
      for quant in "${quantizations[@]}"; do
      echo "test option: " $profile $clip
      python ../imagenet_topk_accuracy_driver_py3.py --verbose \
      --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
      --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-nonfusion
      -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
      -model-input-name=data
      -image-mode=0to255
      -image-layout=NHWC
      -image-channel-order=RGB
      -compute-softmax
      -backend=CPU
      -topk=5
      -enable-channelwise
      -quantization-schema=$quant
      -quantization-calibration=$clip
      -load-profile=/home/jemin/quantization/yaml_09/resnet18/$profile.yaml
      -keep-original-precision-for-nodes=Div,Sub
      -minibatch=0 -"
    done
  done
done