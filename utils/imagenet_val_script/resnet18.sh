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


# FP32
# if   -use-imagenet-normalization option is added, the accuracy is 16.82% with imagenet12 val.
if [ $input -eq 0 ]
then
  python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --fail-list \
  --ignore-exist-file-check \
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
  -minibatch=0 -"
fi

# DeepX Test
if [ $input -eq 99 ]
then
  python3 ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/ssd/imagenet/val_processed \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier
  -m=/ssd/nestc-data/glow_data/mxnet_exported_resnet18.onnx
  -model-input-name=data
  -image-mode=0to255
  -image-layout=NHWC
  -image-channel-order=RGB
  -compute-softmax
  -backend=OpenCL
  -topk=5
  -minibatch=0 -"
fi

# DeepX Test
if [ $input -eq 98 ]
then
  python3 ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/ssd/imagenet/val_processed \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier
  -m=/ssd/nestc-data/glow_data/mxnet_exported_resnet18.onnx
  -model-input-name=data
  -image-mode=0to255
  -image-layout=NHWC
  -image-channel-order=RGB
  -compute-softmax
  -quantization-schema=symmetric_with_power2_scale
  -quantization-calibration=none
  -load-profile=/home/jemin/development/nest-compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18_calib_1.yaml
  -backend=VTAInterpreter
  -topk=5
  -minibatch=0 -"
fi

# CPU
if [ $input -eq 1 ]
then
  python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --fail-list \
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
  -minibatch=0 -"
fi

if [ $input -eq 2 ]
then
  for clip in "${clips[@]}"; do
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --fail-list \
    --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-nonfusion
    -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
    -model-input-name=data
    -image-mode=0to255
    -image-layout=NHWC
    -image-channel-order=RGB
    -compute-softmax
    -backend=VTAInterpreter
    -topk=5
    -quantization-schema=symmetric_with_power2_scale
    -quantization-calibration=$clip
    -load-profile=/home/jemin/quantization/yaml_09/resnet18/${profiles[0]}.yaml
    -minibatch=0 -"
  done
fi

if [ $input -eq 3 ]
then
  for clip in "${clips[@]}"; do
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --ignore-exist-file-check \
    --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-nonfusion
    -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
    -model-input-name=data
    -image-mode=0to255
    -image-layout=NHWC
    -image-channel-order=RGB
    -compute-softmax
    -backend=VTAInterpreter
    -topk=5
    -quantization-schema=symmetric_with_power2_scale
    -quantization-calibration=${clips[0]}
    -load-profile=/home/jemin/quantization/yaml_09/resnet18/${profiles[1]}.yaml
    -minibatch=0 -"
  done
fi

if [ $input -eq 4 ]
then
  for clip in "${clips[@]}"; do
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --ignore-exist-file-check \
    --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-nonfusion
    -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
    -model-input-name=data
    -image-mode=0to255
    -image-layout=NHWC
    -image-channel-order=RGB
    -compute-softmax
    -backend=VTAInterpreter
    -topk=5
    -quantization-schema=symmetric_with_power2_scale
    -quantization-calibration=${clips[1]}
    -load-profile=/home/jemin/quantization/yaml_09/resnet18/${profiles[2]}.yaml
    -minibatch=0 -"
  done
fi

if [ $input -eq 5 ]
then
  for clip in "${clips[@]}"; do
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --fail-list \
    --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
    -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
    -model-input-name=data
    -image-mode=0to255
    -image-layout=NHWC
    -image-channel-order=RGB
    -compute-softmax
    -backend=VTAInterpreter
    -topk=5
    -quantization-schema=symmetric_with_power2_scale
    -quantization-calibration=$clip
    -load-profile=/home/jemin/quantization/yaml_09/resnet18/${profiles[0]}.yaml
    -minibatch=0 -"
  done
fi

if [ $input -eq 6 ]
then
  for clip in "${clips[@]}"; do
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --fail-list \
    --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
    -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
    -model-input-name=data
    -image-mode=0to255
    -image-layout=NHWC
    -image-channel-order=RGB
    -compute-softmax
    -backend=VTAInterpreter
    -topk=5
    -quantization-schema=symmetric_with_power2_scale
    -quantization-calibration=$clip
    -load-profile=/home/jemin/quantization/yaml_09/resnet18/${profiles[1]}.yaml
    -minibatch=0 -"
  done
fi

#--fail-list \
if [ $input -eq 7 ]
then
  for clip in "${clips[@]}"; do
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --ignore-exist-file-check \
    --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
    -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
    -model-input-name=data
    -image-mode=0to255
    -image-layout=NHWC
    -image-channel-order=RGB
    -compute-softmax
    -backend=VTAInterpreter
    -topk=5
    -quantization-schema=symmetric_with_power2_scale
    -quantization-calibration=${clips[1]}
    -load-profile=/home/jemin/quantization/yaml_09/resnet18/${profiles[2]}.yaml
    -minibatch=0 -"
  done
fi


# GPU backend (2080ti)
if [ $input -eq 8 ]
then
  for clip in "${clips[@]}"; do
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --ignore-exist-file-check \
    --fail-list \
    --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
    -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
    -model-input-name=data
    -image-mode=0to255
    -image-layout=NHWC
    -image-channel-order=RGB
    -compute-softmax
    -backend=OpenCL
    -topk=5
    -quantization-schema=symmetric_with_power2_scale
    -quantization-calibration=$clip
    -load-profile=/home/jemin/quantization/yaml_09/resnet18/${profiles[2]}.yaml
    -minibatch=0 -"
  done
fi

if [ $input -eq 9 ]
then
  for clip in "${clips[@]}"; do
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --fail-list \
    --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
    -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
    -model-input-name=data
    -image-mode=0to255
    -image-layout=NHWC
    -image-channel-order=RGB
    -compute-softmax
    -backend=OpenCL
    -topk=5
    -quantization-schema=symmetric
    -quantization-calibration=$clip
    -load-profile=/home/jemin/quantization/yaml_09/resnet18/${profiles[2]}.yaml
    -minibatch=0 -"
  done
fi

if [ $input -eq 10 ]
then
  for clip in "${clips[@]}"; do
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --fail-list \
    --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
    -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
    -model-input-name=data
    -image-mode=0to255
    -image-layout=NHWC
    -image-channel-order=RGB
    -compute-softmax
    -backend=OpenCL
    -topk=5
    -quantization-schema=symmetric_with_uint8
    -quantization-calibration=$clip
    -load-profile=/home/jemin/quantization/yaml_09/resnet18/${profiles[2]}.yaml
    -minibatch=0 -"
  done
fi

if [ $input -eq 11 ]
then
  for clip in "${clips[@]}"; do
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --fail-list \
    --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
    -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
    -model-input-name=data
    -image-mode=0to255
    -image-layout=NHWC
    -image-channel-order=RGB
    -compute-softmax
    -backend=OpenCL
    -topk=5
    -quantization-schema=symmetric_with_power2_scale
    -quantization-calibration=$clip
    -load-profile=/home/jemin/quantization/yaml_09/resnet18/${profiles[2]}.yaml
    -minibatch=0 -"
  done
fi

if [ $input -eq 12 ]
then
  for clip in "${clips[@]}"; do
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
    --fail-list \
    --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
    -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
    -model-input-name=data
    -image-mode=0to255
    -image-layout=NHWC
    -image-channel-order=RGB
    -compute-softmax
    -backend=OpenCL
    -topk=5
    -quantization-schema=symmetric_with_power2_scale
    -quantization-calibration=$clip
    -load-profile=/home/jemin/quantization/yaml_09/resnet18/${profiles[2]}.yaml
    -minibatch=0 -"
  done
fi


# ----------- for parallels  ----------- #
#for profile in "${profiles[@]}"; do
#  for clip in "${clips[@]}"; do
#    echo "test option: " $profile $clip
#    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
#    --validation-images-dir=/home/jemin/development/dataset/small_imagenet \
#    --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-nonfusion
#    -m=/home/jemin/development/nest_compiler/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
#    -model-input-name=data
#    -image-mode=0to255
#    -image-layout=NHWC
#    -compute-softmax
#    -backend=VTAInterpreter
#    -topk=5
#    -quantization-schema=symmetric_with_power2_scale
#    -quantization-calibration=$clip
#    -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/$profile.yaml
#    -minibatch=0 -"
#  done
#done
