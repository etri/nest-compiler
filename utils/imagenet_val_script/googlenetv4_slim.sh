#!/bin/bash
input=$1
trap "exit" INT

python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
--ignore-exist-file-check \
--image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-nonfusion
-m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
-model-input-name=input:0
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-image-layout=NHWC
-topk=5
-label-offset=1
-quantization-schema=asymmetric
-load-profile=/home/jemin/quantization/yaml_09/googlenet_v4_slim/10000_googlenet_v4_slim_bin1000.yaml
-minibatch=0 -"

#FP 32
if [ $input -eq 0 ]
then
  python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier
  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
  -model-input-name=input:0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=OpenCL
  -image-layout=NHWC
  -topk=5
  -label-offset=1
  -minibatch=0 -"
fi

# -- GoogleNet v4 slim, 299x299
# per tensor
# max asymmetric
# mixed-precision (Conv, Relu, FC)
if [ $input -eq 1 ]
then
  python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier_googlenet_mixed_1
  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
  -model-input-name=input:0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=OpenCL
  -image-layout=NHWC
  -topk=5
  -label-offset=1
  -quantization-schema=asymmetric
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_googlenet_v4_slim_bin1000.yaml
  -minibatch=0 -"

   python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier
  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
  -model-input-name=input:0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=OpenCL
  -image-layout=NHWC
  -topk=5
  -label-offset=1
  -quantization-schema=asymmetric
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_googlenet_v4_slim_bin1000.yaml
  -minibatch=0 -"
fi

# MAX Symmetric
if [ $input -eq 2 ]
then
#   python ../imagenet_topk_accuracy_driver_py3.py --verbose \
#  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
#  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier_googlenet_mixed_1
#  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
#  -model-input-name=input:0
#  -image-mode=0to1
#  -use-imagenet-normalization
#  -backend=OpenCL
#  -image-layout=NHWC
#  -topk=5
#  -label-offset=1
#  -quantization-schema=symmetric
#  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_googlenet_v4_slim_bin1000.yaml
#  -minibatch=0 -"

  python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
  -model-input-name=input:0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -image-layout=NHWC
  -topk=5
  -label-offset=1
  -enable-channelwise
  -quantization-schema=symmetric
  -quantization-calibration=none
  -load-profile=/home/jemin/quantization/yaml_09/googlenet_v4_slim/1000_googlenet_v4_slim_bin1000.yaml
  -minibatch=0 -"
fi

# Max Symmetric_uint8
if [ $input -eq 3 ]
then
#    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
#  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
#  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier_googlenet_mixed_1
#  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
#  -model-input-name=input:0
#  -image-mode=0to1
#  -use-imagenet-normalization
#  -backend=OpenCL
#  -image-layout=NHWC
#  -topk=5
#  -label-offset=1
#  -quantization-schema=symmetric_with_uint8
#  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_googlenet_v4_slim_bin1000.yaml
#  -minibatch=0 -"

  python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
  -model-input-name=input:0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -image-layout=NHWC
  -topk=5
  -label-offset=1
  -enable-channelwise
  -quantization-schema=symmetric_with_uint8
  -quantization-calibration=none
  -load-profile=/home/jemin/quantization/yaml_09/googlenet_v4_slim/1000_googlenet_v4_slim_bin1000.yaml
  -minibatch=0 -"
fi

# Max symmetric_with_power2_scale
if [ $input -eq 4 ]
then
  python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
  -model-input-name=input:0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -image-layout=NHWC
  -topk=5
  -label-offset=1
  -enable-channelwise
  -quantization-schema=symmetric_with_power2_scale
  -quantization-calibration=none
  -load-profile=/home/jemin/quantization/yaml_09/googlenet_v4_slim/1000_googlenet_v4_slim_bin1000.yaml
  -minibatch=0 -"
fi

# per tensor
# KL Asymmetric
if [ $input -eq 5 ]
then
   python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
  -model-input-name=input:0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -image-layout=NHWC
  -topk=5
  -label-offset=1
  -enable-channelwise
  -quantization-schema=asymmetric
  -quantization-calibration=KL
  -load-profile=/home/jemin/quantization/yaml_09/googlenet_v4_slim/1000_googlenet_v4_slim_bin1000.yaml
  -minibatch=0 -"
fi

# Per channelwise, KL, Symmetric
if [ $input -eq 6 ]
then
   python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
  -model-input-name=input:0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -image-layout=NHWC
  -topk=5
  -label-offset=1
  -enable-channelwise
  -quantization-schema=symmetric
  -quantization-calibration=KL
  -load-profile=/home/jemin/quantization/yaml_09/googlenet_v4_slim/1000_googlenet_v4_slim_bin1000.yaml
  -minibatch=0 -"
fi

# Per channelwise, KL, symmetric_with_uint8
if [ $input -eq 7 ]
then
   python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
  -model-input-name=input:0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -image-layout=NHWC
  -topk=5
  -label-offset=1
  -enable-channelwise
  -quantization-schema=symmetric_with_uint8
  -quantization-calibration=KL
  -load-profile=/home/jemin/quantization/yaml_09/googlenet_v4_slim/1000_googlenet_v4_slim_bin1000.yaml
  -minibatch=0 -"
fi

# # Per channelwise, KL,symmetric_with_uint8
if [ $input -eq 8 ]
then
   python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier-vtainterpreter-fusion
  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
  -model-input-name=input:0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -image-layout=NHWC
  -topk=5
  -label-offset=1
  -enable-channelwise
  -quantization-schema=symmetric_with_power2_scale
  -quantization-calibration=KL
  -load-profile=/home/jemin/quantization/yaml_09/googlenet_v4_slim/1000_googlenet_v4_slim_bin1000.yaml
  -minibatch=0 -"
fi

# Per channelwise, KL, symmetric_with_power2_scale
if [ $input -eq 9 ]
then
#    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
#  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
#  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier_googlenet_mixed_1
#  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
#  -model-input-name=input:0
#  -image-mode=0to1
#  -use-imagenet-normalization
#  -backend=CPU
#  -image-layout=NHWC
#  -topk=5
#  -label-offset=1
#  -quantization-schema=symmetric_with_power2_scale
#  -quantization-calibration=KL
#  -enable-channelwise
#  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_googlenet_v4_slim_bin1000.yaml
#  -minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier_googlenet_mixed_1
  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
  -model-input-name=input:0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -image-layout=NHWC
  -topk=5
  -label-offset=1
  -quantization-schema=symmetric_with_power2_scale
  -quantization-calibration=KL
  -enable-channelwise
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_googlenet_v4_slim_bin1000.yaml
  -minibatch=0 -"
fi