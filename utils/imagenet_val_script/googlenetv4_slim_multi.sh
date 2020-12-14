#!/bin/bash
input=$1
trap "exit" INT

# Declare an array of string with type
declare -a quantizations=("asymmetric" "symmetric" "symmetric_with_uint8" "symmetric_with_power2_scale")
declare -a small_quantizations=("symmetric" "symmetric_with_uint8" "symmetric_with_power2_scale")
declare -a executors=("image-classifier" "image-classifier_googlenet_mixed_1")
declare -a clips=("none" "KL")
declare -a profiles=("goldfish_googlenet_v4_slim_bin1000"
                      "1000_googlenet_v4_slim_bin1000"
                      "10000_googlenet_v4_slim_bin1000")

# -enable-channelwise
# -quantization-calibration=KL

##  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
# for testing: /home/jemin/development/dataset/small_imagenet_299
for profile in "${profiles[@]}"; do
  for clip in "${clips[@]}"; do
    for quant in "${quantizations[@]}"; do
    echo "test option: "$profile $clip $quant
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
  -quantization-schema=$quant
  -quantization-calibration=$clip
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/$profile.yaml
  -minibatch=0 -"
    done
  done
done

# 10,000, per channel, Max, three quantization methods.
for quant in "${small_quantizations[@]}"; do
  echo "test option: " $quant
  python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
--image-classifier-cmd="../../cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
-model-input-name=input:0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-image-layout=NHWC
-topk=5
-label-offset=1
-quantization-schema=$quant
-enable-channelwise
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_googlenet_v4_slim_bin1000.yaml
-minibatch=0 -"
done



#  ../../cmake-build-release/bin/image-classifier \
#  /home/jemin/development/nest_compiler/tests/images/imagenet/$im \
#  -m=/home/jemin/hdd/models/squeezenet/model.onnx \
#  -model-input-name=data_0 \
#  -image-mode=neg128to127 \
#  -backend=OpenCL \
#  -topk=5 \
#  -minibatch=1

#FP 32
#if [ $input -eq 0 ]
#then
#  python ../imagenet_topk_accuracy_driver_py3.py --verbose \



#  --validation-images-dir=/home/jemin/hdd/imagenet/val_processed_299 \
#  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier
#  -m=/home/jemin/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx
#  -model-input-name=input:0
#  -image-mode=0to1
#  -use-imagenet-normalization
#  -backend=OpenCL
#  -image-layout=NHWC
#  -topk=5
#  -label-offset=1
#  -minibatch=0 -"
#fi
