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
# for small testing: /ssd/dataset/small_imagenet
# for exhaustive test: /ssd/dataset/imagenet2012_processed
# image-classifier: image-classifier-vtainterpreter-fusion, image-classifier-vtainterpreter-nonfusion

#--ignore-exist-file-check

# mixed precision test
declare -i i=0
#printf "" > /home/user/development/nest-compiler/cmake-build-release/glow/bin/VTASkipQuantizeNodes.txt
printf "" > /home/user/development/nest-compiler/utils/imagenet_val_script/VTASkipQuantizeNodes.txt
while read node; do
  grep -v $node ./resnet18_graph.txt > ./VTASkipQuantizeNodes.txt
  diff resnet18_graph.txt VTASkipQuantizeNodes.txt
  echo "data: " $i
  python ./imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/ssd/dataset/imagenet2012_processed \
  --ignore-exist-file-check \
  --filename=$node \
  --image-classifier-cmd="../../cmake-build-release/glow/bin/${executors[0]}
  -m=/ssd/dataset/glow_models/mxnet_exported_resnet18.onnx
  -model-input-name=data
  -image-mode=0to255
  -image-layout=NHWC
  -image-channel-order=RGB
  -compute-softmax
  -backend=VTA
  -dump-graph-DAG=./opts/$i.dot
  -topk=5
  -quantization-schema=symmetric_with_power2_scale
  -quantization-calibration=none
  -load-profile=/home/user/development/nest-compiler/cmake-build-release/glow/bin/test.yaml
  -keep-original-precision-for-nodes=Div,Sub
  -minibatch=0 -"
  ((i++))
done < resnet18_graph.txt