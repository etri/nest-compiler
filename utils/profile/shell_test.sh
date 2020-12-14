#!/bin/bash
name=$1

# Declare an array of string with type
declare -a images=("cat_285.png" "dog_207.png" "zebra_340.png")

printf "value = %s\n" "${images[1]}"

if [ $name = "jemin" ]; then
  echo 'Welcome to Jemin'
else
  echo "you are not jemin"
fi

## onnx(caffe2) resnet50 quantization option
#for im in "${images[@]}"; do
#  ../../cmake-build-release/bin/image-classifier \
#  /home/jemin/development/nest_compiler/tests/images/imagenet/$im \
#  -m=/home/jemin/hdd/models/squeezenet/model.onnx \
#  -model-input-name=data_0 \
#  -image-mode=neg128to127 \
#  -backend=OpenCL \
#  -topk=5 \
#  -minibatch=1
#done


#for im in /home/jemin/development/nest_compiler/tests/images/imagenet/*.png; do
#  ../../cmake-build-release/bin/image-classifier \
#  $im \
#  -m=/home/jemin/hdd/models/squeezenet/model.onnx \
#  -model-input-name=data_0 \
#  -image-mode=neg128to127 \
#  -backend=OpenCL \
#  -topk=5 \
#  -minibatch=1
#done
