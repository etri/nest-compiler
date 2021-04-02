#!/bin/bash
input=$1
trap "exit" INT

declare -a validation="/ssd/imagenet/val_processed"
declare -a model="/ssd/nestc-data/glow_data/mobilenet_v2"


# caffe2 mobilenet2 Top#1 76.08 / 4011
# quantization per tensor
if [ $input -eq 1 ]
then
    python3 ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=$validation \
  --image-classifier-cmd="../../cmake-build-release/bin/image-classifier
  -m=$model
  -model-input-name=data
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -topk=5
  -minibatch=0 -"
fi


# ------------------- MobileNet ---------------------------
# per tensor
# Max Asymmetric
if [ $input -eq 2 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/mobilenet_v2/
-model-input-name=data
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_mobilenet_bin1000.yaml
-quantization-schema=asymmetric
-minibatch=0 -"
fi
# Max symmetric
if [ $input -eq 3 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/mobilenet_v2/
-model-input-name=data
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_mobilenet_bin1000.yaml
-quantization-schema=symmetric
-minibatch=0 -"
fi
# Max symmetric_with_uint8
if [ $input -eq 4 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/mobilenet_v2/
-model-input-name=data
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_mobilenet_bin1000.yaml
-quantization-schema=symmetric_with_uint8
-minibatch=0 -"
fi
# Max symmetric_with_power2_scale
if [ $input -eq 5 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/mobilenet_v2/
-model-input-name=data
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_mobilenet_bin1000.yaml
-quantization-schema=symmetric_with_power2_scale
-minibatch=0 -"
fi

# KL asymmetric
if [ $input -eq 6 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/mobilenet_v2/
-model-input-name=data
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_mobilenet_bin1000.yaml
-quantization-schema=asymmetric
-quantization-calibration=KL
-minibatch=0 -"
fi
# KL symmetric
if [ $input -eq 7 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/mobilenet_v2/
-model-input-name=data
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_mobilenet_bin1000.yaml
-quantization-schema=symmetric
-quantization-calibration=KL
-minibatch=0 -"
fi
# KL symmetric_with_uint8
# it is not worked under CPU and OpenCL backends.
if [ $input -eq 8 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/mobilenet_v2/
-model-input-name=data
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_mobilenet_bin1000.yaml
-quantization-schema=symmetric_with_uint8
-quantization-calibration=KL
-minibatch=0 -"
fi

# KL symmetric_with_power2_scale
if [ $input -eq 9 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/mobilenet_v2/
-model-input-name=data
-image-mode=0to1
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_mobilenet_bin1000.yaml
-quantization-schema=symmetric_with_power2_scale
-quantization-calibration=KL
-minibatch=0 -"
fi

#--- end of per-tensor MobileNet ---


# Channel wise quantization
# MAX, Per channelwise, Asymmetric
if [ $input -eq 20 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="../cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/mobilenet_v2/
-model-input-name=data
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_mobilenet_bin1000.yaml
-quantization-schema=asymmetric
-enable-channelwise
-minibatch=0 -"
fi

# KL, Per channelwise, Asymmetric
if [ $input -eq 21 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="../cmake-build-release/bin/image-classifier
  -m=/home/jemin/hdd/models/mobilenet_v2/
  -model-input-name=data
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -topk=5
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_mobilenet_bin1000.yaml
  -quantization-schema=asymmetric
  -quantization-calibration=KL
  -enable-channelwise
  -minibatch=0 -"
fi

# KL, Per channelwise, symmetric
if [ $input -eq 22 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="../cmake-build-release/bin/image-classifier
  -m=/home/jemin/hdd/models/mobilenet_v2/
  -model-input-name=data
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -topk=5
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_mobilenet_bin1000.yaml
  -quantization-schema=symmetric
  -quantization-calibration=KL
  -enable-channelwise
  -minibatch=0 -"
fi

# KL, Per channelwise, symmetric_with_uint8
if [ $input -eq 23 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="../cmake-build-release/bin/image-classifier
  -m=/home/jemin/hdd/models/mobilenet_v2/
  -model-input-name=data
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -topk=5
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_mobilenet_bin1000.yaml
  -quantization-schema=symmetric_with_uint8
  -quantization-calibration=KL
  -enable-channelwise
  -minibatch=0 -"
fi

# KL, Per channelwise, symmetric_with_power2_scale
if [ $input -eq 24 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="../cmake-build-release/bin/image-classifier
  -m=/home/jemin/hdd/models/mobilenet_v2/
  -model-input-name=data
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -topk=5
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_mobilenet_bin1000.yaml
  -quantization-schema=symmetric_with_power2_scale
  -quantization-calibration=KL
  -enable-channelwise
  -minibatch=0 -"
fi


