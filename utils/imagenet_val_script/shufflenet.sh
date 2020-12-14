#!/bin/bash
input=$1

# original model for shufflenet
if [ $input -eq 1 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier-vtainterpreter-fusion
  -m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
  -model-input-name=gpu_0/data_0
  -use-imagenet-normalization
  -image-mode=0to1
  -backend=CPU
  -topk=5
  -minibatch=0 -"
fi

## ------------------------------------------------------------------------------------------------------------ ##
# per tensor
# Max Asymmetric
if [ $input -eq 2 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py  \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_shufflenet_bin1000.yaml
-quantization-schema=asymmetric
-minibatch=0 -"
fi

# Max Symmetric
if [ $input -eq 3 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_shufflenet_bin1000.yaml
-quantization-schema=symmetric
-minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_shufflenet_bin1000.yaml
-quantization-schema=symmetric
-minibatch=0 -"
fi
# Max symmetric_with_uint8
if [ $input -eq 4 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_shufflenet_bin1000.yaml
-quantization-schema=symmetric_with_uint8
-minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_shufflenet_bin1000.yaml
-quantization-schema=symmetric_with_uint8
-minibatch=0 -"
fi
# Max symmetric_with_power2_scale
if [ $input -eq 5 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_shufflenet_bin1000.yaml
-quantization-schema=symmetric_with_power2_scale
-minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_shufflenet_bin1000.yaml
-quantization-schema=symmetric_with_power2_scale
-minibatch=0 -"
fi

# KL Asymmetric
if [ $input -eq 6 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_shufflenet_bin1000.yaml
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
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_shufflenet_bin1000.yaml
-quantization-schema=symmetric
-quantization-calibration=KL
-minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_shufflenet_bin1000.yaml
-quantization-schema=symmetric
-quantization-calibration=KL
-minibatch=0 -"
fi

# KL symmetric_with_uint8
if [ $input -eq 8 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_shufflenet_bin1000.yaml
-quantization-schema=symmetric_with_uint8
-quantization-calibration=KL
-minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_shufflenet_bin1000.yaml
-quantization-schema=symmetric_with_uint8
-quantization-calibration=KL
-minibatch=0 -"
fi

# KL symmetric_with_power2_scale
if [ $input -eq 9 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_shufflenet_bin1000.yaml
-quantization-schema=symmetric_with_power2_scale
-quantization-calibration=KL
-minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
-model-input-name=gpu_0/data_0
-image-mode=0to1
-use-imagenet-normalization
-backend=CPU
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_shufflenet_bin1000.yaml
-quantization-schema=symmetric_with_power2_scale
-quantization-calibration=KL
-minibatch=0 -"
fi
#---- End of per tensor ----


# Channel wise quantization (Shuffle-Net)
# MAX, Per channelwise, Asymmetric
if [ $input -eq 9 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="../cmake-build-release/bin/image-classifier
  -m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
  -model-input-name=gpu_0/data_0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -topk=5
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_shufflenet_bin1000.yaml
  -quantization-schema=asymmetric
  -enable-channelwise
  -minibatch=0 -"
fi

# KL, Per channelwise, Asymmetric
if [ $input -eq 10 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="../cmake-build-release/bin/image-classifier
  -m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
  -model-input-name=gpu_0/data_0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -topk=5
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_shufflenet_bin1000.yaml
  -quantization-schema=asymmetric
  -quantization-calibration=KL
  -enable-channelwise
  -minibatch=0 -"
fi

# KL, Per channelwise, symmetric
if [ $input -eq 11 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="../cmake-build-release/bin/image-classifier
  -m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
  -model-input-name=gpu_0/data_0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -topk=5
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_shufflenet_bin1000.yaml
  -quantization-schema=symmetric
  -quantization-calibration=KL
  -enable-channelwise
  -minibatch=0 -"
fi

# KL, Per channelwise, symmetric_with_uint8
if [ $input -eq 12 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="../cmake-build-release/bin/image-classifier
  -m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
  -model-input-name=gpu_0/data_0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -topk=5
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_shufflenet_bin1000.yaml
  -quantization-schema=symmetric_with_uint8
  -quantization-calibration=KL
  -enable-channelwise
  -minibatch=0 -"
fi

# KL, Per channelwise, symmetric_with_power2_scale
if [ $input -eq 13 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="../cmake-build-release/bin/image-classifier
  -m=/home/jemin/development/dataset/glow_models/shufflenet.onnx
  -model-input-name=gpu_0/data_0
  -image-mode=0to1
  -use-imagenet-normalization
  -backend=CPU
  -topk=5
  -load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_shufflenet_bin1000.yaml
  -quantization-schema=symmetric_with_power2_scale
  -quantization-calibration=KL
  -enable-channelwise
  -minibatch=0 -"
fi

