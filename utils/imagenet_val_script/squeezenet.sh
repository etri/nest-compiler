#!/bin/bash
input=$1

#-------------------- SqueezeNet ----------------------------------------#
#FP 32
if [ $input -eq 1 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
  --validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
  --image-classifier-cmd="../cmake-build-release/bin/image-classifier
  -m=/home/jemin/hdd/models/squeezenet/model.onnx
  -model-input-name=data_0
  -image-mode=neg128to127
  -backend=OpenCL
  -topk=5
  -minibatch=0 -"
fi

# per tensor
# Max Asymmetric
if [ $input -eq 2 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_squeezenet_bin1000.yaml
-quantization-schema=asymmetric
-minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_squeezenet_bin1000.yaml
-quantization-schema=asymmetric
-minibatch=0 -"
fi

# per tensor
# Max Symmetric
if [ $input -eq 3 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_squeezenet_bin1000.yaml
-quantization-schema=symmetric
-minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_squeezenet_bin1000.yaml
-quantization-schema=symmetric
-minibatch=0 -"
fi

# per tensor
# Max symmetric_with_uint8
if [ $input -eq 4 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_squeezenet_bin1000.yaml
-quantization-schema=symmetric_with_uint8
-minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_squeezenet_bin1000.yaml
-quantization-schema=symmetric_with_uint8
-minibatch=0 -"
fi

# per tensor
# Max symmetric_with_power2_scale
if [ $input -eq 5 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_squeezenet_bin1000.yaml
-quantization-schema=symmetric_with_power2_scale
-minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_squeezenet_bin1000.yaml
-quantization-schema=symmetric_with_power2_scale
-minibatch=0 -"
fi

# per tensor
# KL Asymmetric
if [ $input -eq 6 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_squeezenet_bin1000.yaml
-quantization-schema=asymmetric
-quantization-calibration=KL
-minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_squeezenet_bin1000.yaml
-quantization-schema=asymmetric
-quantization-calibration=KL
-minibatch=0 -"
fi
# KL symmetric
if [ $input -eq 7 ]
then
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_squeezenet_bin1000.yaml
-quantization-schema=symmetric
-quantization-calibration=KL
-minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_squeezenet_bin1000.yaml
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
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=Interpreter
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_squeezenet_bin1000.yaml
-quantization-schema=symmetric_with_uint8
-quantization-calibration=KL
-minibatch=0 -"
    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=Interpreter
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_squeezenet_bin1000.yaml
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
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/1000_squeezenet_bin1000.yaml
-quantization-schema=symmetric_with_power2_scale
-quantization-calibration=KL
-minibatch=0 -"

    python ../imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=OpenCL
-topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_squeezenet_bin1000.yaml
-quantization-schema=symmetric_with_power2_scale
-quantization-calibration=KL
-minibatch=0 -"
fi

# Channel wise quantization
# MAX, Per channelwise, Asymmetric
if [ $input -eq 10 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=CPU
 -topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_squeezenet_bin1000.yaml
-quantization-schema=asymmetric
-enable-channelwise
-minibatch=0 -"
fi

# KL, Per channelwise, Asymmetric
if [ $input -eq 11 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=CPU
 -topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_squeezenet_bin1000.yaml
-quantization-schema=asymmetric
-quantization-calibration=KL
-enable-channelwise
-minibatch=0 -"
fi
# -keep-original-precision-for-nodes=Convolution,Relu,MaxPool,Concat,SoftMax,Reshape,AvgPool

# KL, Per channelwise, symmetric
if [ $input -eq 12 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=CPU
 -topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_squeezenet_bin1000.yaml
-quantization-schema=symmetric
-quantization-calibration=KL
-enable-channelwise
-minibatch=0 -"
fi

# KL, Per channelwise, symmetric_with_uint8
if [ $input -eq 13 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=CPU
 -topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_squeezenet_bin1000.yaml
-quantization-schema=symmetric_with_uint8
-quantization-calibration=KL
-enable-channelwise
-minibatch=0 -"
fi

# KL, Per channelwise, symmetric_with_power2_scale
if [ $input -eq 14 ]
then
    python3.6 imagenet_topk_accuracy_driver_py3.py --verbose \
--validation-images-dir=/home/jemin/development/dataset/imagenet2012_processed \
--image-classifier-cmd="/home/jemin/development/nest_compiler/cmake-build-release/bin/image-classifier
-m=/home/jemin/hdd/models/squeezenet/model.onnx
-model-input-name=data_0
-image-mode=neg128to127
-use-imagenet-normalization
-backend=CPU
 -topk=5
-load-profile=/home/jemin/development/nest_compiler/cmake-build-release/bin/10000_squeezenet_bin1000.yaml
-quantization-schema=symmetric_with_power2_scale
-quantization-calibration=KL
-enable-channelwise
-minibatch=0 -"
fi

