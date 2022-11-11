option(VTA_RESNET18_WITH_SKIPQUANT0 "Build the LLVM-based JIT CPU backend" OFF)
option(NESTC_USE_PRECOMPILED_BUNDLE_FROM_AWS "Download bundles from AWS S3" OFF)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.onnx
        COMMAND
        mkdir -p ${NESTC_BINARY_DIR}/models
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/vgg16_cifar10.onnx ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.onnx --no-sign-request
)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vgg16_calib_1.yaml
            COMMAND
            image-classifier ${NESTC_ROOT_DIR}/examples/bundles/vgg16_cifar10/cifar10/test/airplane/0001.png
            -m=${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.onnx
            -model-input-name=input
            -backend=VTAInterpreter #VTA
            #-dump-graph-DAG="opt.dot"
            -image-layout=NCHW #NHWC
            -compute-softmax
            -topk=5
            -image-mode=0to1
            #-image-channel-order=RGB
            -dump-profile=${CMAKE_CURRENT_BINARY_DIR}/vgg16_calib_1.yaml
            -quantization-schema="symmetric_with_power2_scale"
            DEPENDS image-classifier ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.onnx
    )

    if(NOT VTA_VGG16_WITH_SKIPQUANT0)
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.h
                ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.weights.bin
                ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                COMMAND
                model-compiler -g
                -model=${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.onnx
                -backend=VTA
                -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
                -bundle-api=dynamic
                -model-input=input,float,[1,3,32,32] #data,float,[1,32,32,3]
                #-dump-graph-DAG=opt.dot
                -load-profile=${CMAKE_CURRENT_BINARY_DIR}/vgg16_calib_1.yaml
                -quantization-schema=symmetric_with_power2_scale
                -instrument-debug
                -instrument-debug-format=rawbin
                DEPENDS
                model-compiler VTA ${CMAKE_CURRENT_BINARY_DIR}/vgg16_calib_1.yaml #Interpreter
        )
    else()
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.h
                ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.weights.bin
                ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                ${CMAKE_CURRENT_BINARY_DIR}/VTASkipQuantizeNodes.txt
                COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/VTASkipQuantizeNodes.txt
                ${CMAKE_CURRENT_BINARY_DIR}/VTASkipQuantizeNodes.txt
                COMMAND
                model-compiler -g
                -model=${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.onnx
                -backend=VTA
                -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
                -bundle-api=dynamic
                -model-input=input,float,[1,3,32,32] #data,float,[1,32,32,3]
                #-dump-graph-DAG=opt.dot
                -load-profile=${CMAKE_CURRENT_BINARY_DIR}/vgg16_calib_1.yaml
                -quantization-schema=symmetric_with_power2_scale
                #-instrument-debug
                #-instrument-debug-format=rawbin
                DEPENDS
                model-compiler  VTA ${CMAKE_CURRENT_BINARY_DIR}/vgg16_calib_1.yaml
        )
    endif()
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vgg16_calib_1.yaml
            COMMAND
	    wget -P ${CMAKE_CURRENT_BINARY_DIR} https://www.dropbox.com/s/54zw4p7rxynda6u/vgg16_calib_1.yaml
    )

    add_custom_target(vtaVGG16Cifar10BundleONNX
            DEPENDS
            ${CMAKE_CURRENT_BINARY_DIR}/vgg16_calib_1.yaml
    )
    if(NOT NESTC_USE_PRECOMPILED_BUNDLE_FROM_AWS)
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.h
                ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.weights.bin
                ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                COMMAND
                aws s3 cp s3://nestc-data-pub/VTABundleTests/Vgg16Cifar10Test/vgg16_cifar10.cpp ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.cpp --no-sign-request
                COMMAND
                aws s3 cp s3://nestc-data-pub/VTABundleTests/Vgg16Cifar10Test/VTARuntime.h ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h --no-sign-request
                COMMAND
                aws s3 cp s3://nestc-data-pub/VTABundleTests/Vgg16Cifar10Test/vgg16_cifar10.h ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.h --no-sign-request
                COMMAND
                aws s3 cp s3://nestc-data-pub/VTABundleTests/Vgg16Cifar10Test/vgg16_cifar10.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.weights.bin --no-sign-request
        )
    else()
        if(DEFINED $ENV{S3_BUCKET})
            set(AWS_LOC $ENV{S3_BUCKET})
        else()
            set(AWS_LOC "nestc-pub")
        endif()
        if(${AWS_LOC} STREQUAL "nestc-pub")
            set(AWS_PARAM "--no-sign-request")
        endif()
        add_custom_command(
        OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.h
            ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            aws s3 cp s3://${AWS_LOC}/vta/bundles/Vgg16Cifar10Test/vgg16_cifar10.cpp ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.cpp
            ARGS ${AWS_PARAM}
            COMMAND
            aws s3 cp s3://${AWS_LOC}/vta/bundles/Vgg16Cifar10Test/vgg16_cifar10.h ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.h
            ARGS ${AWS_PARAM}
            COMMAND
            aws s3 cp s3://${AWS_LOC}/vta/bundles/Vgg16Cifar10Test/vgg16_cifar10.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.weights.bin
            ARGS ${AWS_PARAM}
            COMMAND
            aws s3 cp s3://${AWS_LOC}/vta/bundles/Vgg16Cifar10Test/VTARuntime.h ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            ARGS ${AWS_PARAM}
        )
    endif()
endif()
add_custom_target(vtaVGG16Cifar10BundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.cpp ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.weights.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaVGG16Cifar10Bundle vgg16BundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.cpp)
add_dependencies(vtaVGG16Cifar10Bundle vtaVGG16Cifar10BundleNet ${BUNDLE_DEPENDS})
target_link_libraries(vtaVGG16Cifar10Bundle VTABundle png ${BUNDLE_LINK_LIB})
set_target_properties(vtaVGG16Cifar10Bundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaVGG16Cifar10Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaVGG16Cifar10Bundle USE_SH 1 PARAMS ${NESTC_ROOT_DIR}/examples/bundles/vgg16_cifar10/cifar10/test/airplane/0001.png)
