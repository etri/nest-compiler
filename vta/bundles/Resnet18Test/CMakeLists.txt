option(VTA_RESNET18_WITH_SKIPQUANT0 "Build the LLVM-based JIT CPU backend" OFF)
option(NESTC_USE_PRECOMPILED_BUNDLE_FROM_AWS "Download bundles from AWS S3" OFF)


if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
            COMMAND
            image-classifier ${NESTC_ROOT_DIR}/glow/tests/images/imagenet/cat_285.png
            -m=${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
            -model-input-name=data
            -backend=VTAInterpreter
            -dump-graph-DAG="opt.dot"
            -image-layout=NHWC
            -compute-softmax
            -topk=5
            -image-mode=0to255
            -dump-profile=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
            -quantization-schema="symmetric_with_power2_scale"
            DEPENDS image-classifier
    )

    if(NOT VTA_RESNET18_WITH_SKIPQUANT0)
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin
                ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/VTAConvolutionTune.txt
                ${CMAKE_CURRENT_BINARY_DIR}/VTAConvolutionTune.txt
                COMMAND
                model-compiler -g
                -model=${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
                -backend=VTA
                -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
                -bundle-api=dynamic
                -model-input=data,float,[1,224,224,3]
                -dump-graph-DAG=opt.dot
                -load-profile=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
                -quantization-schema=symmetric_with_power2_scale
                #-instrument-debug
                #-instrument-debug-format=rawbin
                DEPENDS
                model-compiler VTA MxnetResnet18ONNXDownload ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml

        )
    else()
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin
                ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                ${CMAKE_CURRENT_BINARY_DIR}/VTASkipQuantizeNodes.txt
                COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/VTAConvolutionTune.txt
                ${CMAKE_CURRENT_BINARY_DIR}/VTAConvolutionTune.txt
                COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/VTASkipQuantizeNodes.txt
                ${CMAKE_CURRENT_BINARY_DIR}/VTASkipQuantizeNodes.txt
                COMMAND
                model-compiler -g
                -model=${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
                -backend=VTA
                -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
                -bundle-api=dynamic
                -model-input=data,float,[1,224,224,3]
                -dump-graph-DAG=opt.dot
                -load-profile=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
                -quantization-schema=symmetric_with_power2_scale
                #-instrument-debug
                #-instrument-debug-format=rawbin
                DEPENDS
                model-compiler VTA MxnetResnet18ONNXDownload ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
        )
    endif()
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18_calib_1.yaml -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
    )

    add_custom_target(vtaMxnetResnet18BundleONNX
            DEPENDS
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
            )
    if(NOT NESTC_USE_PRECOMPILED_BUNDLE_FROM_AWS)
        if(NOT VTA_RESNET18_WITH_SKIPQUANT0)
            add_custom_command(
                    OUTPUT
                    ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_profile.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_debug.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h
                    ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin
                    ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                    COMMAND
                    wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp
                    COMMAND
                    wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18_profile.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_profile.cpp
                    COMMAND
                    wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18_debug.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_debug.cpp
                    COMMAND
                    wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/VTARuntime.h -O ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                    COMMAND
                    wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18.h -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h
                    COMMAND
                    wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin
            )
        else()
            add_custom_command(
                    OUTPUT
                    ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_profile.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_debug.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h
                    ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin
                    ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                    COMMAND
                    wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/skip_quant_0/mxnet_exported_resnet18.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp
                    COMMAND
                    wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/skip_quant_0/mxnet_exported_resnet18_profile.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_profile.cpp
                    COMMAND
                    wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/skip_quant_0/mxnet_exported_resnet18_debug.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_debug.cpp
                    COMMAND
                    wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/skip_quant_0/VTARuntime.h -O ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                    COMMAND
                    wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/skip_quant_0/mxnet_exported_resnet18.h -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h
                    COMMAND
                    wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/skip_quant_0/mxnet_exported_resnet18.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin
            )
        endif()
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
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            aws s3 cp s3://${AWS_LOC}/vta/bundles/Resnet18Test/mxnet_exported_resnet18.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp
            ARGS ${AWS_PARAM}
            COMMAND
            aws s3 cp s3://${AWS_LOC}/vta/bundles/Resnet18Test/mxnet_exported_resnet18.h ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h
            ARGS ${AWS_PARAM}
            COMMAND
            aws s3 cp s3://${AWS_LOC}/vta/bundles/Resnet18Test/mxnet_exported_resnet18.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin
            ARGS ${AWS_PARAM}
            COMMAND
            aws s3 cp s3://${AWS_LOC}/vta/bundles/Resnet18Test/VTARuntime.h ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            ARGS ${AWS_PARAM}
        )
    endif()


endif()
add_custom_target(vtaMxnetResnet18BundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaMxnetResnet18Bundle mxnet_exported_resnet18BundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp)
add_dependencies(vtaMxnetResnet18Bundle vtaMxnetResnet18BundleNet ${BUNDLE_DEPENDS})

target_link_libraries(vtaMxnetResnet18Bundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaMxnetResnet18Bundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaMxnetResnet18Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18Bundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
