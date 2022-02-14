option(VTA_RESNET18_WITH_SKIPQUANT0 "Build the LLVM-based JIT CPU backend" OFF)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
        COMMAND
        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/mxnet_exported_resnet18_calib_1.yaml ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml --no-sign-request
)

add_custom_target(singlevtaMxnetResnet18BundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin
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
                model-compiler VTA singlevtaMxnetResnet18BundleONNX MxnetResnet18ONNXDownload
        )
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_1.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_1.h
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_1.weights.bin
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
                -index-multi-evta=2
                -network-name=mxnet_exported_resnet18_1
                -main-entry-name=mxnet_exported_resnet18_1
                #-instrument-debug
                #-instrument-debug-format=rawbin
                DEPENDS
                model-compiler VTA singlevtaMxnetResnet18BundleONNX MxnetResnet18ONNXDownload
        )
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_2.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_2.h
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_2.weights.bin
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
                -index-multi-evta=4
                -network-name=mxnet_exported_resnet18_2
                -main-entry-name=mxnet_exported_resnet18_2
                #-instrument-debug
                #-instrument-debug-format=rawbin
                DEPENDS
                model-compiler VTA singlevtaMxnetResnet18BundleONNX MxnetResnet18ONNXDownload
        )
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_3.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_3.h
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_3.weights.bin
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
                -index-multi-evta=8
                -network-name=mxnet_exported_resnet18_3
                -main-entry-name=mxnet_exported_resnet18_3
                #-instrument-debug
                #-instrument-debug-format=rawbin
                DEPENDS
                model-compiler VTA singlevtaMxnetResnet18BundleONNX MxnetResnet18ONNXDownload
        )
else()
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/MultiEVTAResnet18Test.tar
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/MultiEVTAResnet18Test/MultiEVTAResnet18Test.tar ${CMAKE_CURRENT_BINARY_DIR}/MultiEVTAResnet18Test.tar --no-sign-request
    )
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/VTASkipQuantizeNodes.txt
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_1.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_2.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_3.cpp
            COMMAND
            ${CMAKE_COMMAND} -E tar "xvf" ${CMAKE_CURRENT_BINARY_DIR}/MultiEVTAResnet18Test.tar
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/MultiEVTAResnet18Test.tar
            )

            #[[
            if(NOT VTA_RESNET18_WITH_SKIPQUANT0)
                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_profile.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_debug.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h
                        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin
                        COMMAND
                        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/mxnet_exported_resnet18.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp --no-sign-request
                        COMMAND
                        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/mxnet_exported_resnet18_profile.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_profile.cpp --no-sign-request
                        COMMAND
                        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/mxnet_exported_resnet18_debug.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_debug.cpp --no-sign-request
                        COMMAND
                        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/VTARuntime.h ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h --no-sign-request
                        COMMAND
                        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/mxnet_exported_resnet18.h ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h --no-sign-request
                        COMMAND
                        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/mxnet_exported_resnet18.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin --no-sign-request
                )
            else()
                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_profile.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_debug.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h
                        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin
                        COMMAND
                        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/skip_quant_0/mxnet_exported_resnet18.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp --no-sign-request
                        COMMAND
                        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/skip_quant_0/mxnet_exported_resnet18_profile.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_profile.cpp --no-sign-request
                        COMMAND
                        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/skip_quant_0/mxnet_exported_resnet18_debug.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_debug.cpp --no-sign-request
                        COMMAND
                        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/skip_quant_0/VTARuntime.h ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h --no-sign-request
                        COMMAND
                        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/skip_quant_0/mxnet_exported_resnet18.h ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.h --no-sign-request
                        COMMAND
                        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/skip_quant_0/mxnet_exported_resnet18.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin --no-sign-request
                )
            endif()
            ]]
endif()
add_custom_target(singlevtaMxnetResnet18BundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.weights.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(singlevtaMxnetResnet18Bundle single_thread_main.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp)
add_dependencies(singlevtaMxnetResnet18Bundle singlevtaMxnetResnet18BundleNet ${BUNDLE_DEPENDS})

target_link_libraries(singlevtaMxnetResnet18Bundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(singlevtaMxnetResnet18Bundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 MULTIEVTA NAME singlevtaMxnetResnet18Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/singlevtaMxnetResnet18Bundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)






add_custom_target(multivtaMxnetResnet18BundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_1.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_2.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_3.cpp)
add_executable(multivtaMxnetResnet18Bundle multi_threads_main.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_1.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_2.cpp ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_3.cpp)
add_dependencies(multivtaMxnetResnet18Bundle multivtaMxnetResnet18BundleNet ${BUNDLE_DEPENDS})
target_link_libraries(multivtaMxnetResnet18Bundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(multivtaMxnetResnet18Bundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 MULTIEVTA NAME multivtaMxnetResnet18Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/multivtaMxnetResnet18Bundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
