add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/resnet50_calib_2.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/resnet50.onnx
        COMMAND
        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet50/resnet50_calib_2.yaml ${CMAKE_CURRENT_BINARY_DIR}/resnet50_calib_2.yaml --no-sign-request
        COMMAND
        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet50/resnet50.onnx ${CMAKE_CURRENT_BINARY_DIR}/resnet50.onnx --no-sign-request
)

add_custom_target(vtaCaffe2Resnet50BundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/resnet50_calib_2.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/resnet50.onnx
        )
if(NESTC_USE_VTASIM)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/resnet50.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/resnet50.h
            ${CMAKE_CURRENT_BINARY_DIR}/resnet50.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            model-compiler -g
            -model=${CMAKE_CURRENT_BINARY_DIR}/resnet50.onnx
            -keep-original-precision-for-nodes=SoftMax
            -backend=VTA
            -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
            -bundle-api=dynamic
            -model-input-name=gpu_0/data_0,float,[1,3,224,224]
            -dump-graph-DAG=opt.dot
            -load-profile=${CMAKE_CURRENT_BINARY_DIR}/resnet50_calib_2.yaml
            -quantization-schema=symmetric_with_power2_scale
            #-instrument-debug
            #-instrument-debug-format=rawbin
            DEPENDS
            model-compiler VTA vtaCaffe2Resnet50BundleONNX
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/resnet50.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/resnet50.h
            ${CMAKE_CURRENT_BINARY_DIR}/resnet50.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet50/resnet50.cpp ${CMAKE_CURRENT_BINARY_DIR}/resnet50.cpp --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet50/resnet50.h ${CMAKE_CURRENT_BINARY_DIR}/resnet50.h --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet50/resnet50.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/resnet50.weights.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet50/VTARuntime.h ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h --no-sign-request
    )
endif()

add_custom_target(vtaCaffe2Resnet50BundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/resnet50.cpp )
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaCaffe2Resnet50Bundle caffe2_resnet50Main.cpp ${CMAKE_CURRENT_BINARY_DIR}/resnet50.cpp)
add_dependencies(vtaCaffe2Resnet50Bundle vtaCaffe2Resnet50BundleNet ${BUNDLE_DEPENDS})
target_link_libraries(vtaCaffe2Resnet50Bundle VTABundle png ${BUNDLE_LINK_LIB})
set_target_properties(vtaCaffe2Resnet50Bundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_RUN_WITH_GENERIC_BUNDLE)
    add_nestc_test(ZCU102 NAME vtaCaffe2Resnet50Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaCaffe2Resnet50Bundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
else()
    add_nestc_test(NAME vtaCaffe2Resnet50Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaCaffe2Resnet50Bundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
endif()
