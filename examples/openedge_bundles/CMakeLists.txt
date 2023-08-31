
add_custom_command(
        OUTPUT
        ${NESTC_BINARY_DIR}/models/resnet50-v2-7.onnx
        COMMAND
        mkdir -p ${NESTC_BINARY_DIR}/models
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/resnet50-v2-7.onnx ${NESTC_BINARY_DIR}/models/resnet50-v2-7.onnx --no-sign-request
)

add_custom_target(openedgeResnet50ONNXDownload
        DEPENDS
        ${NESTC_BINARY_DIR}/models/resnet50-v2-7.onnx
        )

add_custom_command(
        OUTPUT
        ${NESTC_BINARY_DIR}/models/mobilenetv2-7.onnx
        COMMAND
        mkdir -p ${NESTC_BINARY_DIR}/models
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/mobilenetv2-7.onnx ${NESTC_BINARY_DIR}/models/mobilenetv2-7.onnx --no-sign-request
)

add_custom_target(openedgeMobilenetv2ONNXDownload
        DEPENDS
        ${NESTC_BINARY_DIR}/models/mobilenetv2-7.onnx
        )

add_custom_command(
        OUTPUT
        ${NESTC_BINARY_DIR}/models/Golden_mobilenetv2-7.onnx
        COMMAND
        mkdir -p ${NESTC_BINARY_DIR}/models
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/Golden_mobilenetv2-7.onnx ${NESTC_BINARY_DIR}/models/Golden_mobilenetv2-7.onnx --no-sign-request
)

add_custom_target(openedgeGoldenMobilenetONNXDownload
        DEPENDS
        ${NESTC_BINARY_DIR}/models/Golden_mobilenetv2-7.onnx
        )

add_custom_command(
        OUTPUT
        ${NESTC_BINARY_DIR}/models/Golden_resnet50-v2-7.onnx
        COMMAND
        mkdir -p ${NESTC_BINARY_DIR}/models
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/Golden_resnet50-v2-7.onnx ${NESTC_BINARY_DIR}/models/Golden_resnet50-v2-7.onnx --no-sign-request
)

add_custom_target(openedgeGoldenResnet50ONNXDownload
        DEPENDS
        ${NESTC_BINARY_DIR}/models/Golden_resnet50-v2-7.onnx
        )

add_custom_command(
        OUTPUT
        ${NESTC_BINARY_DIR}/models/resnet50_1_1.onnx
        COMMAND
        mkdir -p ${NESTC_BINARY_DIR}/models
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/resnet50_1_1.onnx ${NESTC_BINARY_DIR}/models/resnet50_1_1.onnx --no-sign-request
)

add_custom_target(openedgeresnet50_1_1Download
        DEPENDS
        ${NESTC_BINARY_DIR}/models/resnet50_1_1.onnx
        )

add_custom_command(
        OUTPUT
        ${NESTC_BINARY_DIR}/models/Golden_mobilenetv2-7.onnx
        COMMAND
        mkdir -p ${NESTC_BINARY_DIR}/models
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/Golden_mobilenetv2-7.onnx ${NESTC_BINARY_DIR}/models/Golden_mobilenetv2-7.onnx --no-sign-request
)

add_custom_target(openedgeGoldenMobilenetv2Download
        DEPENDS
        ${NESTC_BINARY_DIR}/models/Golden_mobilenetv2-7.onnx
        )

add_custom_command(
        OUTPUT
        ${NESTC_BINARY_DIR}/models/Golden_resnet50-v2-7.onnx
        COMMAND
        mkdir -p ${NESTC_BINARY_DIR}/models
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/Golden_resnet50-v2-7.onnx ${NESTC_BINARY_DIR}/models/Golden_resnet50-v2-7.onnx --no-sign-request
)

add_custom_target(openedgeGoldenResnet50Download
        DEPENDS
        ${NESTC_BINARY_DIR}/models/Golden_resnet50-v2-7.onnx
        )

set(CMAKE_CXX_STANDARD 17)
FILE(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(object ${subdirs})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${object})
        if(NOT ${object} STREQUAL bnn)
            message(STATUS "Adding ${object} openedge Bundles.")
            add_subdirectory(${object})
        endif()
    endif()
endforeach()
