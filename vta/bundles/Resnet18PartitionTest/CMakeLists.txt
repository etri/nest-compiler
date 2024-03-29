option(NESTC_TARGET_AARCH64 "Generate bundles for publish" OFF)
option(VTA_RESNET18_WITH_SKIPQUANT0 "Build the LLVM-based JIT CPU backend" OFF)
option(NESTC_USE_PRECOMPILED_BUNDLE_FROM_AWS "Download bundles from AWS S3" OFF)

set(MODEL_INPUT_NAME "data")
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
        COMMAND
        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/mxnet_exported_resnet18_calib_1.yaml ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml --no-sign-request
)

add_custom_target(vtaMxnetResnet18PartitionBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
)
if(NESTC_TARGET_AARCH64)
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o
                ${CMAKE_CURRENT_BINARY_DIR}/p0.h ${CMAKE_CURRENT_BINARY_DIR}/p1.h ${CMAKE_CURRENT_BINARY_DIR}/p2.h ${CMAKE_CURRENT_BINARY_DIR}/p3.h ${CMAKE_CURRENT_BINARY_DIR}/p4.h ${CMAKE_CURRENT_BINARY_DIR}/p5.h ${CMAKE_CURRENT_BINARY_DIR}/p6.h ${CMAKE_CURRENT_BINARY_DIR}/p7.h ${CMAKE_CURRENT_BINARY_DIR}/p8.h ${CMAKE_CURRENT_BINARY_DIR}/p9.h ${CMAKE_CURRENT_BINARY_DIR}/p10.h ${CMAKE_CURRENT_BINARY_DIR}/p11.h ${CMAKE_CURRENT_BINARY_DIR}/p12.h ${CMAKE_CURRENT_BINARY_DIR}/p13.h ${CMAKE_CURRENT_BINARY_DIR}/p14.h ${CMAKE_CURRENT_BINARY_DIR}/p15.h ${CMAKE_CURRENT_BINARY_DIR}/p16.h ${CMAKE_CURRENT_BINARY_DIR}/p17.h
                ${CMAKE_CURRENT_BINARY_DIR}/p0.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p1.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p2.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p3.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p4.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p5.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p6.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p7.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p8.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p9.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p10.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p11.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p12.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p13.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p14.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p15.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p16.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p17.weights.bin
                ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                COMMAND
                model-partition-tuner -model=${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
                -model-input=${MODEL_INPUT_NAME},float,[1,224,224,3]
                -target=aarch64 -mcpu=cortex-a53
                -load-device-configs=${NESTC_ROOT_DIR}/tests/runtime_test/VTATestDeviceConfigs.yaml
                -partition-plan=${NESTC_ROOT_DIR}/tests/runtime_test/partition_perform_profile/NESTOptimalFusedPlan.yaml
                -exe-type=0
                -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
                -bundle-api=dynamic
                -load-profile=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
                DEPENDS
                model-partition-tuner VTA vtaMxnetResnet18PartitionBundleONNX MxnetResnet18ONNXDownload)
        add_custom_target(vtaMxnetResnet18PartitionBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o)
else()
        if(NESTC_USE_VTASIM)
                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p0.h ${CMAKE_CURRENT_BINARY_DIR}/p1.h ${CMAKE_CURRENT_BINARY_DIR}/p2.h ${CMAKE_CURRENT_BINARY_DIR}/p3.h ${CMAKE_CURRENT_BINARY_DIR}/p4.h ${CMAKE_CURRENT_BINARY_DIR}/p5.h ${CMAKE_CURRENT_BINARY_DIR}/p6.h ${CMAKE_CURRENT_BINARY_DIR}/p7.h ${CMAKE_CURRENT_BINARY_DIR}/p8.h ${CMAKE_CURRENT_BINARY_DIR}/p9.h ${CMAKE_CURRENT_BINARY_DIR}/p10.h ${CMAKE_CURRENT_BINARY_DIR}/p11.h ${CMAKE_CURRENT_BINARY_DIR}/p12.h ${CMAKE_CURRENT_BINARY_DIR}/p13.h ${CMAKE_CURRENT_BINARY_DIR}/p14.h ${CMAKE_CURRENT_BINARY_DIR}/p15.h ${CMAKE_CURRENT_BINARY_DIR}/p16.h ${CMAKE_CURRENT_BINARY_DIR}/p17.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p0.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p1.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p2.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p3.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p4.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p5.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p6.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p7.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p8.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p9.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p10.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p11.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p12.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p13.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p14.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p15.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p16.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p17.weights.bin
                        ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                        COMMAND
                        model-partition-tuner -model=${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
                        -model-input=${MODEL_INPUT_NAME},float,[1,224,224,3]
                        -load-device-configs=${NESTC_ROOT_DIR}/tests/runtime_test/VTATestDeviceConfigs.yaml
                        -partition-plan=${NESTC_ROOT_DIR}/tests/runtime_test/partition_perform_profile/NESTOptimalFusedPlan.yaml
                        -exe-type=0
                        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
                        -bundle-api=dynamic
                        -load-profile=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
                        DEPENDS
                        model-partition-tuner VTA vtaMxnetResnet18PartitionBundleONNX MxnetResnet18ONNXDownload)
                add_custom_target(vtaMxnetResnet18PartitionBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o)
                INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
                add_executable(vtaMxnetResnet18PartitionBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o)
                add_dependencies(vtaMxnetResnet18PartitionBundle vtaMxnetResnet18PartitionBundleNet ${BUNDLE_DEPENDS})

                target_link_libraries(vtaMxnetResnet18PartitionBundle VTABundle ${BUNDLE_LINK_LIB})
                set_target_properties(vtaMxnetResnet18PartitionBundle
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )

                add_nestc_test(NAME vtaMxnetResnet18PartitionBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18PartitionBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
        endif()
endif()

if(NOT NESTC_USE_VTASIM)
        if(NESTC_USE_PRECOMPILED_BUNDLE_FROM_AWS)
                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o
                        COMMAND
                        aws s3 cp s3://$ENV{S3_BUCKET}/vta/bundles/Resnet18PartitionTest/ ${CMAKE_CURRENT_BINARY_DIR}/ --recursive
                )
                add_executable(vtaMxnetResnet18PartitionBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o)
                add_dependencies(vtaMxnetResnet18PartitionBundle ${BUNDLE_DEPENDS})

                target_link_libraries(vtaMxnetResnet18PartitionBundle VTABundle ${BUNDLE_LINK_LIB})
                set_target_properties(vtaMxnetResnet18PartitionBundle
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )
                INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
                add_nestc_test(ZCU102 NAME vtaMxnetResnet18PartitionBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18PartitionBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
        endif()
endif()