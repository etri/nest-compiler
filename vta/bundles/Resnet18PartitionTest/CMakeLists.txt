option(VTA_RESNET18_WITH_SKIPQUANT0 "Build the LLVM-based JIT CPU backend" OFF)
set(MODEL_INPUT_NAME "data")
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18_calib_1.yaml -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
)

add_custom_target(vtaMxnetResnet18PartitionBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
)

if(NESTC_EVTA_BUNDLE_TEST)
if(NESTC_USE_VTASIM)
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o
                ${CMAKE_CURRENT_BINARY_DIR}/p0.h ${CMAKE_CURRENT_BINARY_DIR}/p1.h ${CMAKE_CURRENT_BINARY_DIR}/p2.h ${CMAKE_CURRENT_BINARY_DIR}/p3.h ${CMAKE_CURRENT_BINARY_DIR}/p4.h ${CMAKE_CURRENT_BINARY_DIR}/p5.h ${CMAKE_CURRENT_BINARY_DIR}/p6.h ${CMAKE_CURRENT_BINARY_DIR}/p7.h ${CMAKE_CURRENT_BINARY_DIR}/p8.h ${CMAKE_CURRENT_BINARY_DIR}/p9.h ${CMAKE_CURRENT_BINARY_DIR}/p10.h ${CMAKE_CURRENT_BINARY_DIR}/p11.h ${CMAKE_CURRENT_BINARY_DIR}/p12.h ${CMAKE_CURRENT_BINARY_DIR}/p13.h ${CMAKE_CURRENT_BINARY_DIR}/p14.h ${CMAKE_CURRENT_BINARY_DIR}/p15.h ${CMAKE_CURRENT_BINARY_DIR}/p16.h ${CMAKE_CURRENT_BINARY_DIR}/p17.h
                ${CMAKE_CURRENT_BINARY_DIR}/p0.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p1.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p2.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p3.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p4.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p5.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p6.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p7.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p8.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p9.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p10.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p11.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p12.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p13.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p14.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p15.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p16.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p17.weights.bin
                ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                COMMAND
                model-partition-tuner -model=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
                -model-input=${MODEL_INPUT_NAME},float,[1,224,224,3]
                -load-device-configs=${NESTC_ROOT_DIR}/tests/runtime_test/VTATestDeviceConfigs.yaml
                -partition-plan=${NESTC_ROOT_DIR}/tests/runtime_test/partition_perform_profile/NESTOptimalFusedPlan.yaml
                -exe-type=0
                -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
                -bundle-api=dynamic
                -load-vta-profile=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
                DEPENDS
                model-partition-tuner VTA vtaMxnetResnet18PartitionBundleONNX)
add_custom_target(vtaMxnetResnet18PartitionBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaMxnetResnet18PartitionBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o)
add_dependencies(vtaMxnetResnet18PartitionBundle vtaMxnetResnet18PartitionBundleNet ${BUNDLE_DEPENDS})

target_link_libraries(vtaMxnetResnet18PartitionBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaMxnetResnet18PartitionBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

    add_custom_command(
            TARGET vtaMxnetResnet18PartitionBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png
            ${CMAKE_CURRENT_BINARY_DIR}/cat_285.png)

    add_custom_command(
            TARGET vtaMxnetResnet18PartitionBundle POST_BUILD
            COMMAND
            mkdir -p debug)
    add_custom_command(
            TARGET vtaMxnetResnet18PartitionBundle POST_BUILD
            COMMAND
            vtaMxnetResnet18PartitionBundle cat_285.png)
    add_custom_command(
            TARGET vtaMxnetResnet18PartitionBundle POST_BUILD
            COMMAND
            echo "vtaMxnetResnet18PartitionBundle Test Succeed")
endif()
endif()
