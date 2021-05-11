option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(PARTITION_BUNDLE_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/examples/bundles/resnet18_partition)
set(MODEL_INPUT_NAME "data")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

# Output directories for regular and quantized outputs
#set(PARTITION_BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resnet18_partition)

add_custom_target(ResNet18PartitionBundleDir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
)

add_custom_target(ResNet18PartitionBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
        )

# Final Executables
# =================
# Regular
add_executable(ResNet18PartitionBundle $<TARGET_OBJECTS:ResNet18PartitionBundleMain>)
set_target_properties(ResNet18PartitionBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PARTITION_BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
    target_link_libraries(ResNet18PartitionBundle ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p2.o png)
else()
    target_link_libraries(ResNet18PartitionBundle ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p2.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(ResNet18PartitionBundle ResNet18PartitionBundleMain ResNet18PartitionBundleNet)

# Glow Bundles
# ============
# Regular Bundle
#add_custom_command(
#        OUTPUT
#        ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p2.o
#        COMMAND
#        model-partition-tuner -model=${GLOW_SOURCE_DIR}/tests/models/onnxModels/resnet18v2.onnx
#        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
#        -load-device-configs=${GLOW_SOURCE_DIR}/tests/runtime_test/OSTestDeviceConfigs.yaml
#        -partition-plan=${GLOW_SOURCE_DIR}/tests/runtime_test/partition_perform_profile/OSTestPartitionPlan.yaml
#        -exe-type=0
#        -emit-bundle=${PARTITION_BUNDLE_OUTPUT_DIRECTORY}
#        -bundle-api=dynamic
#        DEPENDS
#        model-partition-tuner ResNet18PartitionBundleDir
#)
add_custom_command(
        OUTPUT
        ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p2.o
        COMMAND
        model-partition-tuner -model=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
        -model-input=${MODEL_INPUT_NAME},float,[1,224,224,3]
        -load-device-configs=${NESTC_ROOT_DIR}/tests/runtime_test/OSTestDeviceConfigs.yaml
        -partition-plan=${NESTC_ROOT_DIR}/tests/runtime_test/partition_perform_profile/CPU4TestPartitionPlanMxNet.yaml
        -exe-type=0
        -emit-bundle=${PARTITION_BUNDLE_OUTPUT_DIRECTORY}
        -bundle-api=dynamic
#        -target=aarch64
#        -mcpu=cortex-a53
#        -load-vta-profile=${GLOW_SOURCE_DIR}/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18_calib_1.yaml
        DEPENDS
        model-partition-tuner ResNet18PartitionBundleDir ResNet18PartitionBundleONNX
)

add_custom_command(
        TARGET ResNet18PartitionBundle POST_BUILD
        COMMAND
        ls ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/main.cpp)

add_custom_command(
        TARGET ResNet18PartitionBundle POST_BUILD
        COMMAND
        ls ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/CMakeLists.txt)

if(GLOW_WITH_VTA_BUNDLE_TEST)
    add_custom_command(
            TARGET ResNet18PartitionBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            -E chdir ${PARTITION_BUNDLE_OUTPUT_DIRECTORY} ./ResNet18PartitionBundle ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png
    )
    add_custom_command(
            TARGET ResNet18PartitionBundle POST_BUILD
            COMMAND
            echo "ResNet18PartitionBundle Test Succeed")
endif()
#============ resnet18v2.onnx ================
#add_custom_command(
#        OUTPUT
#        ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p2.o
#        COMMAND
#        imagenet-partition-profiler ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png
#        -image-mode=0to1
#        -model-input-name=${MODEL_INPUT_NAME}
#        -m=${GLOW_SOURCE_DIR}/tests/models/onnxModels/resnet18v2.onnx
#        -load-device-configs=${GLOW_SOURCE_DIR}/tests/runtime_test/OSTestDeviceConfigs.yaml
#        -partition-plan=${GLOW_SOURCE_DIR}/tests/runtime_test/partition_perform_profile/OSTestPartitionPlan.yaml
#        -emit-bundle=${PARTITION_BUNDLE_OUTPUT_DIRECTORY}
#        -bundle-api=dynamic
#        -time
#        DEPENDS
#        imagenet-partition-profiler ResNet18PartitionBundleDir
#)

##============ mxnet_exported_resnet18.onnx ================
#add_custom_command(
#        OUTPUT
#        ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p2.o
#        COMMAND
#        imagenet-partition-profiler ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png
#        -image-layout=NHWC
#        -compute-softmax
#        -image-mode=0to255
#        -model-input-name=${MODEL_INPUT_NAME}
#        -m=${GLOW_SOURCE_DIR}/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
#        -load-device-configs=${GLOW_SOURCE_DIR}/tests/runtime_test/OSTestDeviceConfigs.yaml
#        -partition-plan=${GLOW_SOURCE_DIR}/tests/runtime_test/partition_perform_profile/OSTestPartitionPlanMxNet.yaml
#        -emit-bundle=${PARTITION_BUNDLE_OUTPUT_DIRECTORY}
#        -bundle-api=dynamic
#        -time
#        DEPENDS
#        imagenet-partition-profiler ResNet18PartitionBundleDir
#)
#

add_custom_target(ResNet18PartitionBundleNet DEPENDS ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${PARTITION_BUNDLE_OUTPUT_DIRECTORY}/p2.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(ResNet18PartitionBundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
    target_compile_options(ResNet18PartitionBundleMain PRIVATE -std=c++11 -g)
else()
    target_compile_options(ResNet18PartitionBundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(ResNet18PartitionBundleMain PUBLIC ${PARTITION_BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(ResNet18PartitionBundleMain ResNet18PartitionBundleNet)

