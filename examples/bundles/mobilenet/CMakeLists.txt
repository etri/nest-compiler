option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(MODEL_INPUT_NAME "data")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

# Output directories for regular and quantized outputs
set(MOBILENET_BUNDLE_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/examples/bundles/mobilenet)

add_custom_target(MobileNetBundleDir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}
        )

# Final Executables
# =================
# Regular
#INCLUDE_DIRECTORIES(${GLOW_SOURCE_DIR}/vta/vtalib/include/simulator)
#INCLUDE_DIRECTORIES(${GLOW_SOURCE_DIR}/vta/vtalib/include/Bundle)
#INCLUDE_DIRECTORIES(include ${GLOW_SOURCE_DIR}/vta/vtalib/3rdparty/dlpack/include)
#INCLUDE_DIRECTORIES(include ${GLOW_SOURCE_DIR}/vta/vtalib/3rdparty/dmlc-core/include)

add_executable(MobileNetBundle $<TARGET_OBJECTS:MobileNetBundleMain>)
set_target_properties(MobileNetBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
    target_link_libraries(MobileNetBundle ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p2.o png)
else()
    target_link_libraries(MobileNetBundle ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p2.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(MobileNetBundle MobileNetBundleMain MobileNetBundleNet)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
        OUTPUT
        ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p2.o
        COMMAND
        model-partition-tuner -model=${GLOW_SOURCE_DIR}/tests/models/nestModels/onnxzoo_mobilenet/onnxzoo_mobilenet_v210.onnx
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        -load-device-configs=${GLOW_SOURCE_DIR}/tests/runtime_test/OSTestDeviceConfigs.yaml
        -partition-plan=${GLOW_SOURCE_DIR}/tests/runtime_test/partition_perform_profile/NESTOptimalPlan-Mobilenet-Test.yaml
        -exe-type=0
        -emit-bundle=${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}
        -bundle-api=dynamic
        DEPENDS
        model-partition-tuner MobileNetBundleDir
)

add_custom_command(
        TARGET MobileNetBundle POST_BUILD
        COMMAND
        ls ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/main.cpp)

add_custom_command(
        TARGET MobileNetBundle POST_BUILD
        COMMAND
        ls ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/CMakeLists.txt)

if(GLOW_WITH_VTA_BUNDLE_TEST)
    add_custom_command(
            TARGET MobileNetBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            -E chdir ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY} ./MobileNetBundle ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png
    )
    add_custom_command(
            TARGET MobileNetBundle POST_BUILD
            COMMAND
            echo "MobileNetBundle Test Succeed")
endif()
#add_custom_command(
#        OUTPUT
#        ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/onnxzoo_mobilenet_v210.o
#        COMMAND
#        model-compiler -model=${GLOW_SOURCE_DIR}/tests/models/nestModels/onnxzoo_mobilenet/onnxzoo_mobilenet_v210.onnx
#        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
##        -load-device-configs=${GLOW_SOURCE_DIR}/tests/runtime_test/OSTestDeviceConfigs.yaml
##        -partition-plan=${GLOW_SOURCE_DIR}/tests/runtime_test/partition_perform_profile/CPUTestPartitionPlanMxNet.yaml
##        -exe-type=0
#        -emit-bundle=${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}
#        -bundle-api=dynamic
#        -backend=CPU
#        #        -target=aarch64
#        #        -mcpu=cortex-a53
#        #        -load-vta-profile=${GLOW_SOURCE_DIR}/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18_calib_1.yaml
#        DEPENDS
#        model-compiler MobileNetBundleDir
#)


#============ resnet18v2.onnx ================
#add_custom_command(
#        OUTPUT
#        ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p2.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p3.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p4.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p5.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p6.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p7.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p8.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p9.o
#        ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p10.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p11.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p12.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p13.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p14.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p15.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p16.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p17.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p18.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p19.o
#        ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p20.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p21.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p22.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p23.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p24.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p25.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p26.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p27.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p28.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p29.o
#        ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p30.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p31.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p32.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p33.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p34.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p35.o
#        COMMAND
#        imagenet-partition-profiler ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png
#        -image-mode=0to1
#        -model-input-name=${MODEL_INPUT_NAME}
#        -m=${GLOW_SOURCE_DIR}/tests/models/onnxModels/resnet18v2.onnx
#        -load-device-configs=${GLOW_SOURCE_DIR}/tests/runtime_test/CPUTestDeviceConfigs.yaml
#        -partition-plan=${GLOW_SOURCE_DIR}/tests/runtime_test/partition_perform_profile/CPUTestPartitionPlan.yaml
#        -emit-bundle=${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}
#        -bundle-api=dynamic
#        -time
#        DEPENDS
#        imagenet-partition-profiler MobileNetBundleDir
#)

####============ mxnet_exported_resnet18.onnx ================
#add_custom_command(
#        OUTPUT
#        ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p2.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p3.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p4.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p5.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p6.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p7.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p8.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p9.o
#        ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p10.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p11.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p12.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p13.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p14.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p15.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p16.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p17.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p18.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p19.o
#        ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p20.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p21.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p22.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p23.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p24.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p25.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p26.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p27.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p28.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p29.o
#        ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p30.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p31.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p32.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p33.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p34.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p35.o
#        COMMAND
#        imagenet-partition-profiler ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png
#        -image-layout=NHWC
#        -compute-softmax
#        -image-mode=0to255
#        -model-input-name=${MODEL_INPUT_NAME}
#        -m=${GLOW_SOURCE_DIR}/tests/models/nestModels/mxnet_exported_resnet18/mxnet_exported_resnet18.onnx
#        -load-device-configs=${GLOW_SOURCE_DIR}/tests/runtime_test/CPUTestDeviceConfigs.yaml
#        -partition-plan=${GLOW_SOURCE_DIR}/tests/runtime_test/partition_perform_profile/CPUTestPartitionPlanMxNet.yaml
#        -emit-bundle=${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}
#        -bundle-api=dynamic
#        -time
##        -target=aarch64
##        -mcpu=cortex-a53
#        DEPENDS
#        imagenet-partition-profiler MobileNetBundleDir
#)


add_custom_target(MobileNetBundleNet DEPENDS ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}/p2.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(MobileNetBundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
    target_compile_options(MobileNetBundleMain PRIVATE -std=c++11 -g)
else()
    target_compile_options(MobileNetBundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(MobileNetBundleMain PUBLIC ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(MobileNetBundleMain MobileNetBundleNet)

