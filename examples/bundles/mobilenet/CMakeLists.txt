option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(MODEL_INPUT_NAME "data")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

# Output directories for regular and quantized outputs
#set(MOBILENET_BUNDLE_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/examples/bundles/mobilenet)
set(MOBILENET_BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})


add_custom_target(MobileNetBundleDir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/onnxzoo_mobilenet_v210.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/onnxzoo_mobilenet_v210.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/onnxzoo_mobilenet_v210.onnx
)

add_custom_target(MobileNetBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/onnxzoo_mobilenet_v210.onnx
        )
# Final Executables
# =================

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
        model-partition-tuner -model=${CMAKE_CURRENT_BINARY_DIR}/onnxzoo_mobilenet_v210.onnx
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        -load-device-configs=${NESTC_ROOT_DIR}/tests/runtime_test/OSTestDeviceConfigs.yaml
        -partition-plan=${NESTC_ROOT_DIR}/tests/runtime_test/partition_perform_profile/NESTOptimalPlan-Mobilenet-Test.yaml
        -exe-type=0
        -emit-bundle=${MOBILENET_BUNDLE_OUTPUT_DIRECTORY}
        -bundle-api=dynamic
        DEPENDS
        model-partition-tuner MobileNetBundleDir MobileNetBundleONNX
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

add_nestc_test(NAME MobileNetBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/MobileNetBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)