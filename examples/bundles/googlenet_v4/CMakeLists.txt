option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(RESNET18_NHWC_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/googlenet)
set(MODEL_INPUT_NAME "data_0")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

# Output directories for regular and quantized outputs
set(NHWC_BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(GoogleNetBundleDir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${NHWC_BUNDLE_OUTPUT_DIRECTORY}
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/googlenet.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/googlenet_v4.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/googlenet.onnx
)

add_custom_target(GoogleNetBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/googlenet.onnx
        )

# Final Executables
# =================
# Regular
add_executable(GoogleNetBundle $<TARGET_OBJECTS:GoogleNetBundleMain>)
set_target_properties(GoogleNetBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NHWC_BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
    target_link_libraries(GoogleNetBundle ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/googlenet.o png)
else()
    target_link_libraries(GoogleNetBundle ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/googlenet.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(GoogleNetBundle GoogleNetBundleMain GoogleNetBundleNet)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
        OUTPUT
        ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/googlenet.o
        COMMAND
        model-compiler -model=${CMAKE_CURRENT_BINARY_DIR}/googlenet.onnx
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        -emit-bundle=${NHWC_BUNDLE_OUTPUT_DIRECTORY}
        -backend=CPU
        -bundle-api=dynamic
    #    -target=aarch64 -mcpu=cortex-a53
        DEPENDS
        model-compiler GoogleNetBundleDir GoogleNetBundleONNX
)
add_custom_target(GoogleNetBundleNet DEPENDS ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/googlenet.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(GoogleNetBundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
    target_compile_options(GoogleNetBundleMain PRIVATE -std=c++11 -g)
else()
    target_compile_options(GoogleNetBundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(GoogleNetBundleMain PUBLIC ${NHWC_BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(GoogleNetBundleMain GoogleNetBundleNet)

add_nestc_test(NAME GoogleNetBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GoogleNetBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)