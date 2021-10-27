option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(RESNET18_NHWC_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/inception)
set(MODEL_INPUT_NAME "data_0")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

# Output directories for regular and quantized outputs
set(NHWC_BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(InceptionBundleDir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${NHWC_BUNDLE_OUTPUT_DIRECTORY}
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/inception.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/inception_v1.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/inception.onnx --no-verbose
)

add_custom_target(InceptionBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/inception.onnx
        )

# Final Executables
# =================
# Regular
add_executable(InceptionBundle $<TARGET_OBJECTS:InceptionBundleMain>)
set_target_properties(InceptionBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NHWC_BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
    target_link_libraries(InceptionBundle ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/inception.o png)
else()
    target_link_libraries(InceptionBundle ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/inception.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(InceptionBundle InceptionBundleMain InceptionBundleNet)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
        OUTPUT
        ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/inception.o
        COMMAND
        model-compiler -model=${CMAKE_CURRENT_BINARY_DIR}/inception.onnx
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        -emit-bundle=${NHWC_BUNDLE_OUTPUT_DIRECTORY}
        -backend=CPU
        -bundle-api=dynamic
   #     -target=aarch64 -mcpu=cortex-a53
        DEPENDS
        model-compiler InceptionBundleDir InceptionBundleONNX
)
add_custom_target(InceptionBundleNet DEPENDS ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/inception.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(InceptionBundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
    target_compile_options(InceptionBundleMain PRIVATE -std=c++11 -g)
else()
    target_compile_options(InceptionBundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(InceptionBundleMain PUBLIC ${NHWC_BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(InceptionBundleMain InceptionBundleNet)

add_nestc_test(NAME InceptionBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/InceptionBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)