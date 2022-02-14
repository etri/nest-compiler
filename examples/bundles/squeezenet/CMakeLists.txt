option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(RESNET18_NHWC_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/squeezenet)
set(MODEL_INPUT_NAME "data_0")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

# Output directories for regular and quantized outputs
set(NHWC_BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(SqueezeNetBundleDir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${NHWC_BUNDLE_OUTPUT_DIRECTORY}
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/squeezenet.onnx
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/squeezenet.onnx ${CMAKE_CURRENT_BINARY_DIR}/squeezenet.onnx --no-sign-request
)

add_custom_target(SqueezeNetBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/squeezenet.onnx
        )

# Final Executables
# =================
# Regular
add_executable(SqueezeNetBundle $<TARGET_OBJECTS:SqueezeNetBundleMain>)
set_target_properties(SqueezeNetBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NHWC_BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
    target_link_libraries(SqueezeNetBundle ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/squeezenet.o png)
else()
    target_link_libraries(SqueezeNetBundle ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/squeezenet.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(SqueezeNetBundle SqueezeNetBundleMain SqueezeNetBundleNet)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
        OUTPUT
        ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/squeezenet.o
        COMMAND
        model-compiler -model=${CMAKE_CURRENT_BINARY_DIR}/squeezenet.onnx
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        -emit-bundle=${NHWC_BUNDLE_OUTPUT_DIRECTORY}
        -backend=CPU
        -bundle-api=dynamic
   #     -target=aarch64 -mcpu=cortex-a53
        DEPENDS
        model-compiler SqueezeNetBundleDir SqueezeNetBundleONNX
)
add_custom_target(SqueezeNetBundleNet DEPENDS ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/squeezenet.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(SqueezeNetBundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
    target_compile_options(SqueezeNetBundleMain PRIVATE -std=c++11 -g)
else()
    target_compile_options(SqueezeNetBundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(SqueezeNetBundleMain PUBLIC ${NHWC_BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(SqueezeNetBundleMain SqueezeNetBundleNet)
add_nestc_test(NAME SqueezeNetBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/SqueezeNetBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)