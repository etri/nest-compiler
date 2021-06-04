option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(RESNET18_NHWC_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/resnet18_nhwc)
set(MODEL_INPUT_NAME "data")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

# Output directories for regular and quantized outputs
set(NHWC_BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(ResNet18NHWCBundleDir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${NHWC_BUNDLE_OUTPUT_DIRECTORY}
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
)

add_custom_target(ResNet18NHWCBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
        )

# Final Executables
# =================
# Regular
add_executable(ResNet18NHWCBundle $<TARGET_OBJECTS:ResNet18NHWCBundleMain>)
set_target_properties(ResNet18NHWCBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NHWC_BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
    target_link_libraries(ResNet18NHWCBundle ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/mxnet_exported_resnet18.o png)
else()
    target_link_libraries(ResNet18NHWCBundle ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/mxnet_exported_resnet18.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(ResNet18NHWCBundle ResNet18NHWCBundleMain ResNet18NHWCBundleNet)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
        OUTPUT
        ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/mxnet_exported_resnet18.o
        COMMAND
        model-compiler -model=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
        -model-input=${MODEL_INPUT_NAME},float,[1,224,224,3]
        -emit-bundle=${NHWC_BUNDLE_OUTPUT_DIRECTORY}
        -backend=CPU
        -bundle-api=dynamic
        DEPENDS
        model-compiler ResNet18NHWCBundleDir ResNet18NHWCBundleONNX
)
add_custom_target(ResNet18NHWCBundleNet DEPENDS ${NHWC_BUNDLE_OUTPUT_DIRECTORY}/mxnet_exported_resnet18.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(ResNet18NHWCBundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
    target_compile_options(ResNet18NHWCBundleMain PRIVATE -std=c++11 -g)
else()
    target_compile_options(ResNet18NHWCBundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(ResNet18NHWCBundleMain PUBLIC ${NHWC_BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(ResNet18NHWCBundleMain ResNet18NHWCBundleNet)

add_nestc_test(NAME ResNet18NHWCBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ResNet18NHWCBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)