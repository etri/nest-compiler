option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(RESNEXT101_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/efficientnet)
set(MODEL_INPUT_NAME "images:0")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

#set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/efficientnet_lite4)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(EfficientNetBundleDir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/efficientnet_lite4.onnx
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/efficientnet_lite4.onnx ${CMAKE_CURRENT_BINARY_DIR}/efficientnet_lite4.onnx --no-sign-request
)

add_custom_target(EfficientNetBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/efficientnet_lite4.onnx
        )
# Final Executables
# =================
# Regular
add_executable(EfficientNetBundle $<TARGET_OBJECTS:EfficientNetBundleMain>)
set_target_properties(EfficientNetBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
    target_link_libraries(EfficientNetBundle ${BUNDLE_OUTPUT_DIRECTORY}/efficientnet_lite4.o png)
else()
    target_link_libraries(EfficientNetBundle ${BUNDLE_OUTPUT_DIRECTORY}/efficientnet_lite4.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(EfficientNetBundle EfficientNetBundleMain)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
        OUTPUT
        ${BUNDLE_OUTPUT_DIRECTORY}/efficientnet_lite4.o
        COMMAND
        model-compiler -g -model=${CMAKE_CURRENT_BINARY_DIR}/efficientnet_lite4.onnx
        -model-input=${MODEL_INPUT_NAME},float,[1,224,224,3]
        -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
        -bundle-api=dynamic
#        -target=aarch64 -mcpu=cortex-a53
        DEPENDS
        model-compiler EfficientNetBundleDir EfficientNetBundleONNX
)
add_custom_target(EfficientNetBundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/efficientnet_lite4.o ${BUNDLE_OUTPUT_DIRECTORY}/efficientnet_lite4.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(EfficientNetBundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
    target_compile_options(EfficientNetBundleMain PRIVATE -std=c++11 -g)
else()
    target_compile_options(EfficientNetBundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(EfficientNetBundleMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(EfficientNetBundleMain EfficientNetBundleNet)

add_nestc_test(NAME EfficientNetBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/EfficientNetBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)