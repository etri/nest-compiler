option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(RESNEXT101_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/efficientnet)
set(MODEL_INPUT_NAME "data_0")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

#set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/densenet)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(DenseNetBundleDir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/densenet.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/densenet-9.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/densenet.onnx
)

add_custom_target(DenseNetBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/densenet.onnx
        )
# Final Executables
# =================
# Regular
add_executable(DenseNetBundle $<TARGET_OBJECTS:DenseNetBundleMain>)
set_target_properties(DenseNetBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
    target_link_libraries(DenseNetBundle ${BUNDLE_OUTPUT_DIRECTORY}/densenet.o png)
else()
    target_link_libraries(DenseNetBundle ${BUNDLE_OUTPUT_DIRECTORY}/densenet.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(DenseNetBundle DenseNetBundleMain)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
        OUTPUT
        ${BUNDLE_OUTPUT_DIRECTORY}/densenet.o
        COMMAND
        model-compiler -g -model=${CMAKE_CURRENT_BINARY_DIR}/densenet.onnx
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
        -bundle-api=dynamic
#        -target=aarch64 -mcpu=cortex-a53
        DEPENDS
        model-compiler DenseNetBundleDir DenseNetBundleONNX
)
add_custom_target(DenseNetBundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/densenet.o ${BUNDLE_OUTPUT_DIRECTORY}/densenet.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(DenseNetBundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
    target_compile_options(DenseNetBundleMain PRIVATE -std=c++11 -g)
else()
    target_compile_options(DenseNetBundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(DenseNetBundleMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(DenseNetBundleMain DenseNetBundleNet)

add_nestc_test(NAME DenseNetBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/DenseNetBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)