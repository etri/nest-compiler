option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(RESNEXT101_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/efficientnet)
set(MODEL_INPUT_NAME "data_0")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

#set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/alexnet)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(AlexNetBundleDir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/alexnet.onnx
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/bvlcalexnet.onnx ${CMAKE_CURRENT_BINARY_DIR}/alexnet.onnx --no-sign-request
)

add_custom_target(AlexNetBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/alexnet.onnx
        )
# Final Executables
# =================
# Regular
add_executable(AlexNetBundle $<TARGET_OBJECTS:AlexNetBundleMain>)
set_target_properties(AlexNetBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
    target_link_libraries(AlexNetBundle ${BUNDLE_OUTPUT_DIRECTORY}/alexnet.o png)
else()
    target_link_libraries(AlexNetBundle ${BUNDLE_OUTPUT_DIRECTORY}/alexnet.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(AlexNetBundle AlexNetBundleMain)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
        OUTPUT
        ${BUNDLE_OUTPUT_DIRECTORY}/alexnet.o
        COMMAND
        model-compiler -g -model=${CMAKE_CURRENT_BINARY_DIR}/alexnet.onnx
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
        -bundle-api=dynamic
#        -target=aarch64 -mcpu=cortex-a53
        DEPENDS
        model-compiler AlexNetBundleDir AlexNetBundleONNX
)
add_custom_target(AlexNetBundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/alexnet.o ${BUNDLE_OUTPUT_DIRECTORY}/alexnet.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(AlexNetBundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
    target_compile_options(AlexNetBundleMain PRIVATE -std=c++11 -g)
else()
    target_compile_options(AlexNetBundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(AlexNetBundleMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(AlexNetBundleMain AlexNetBundleNet)

add_nestc_test(NAME AlexNetBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/AlexNetBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)