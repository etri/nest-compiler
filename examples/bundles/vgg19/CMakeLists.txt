option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(RESNEXT101_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/efficientnet)
set(MODEL_INPUT_NAME "data_0")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

#set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/vgg19)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(VGG19BundleDir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/vgg19.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/vgg19-caffe2-9.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/vgg19.onnx --no-verbose
)

add_custom_target(VGG19BundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/vgg19.onnx
        )
# Final Executables
# =================
# Regular
add_executable(VGG19Bundle $<TARGET_OBJECTS:VGG19BundleMain>)
set_target_properties(VGG19Bundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
    target_link_libraries(VGG19Bundle ${BUNDLE_OUTPUT_DIRECTORY}/vgg19.o png)
else()
    target_link_libraries(VGG19Bundle ${BUNDLE_OUTPUT_DIRECTORY}/vgg19.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(VGG19Bundle VGG19BundleMain)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
        OUTPUT
        ${BUNDLE_OUTPUT_DIRECTORY}/vgg19.o
        COMMAND
        model-compiler -g -model=${CMAKE_CURRENT_BINARY_DIR}/vgg19.onnx
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
        -bundle-api=dynamic
#        -target=aarch64 -mcpu=cortex-a53
        DEPENDS
        model-compiler VGG19BundleDir VGG19BundleONNX
)
add_custom_target(VGG19BundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/vgg19.o ${BUNDLE_OUTPUT_DIRECTORY}/vgg19.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(VGG19BundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
    target_compile_options(VGG19BundleMain PRIVATE -std=c++11 -g)
else()
    target_compile_options(VGG19BundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(VGG19BundleMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(VGG19BundleMain VGG19BundleNet)

add_nestc_test(NAME VGG19Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/VGG19Bundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)