option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(VGG16_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/vgg16)
set(MODEL_INPUT_NAME "input")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/cifar-10-batches-py)

#set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/vgg16)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(VGG16CIFAR10BundleDir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.onnx
        COMMAND
        mkdir -p ${NESTC_BINARY_DIR}/models
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/vgg16_cifar10.onnx ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.onnx --no-sign-request
)


add_custom_target(VGG16CIFAR10BundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.onnx
        )
# Final Executables
# =================
# Regular
add_executable(VGG16CIFAR10Bundle $<TARGET_OBJECTS:VGG16CIFAR10BundleMain>)
set_target_properties(VGG16CIFAR10Bundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
    target_link_libraries(VGG16CIFAR10Bundle ${BUNDLE_OUTPUT_DIRECTORY}/vgg16_cifar10.o png)
else()
    target_link_libraries(VGG16CIFAR10Bundle ${BUNDLE_OUTPUT_DIRECTORY}/vgg16_cifar10.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(VGG16CIFAR10Bundle VGG16CIFAR10BundleMain)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
        OUTPUT
        ${BUNDLE_OUTPUT_DIRECTORY}/vgg16_cifar10.o
        COMMAND
        model-compiler -g -model=${CMAKE_CURRENT_BINARY_DIR}/vgg16_cifar10.onnx
        -model-input=${MODEL_INPUT_NAME},float,[1,3,32,32]
        -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
        -bundle-api=dynamic
#        -target=aarch64 -mcpu=cortex-a53
        DEPENDS
        model-compiler VGG16CIFAR10BundleDir VGG16CIFAR10BundleONNX
)
add_custom_target(VGG16CIFAR10BundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/vgg16_cifar10.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(VGG16CIFAR10BundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
    target_compile_options(VGG16CIFAR10BundleMain PRIVATE -std=c++11 -g)
else()
    target_compile_options(VGG16CIFAR10BundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(VGG16CIFAR10BundleMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(VGG16CIFAR10BundleMain VGG16CIFAR10BundleNet)

add_nestc_test(NAME VGG16CIFAR10Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/VGG16CIFAR10Bundle USE_SH 1 PARAMS ${NESTC_ROOT_DIR}/examples/bundles/vgg16_cifar10/cifar10/test/airplane/0001.png)
