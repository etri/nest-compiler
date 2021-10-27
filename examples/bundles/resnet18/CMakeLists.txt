option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(RESNET18_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/resnet18)
#set(RESNET50_GLOW_S3 "http://fb-glow-assets.s3.amazonaws.com/models/resnet18")
#set(RESNET50_NET_FILES predict_net.pbtxt predict_net.pb init_net.pb)
#set(MODEL_INPUT_NAME "gpu_0/data")
set(MODEL_INPUT_NAME "data")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

# Output directories for regular and quantized outputs
#set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resnet18)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(ResNet18BundleDir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
#  COMMAND ${CMAKE_COMMAND} -E make_directory ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}
)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/resnet18v2.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/resnet18v2.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/resnet18v2.onnx --no-verbose
)

add_custom_target(ResNet18BundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/resnet18v2.onnx
        )
# Final Executables
# =================
# Regular
add_executable(ResNet18Bundle $<TARGET_OBJECTS:ResNet18BundleMain>)
set_target_properties(ResNet18Bundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
#target_link_libraries(ResNet18Bundle ${BUNDLE_OUTPUT_DIRECTORY}/resnet18.o ${BUNDLE_OUTPUT_DIRECTORY}/resnet18v2_onnx_part1.o ${BUNDLE_OUTPUT_DIRECTORY}/resnet18v2_onnx_part2.o png)
if(NOT NEST_BUILD_ON_MAC)
target_link_libraries(ResNet18Bundle ${BUNDLE_OUTPUT_DIRECTORY}/resnet18v2.o png)
else()
target_link_libraries(ResNet18Bundle ${BUNDLE_OUTPUT_DIRECTORY}/resnet18v2.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

#add_dependencies(ResNet18Bundle ResNet18BundleMain ResNet18BundleNet)
add_dependencies(ResNet18Bundle ResNet18BundleMain)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/resnet18v2.o
  COMMAND
    model-compiler -g -model=${CMAKE_CURRENT_BINARY_DIR}/resnet18v2.onnx
    -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
    -backend=CPU
    -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
    -bundle-api=dynamic
  DEPENDS
    model-compiler ResNet18BundleDir ResNet18BundleONNX
)
#add_custom_target(ResNet18BundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/resnet18.o ResNet18BundleNetFiles)
add_custom_target(ResNet18BundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/resnet18v2.o ${BUNDLE_OUTPUT_DIRECTORY}/resnet18v2.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(ResNet18BundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
target_compile_options(ResNet18BundleMain PRIVATE -std=c++11 -g)
else()
target_compile_options(ResNet18BundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(ResNet18BundleMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(ResNet18BundleMain ResNet18BundleNet)

add_nestc_test(NAME ResNet18Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ResNet18Bundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)