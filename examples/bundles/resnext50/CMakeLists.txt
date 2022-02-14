option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(RESNEXT50_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/ResNext50)
set(MODEL_INPUT_NAME "input")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

#set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resnext50)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(ResNext50BundleDir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/resnext50.onnx
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/resnext50.onnx ${CMAKE_CURRENT_BINARY_DIR}/resnext50.onnx --no-sign-request
)

add_custom_target(ResNext50BundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/resnext50.onnx
        )
# Final Executables
# =================
# Regular
add_executable(ResNext50Bundle $<TARGET_OBJECTS:ResNext50BundleMain>)
set_target_properties(ResNext50Bundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
target_link_libraries(ResNext50Bundle ${BUNDLE_OUTPUT_DIRECTORY}/resnext50.o png)
else()
target_link_libraries(ResNext50Bundle ${BUNDLE_OUTPUT_DIRECTORY}/resnext50.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(ResNext50Bundle ResNext50BundleMain)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/resnext50.o
  COMMAND
    model-compiler -g -model=${CMAKE_CURRENT_BINARY_DIR}/resnext50.onnx
    -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
 #   -target=aarch64 -mcpu=cortex-a53
    -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
    -bundle-api=dynamic
  DEPENDS
    model-compiler ResNext50BundleDir ResNext50BundleONNX
)
add_custom_target(ResNext50BundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/resnext50.o ${BUNDLE_OUTPUT_DIRECTORY}/resnext50.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(ResNext50BundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
target_compile_options(ResNext50BundleMain PRIVATE -std=c++11 -g)
else()
target_compile_options(ResNext50BundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(ResNext50BundleMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(ResNext50BundleMain ResNext50BundleNet)

add_nestc_test(NAME ResNext50Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ResNext50Bundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)