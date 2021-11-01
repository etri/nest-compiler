option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/bundles)
set(SHUFFLENET_BUNDLE_DIR ${NESTC_BINARY_DIR}/examples/bundles/shufflenet)
set(MODEL_INPUT_NAME "gpu_0/data_0")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

# Output directories for regular and quantized outputs
#set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shufflenet)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(ShufflenetBundleDir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/shufflenet.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/shufflenet.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/shufflenet.onnx --no-verbose
)

add_custom_target(ShufflenetBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/shufflenet.onnx
        )
# Final Executables
# =================
# Regular
add_executable(ShufflenetBundle $<TARGET_OBJECTS:ShufflenetBundleMain>)
set_target_properties(ShufflenetBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
target_link_libraries(ShufflenetBundle ${BUNDLE_OUTPUT_DIRECTORY}/shufflenet.o png)
else()
target_link_libraries(ShufflenetBundle ${BUNDLE_OUTPUT_DIRECTORY}/shufflenet.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(ShufflenetBundle ShufflenetBundleMain)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/shufflenet.o
  COMMAND
    model-compiler -g -model=${CMAKE_CURRENT_BINARY_DIR}/shufflenet.onnx
    -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
    -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
    -bundle-api=dynamic
  DEPENDS
    model-compiler ShufflenetBundleDir ShufflenetBundleONNX
)
add_custom_target(ShufflenetBundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/shufflenet.o ${BUNDLE_OUTPUT_DIRECTORY}/shufflenet.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(ShufflenetBundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
target_compile_options(ShufflenetBundleMain PRIVATE -std=c++11 -g)
else()
target_compile_options(ShufflenetBundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(ShufflenetBundleMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(ShufflenetBundleMain ShufflenetBundleNet)

add_nestc_test(NAME ShufflenetBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ShufflenetBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)