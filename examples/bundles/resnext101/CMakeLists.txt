option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(RESNEXT101_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/ResNext50)
set(MODEL_INPUT_NAME "input")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

#set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resnext101)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(ResNext101BundleDir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/resnext101.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/resnext101.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/resnext101.onnx
)

add_custom_target(ResNext101BundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/resnext101.onnx
        )
# Final Executables
# =================
# Regular
add_executable(ResNext101Bundle $<TARGET_OBJECTS:ResNext101BundleMain>)
set_target_properties(ResNext101Bundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
target_link_libraries(ResNext101Bundle ${BUNDLE_OUTPUT_DIRECTORY}/resnext101.o png)
else()
target_link_libraries(ResNext101Bundle ${BUNDLE_OUTPUT_DIRECTORY}/resnext101.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(ResNext101Bundle ResNext101BundleMain)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/resnext101.o
  COMMAND
    model-compiler -g -model=${CMAKE_CURRENT_BINARY_DIR}/resnext101.onnx
    -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
    -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
    -bundle-api=dynamic
 #   -target=aarch64 -mcpu=cortex-a53
  DEPENDS
    model-compiler ResNext101BundleDir ResNext101BundleONNX
)
add_custom_target(ResNext101BundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/resnext101.o ${BUNDLE_OUTPUT_DIRECTORY}/resnext101.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(ResNext101BundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
target_compile_options(ResNext101BundleMain PRIVATE -std=c++11 -g)
else()
target_compile_options(ResNext101BundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(ResNext101BundleMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(ResNext101BundleMain ResNext101BundleNet)

add_nestc_test(NAME ResNext101Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ResNext101Bundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)