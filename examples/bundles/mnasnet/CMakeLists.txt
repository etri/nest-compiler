option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(RESNEXT50_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/MNasNet)
set(MODEL_INPUT_NAME "input")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

#set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mnasnet)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(MNasNetDir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mnasnet.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/mnasnet.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/mnasnet.onnx
)

add_custom_target(MNasNetONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/mnasnet.onnx
        )
# Final Executables
# =================
# Regular
add_executable(MNasNet $<TARGET_OBJECTS:MNasNetMain>)
set_target_properties(MNasNet PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
target_link_libraries(MNasNet ${BUNDLE_OUTPUT_DIRECTORY}/mnasnet.o png)
else()
target_link_libraries(MNasNet ${BUNDLE_OUTPUT_DIRECTORY}/mnasnet.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(MNasNet MNasNetMain)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/mnasnet.o
  COMMAND
    model-compiler -g -model=${CMAKE_CURRENT_BINARY_DIR}/mnasnet.onnx
    -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
  #  -target=aarch64 -mcpu=cortex-a53
    -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
    -bundle-api=dynamic
  DEPENDS
    model-compiler MNasNetDir MNasNetONNX
)
add_custom_target(MNasNetNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/mnasnet.o ${BUNDLE_OUTPUT_DIRECTORY}/mnasnet.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(MNasNetMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
target_compile_options(MNasNetMain PRIVATE -std=c++11 -g)
else()
target_compile_options(MNasNetMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(MNasNetMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(MNasNetMain MNasNetNet)

add_nestc_test(NAME MNasNet COMMAND ${CMAKE_CURRENT_BINARY_DIR}/MNasNet USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)