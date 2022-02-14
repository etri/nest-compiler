set(MODEL_INPUT_NAME "data_0")

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/googlenet.onnx
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/googlenet_v4.onnx ${CMAKE_CURRENT_BINARY_DIR}/googlenet.onnx --no-sign-request
)

add_custom_target(GoogleNetRelayBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/googlenet.onnx
        )
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay__googlenet/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/googlenet.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        -relay-opt-level=3
        DEPENDS
        model-compiler Relay GoogleNetRelayBundleONNX
)

add_custom_target(GoogleNetRelayBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__googlenet/genCode.py)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__googlenet/googlenet.cpp
        COMMAND
        python3 relay__googlenet/genCode.py
        DEPENDS GoogleNetRelayBundleNet tvm
)

add_custom_target(GoogleNetRelayBundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__googlenet/googlenet.cpp)        

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__googlenet/googlenet.o
        ${CMAKE_CURRENT_BINARY_DIR}/googlenet_tvm.so
        COMMAND
        make
        COMMAND
        cp ${CMAKE_CURRENT_BINARY_DIR}/module__googlenet/googlenet_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/module__googlenet/
        DEPENDS GoogleNetRelayBundleCPPGen
)

add_custom_target(GoogleNetRelayBundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__googlenet/googlenet.o)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__googlenet/)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )

add_executable(GoogleNetRelayBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}//module__googlenet/googlenet.o)
set_target_properties(GoogleNetRelayBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(GoogleNetRelayBundle -fPIC -I. -pthread png tvm_runtime)
target_compile_features(GoogleNetRelayBundle PRIVATE cxx_std_17)
add_dependencies(GoogleNetRelayBundle GoogleNetRelayBundleObjectGen tvm_runtime)
add_nestc_test(NAME GoogleNetRelayBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GoogleNetRelayBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)

