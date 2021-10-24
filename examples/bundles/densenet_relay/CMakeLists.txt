set(MODEL_INPUT_NAME "data_0")
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/densenet.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/densenet-9.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/densenet.onnx --no-verbose
)

add_custom_target(DenseNetRelayBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/densenet.onnx
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay__densenet/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/densenet.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        -relay-opt-level=3
        DEPENDS
        model-compiler Relay DenseNetRelayBundleONNX
)

add_custom_target(DenseNetRelayBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__densenet/genCode.py)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__densenet/densenet.cpp
        COMMAND
        python3 relay__densenet/genCode.py
        DEPENDS DenseNetRelayBundleNet tvm
)

add_custom_target(DenseNetRelayBundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__densenet/densenet.cpp)        

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__densenet/densenet.o
        ${CMAKE_CURRENT_BINARY_DIR}/densenet_tvm.so
        COMMAND
        make
        COMMAND
        cp ${CMAKE_CURRENT_BINARY_DIR}/module__densenet/densenet_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/module__densenet/
        DEPENDS DenseNetRelayBundleCPPGen
)

add_custom_target(DenseNetRelayBundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__densenet/densenet.o)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__densenet/)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )

add_executable(DenseNetRelayBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}//module__densenet/densenet.o)
set_target_properties(DenseNetRelayBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(DenseNetRelayBundle -fPIC -I. -pthread png tvm_runtime)
target_compile_features(DenseNetRelayBundle PRIVATE cxx_std_17)
add_dependencies(DenseNetRelayBundle DenseNetRelayBundleObjectGen tvm_runtime)
add_nestc_test(NAME DenseNetRelayBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/DenseNetRelayBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)


