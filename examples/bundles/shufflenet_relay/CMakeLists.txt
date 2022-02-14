set(MODEL_INPUT_NAME "gpu_0/data_0")

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/shufflenet.onnx
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/shufflenet.onnx ${CMAKE_CURRENT_BINARY_DIR}/shufflenet.onnx --no-sign-request
)

add_custom_target(ShufflenetBundleRelayONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/shufflenet.onnx
        )
        
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay__shufflenet/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/shufflenet.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        DEPENDS
        model-compiler Relay ShufflenetBundleRelayONNX
)



add_custom_target(ShufflenetRelayBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__shufflenet/genCode.py)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__shufflenet/shufflenet.cpp
        COMMAND
        python3 relay__shufflenet/genCode.py
        DEPENDS ShufflenetRelayBundleNet tvm
)

add_custom_target(ShufflenetRelayBundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__shufflenet/shufflenet.cpp)        

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__shufflenet/shufflenet.o
        ${CMAKE_CURRENT_BINARY_DIR}/shufflenet_tvm.so
        COMMAND
        make
        COMMAND
        cp ${CMAKE_CURRENT_BINARY_DIR}/module__shufflenet/shufflenet_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/module__shufflenet/
        DEPENDS ShufflenetRelayBundleCPPGen
)

add_custom_target(ShufflenetRelayBundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__shufflenet/shufflenet.o)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__shufflenet/)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )

add_executable(ShufflenetRelayBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}//module__shufflenet/shufflenet.o)
set_target_properties(ShufflenetRelayBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ShufflenetRelayBundle -fPIC -I. -pthread png tvm_runtime)
target_compile_features(ShufflenetRelayBundle PRIVATE cxx_std_17)
add_dependencies(ShufflenetRelayBundle ShufflenetRelayBundleObjectGen tvm_runtime)
add_nestc_test(NAME ShufflenetRelayBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ShufflenetRelayBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)



