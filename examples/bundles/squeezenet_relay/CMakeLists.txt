set(MODEL_INPUT_NAME "data_0")


add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/squeezenet.onnx
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/squeezenet.onnx ${CMAKE_CURRENT_BINARY_DIR}/squeezenet.onnx --no-sign-request
)

add_custom_target(SqueezeNetRelayBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/squeezenet.onnx
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay__squeezenet/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/squeezenet.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        DEPENDS
        model-compiler Relay SqueezeNetRelayBundleONNX
)

add_custom_target(SqueezeNetRelayBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__squeezenet/genCode.py)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__squeezenet/squeezenet.cpp
        COMMAND
        python3 relay__squeezenet/genCode.py
        DEPENDS SqueezeNetRelayBundleNet tvm
)

add_custom_target(SqueezeNetRelayBundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__squeezenet/squeezenet.cpp)        

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__squeezenet/squeezenet.o
        ${CMAKE_CURRENT_BINARY_DIR}/squeezenet_tvm.so
        COMMAND
        make
        COMMAND
        cp ${CMAKE_CURRENT_BINARY_DIR}/module__squeezenet/squeezenet_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/module__squeezenet/
        DEPENDS SqueezeNetRelayBundleCPPGen
)

add_custom_target(SqueezeNetRelayBundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__squeezenet/squeezenet.o)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__squeezenet/)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )

add_executable(SqueezeNetRelayBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}//module__squeezenet/squeezenet.o)
set_target_properties(SqueezeNetRelayBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(SqueezeNetRelayBundle -fPIC -I. -pthread png tvm_runtime)
target_compile_features(SqueezeNetRelayBundle PRIVATE cxx_std_17)
add_dependencies(SqueezeNetRelayBundle SqueezeNetRelayBundleObjectGen tvm_runtime)
add_nestc_test(NAME SqueezeNetRelayBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/SqueezeNetRelayBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
