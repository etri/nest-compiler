set(MODEL_INPUT_NAME "gpu_0/data_0")

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/zfnet.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/zfnet512-9.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/zfnet.onnx --no-verbose
)

add_custom_target(ZfNetRelayBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/zfnet.onnx
        )


add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay__zfnet/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/zfnet.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        DEPENDS
        model-compiler Relay ZfNetRelayBundleONNX
)


add_custom_target(ZfNetRelayBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__zfnet/genCode.py)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__zfnet/zfnet.cpp
        COMMAND
        python3 relay__zfnet/genCode.py
        DEPENDS ZfNetRelayBundleNet tvm
)

add_custom_target(ZfNetRelayBundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__zfnet/zfnet.cpp)        

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__zfnet/zfnet.o
        ${CMAKE_CURRENT_BINARY_DIR}/zfnet_tvm.so
        COMMAND
        make
        COMMAND
        cp ${CMAKE_CURRENT_BINARY_DIR}/module__zfnet/zfnet_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/module__zfnet/
        DEPENDS ZfNetRelayBundleCPPGen
)

add_custom_target(ZfNetRelayBundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__zfnet/zfnet.o)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__zfnet/)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )

add_executable(ZfNetRelayBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}//module__zfnet/zfnet.o)
set_target_properties(ZfNetRelayBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ZfNetRelayBundle -fPIC -I. -pthread png tvm_runtime)
target_compile_features(ZfNetRelayBundle PRIVATE cxx_std_17)
add_dependencies(ZfNetRelayBundle ZfNetRelayBundleObjectGen tvm_runtime)
#add_nestc_test(NAME ZfNetRelayBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ZfNetRelayBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
