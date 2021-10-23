set(MODEL_INPUT_NAME "input")


add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mnasnet.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/mnasnet.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/mnasnet.onnx
)

add_custom_target(MNasNetRelayONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/mnasnet.onnx
        )
	
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay__mnasnet/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/mnasnet.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        DEPENDS
        model-compiler Relay MNasNetRelayONNX
)


add_custom_target(MNasNetRelayBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__mnasnet/genCode.py)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__mnasnet/mnasnet.cpp
        COMMAND
        python3 relay__mnasnet/genCode.py
        DEPENDS MNasNetRelayBundleNet tvm
)

add_custom_target(MNasNetRelayBundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__mnasnet/mnasnet.cpp)        

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__mnasnet/mnasnet.o
        ${CMAKE_CURRENT_BINARY_DIR}/mnasnet_tvm.so
        COMMAND
        make
        COMMAND
        cp ${CMAKE_CURRENT_BINARY_DIR}/module__mnasnet/mnasnet_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/module__mnasnet/
        DEPENDS MNasNetRelayBundleCPPGen
)

add_custom_target(MNasNetRelayBundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__mnasnet/mnasnet.o)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__mnasnet/)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )

add_executable(MNasNetRelayBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}//module__mnasnet/mnasnet.o)
set_target_properties(MNasNetRelayBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(MNasNetRelayBundle -fPIC -I. -pthread png tvm_runtime)
target_compile_features(MNasNetRelayBundle PRIVATE cxx_std_17)
add_dependencies(MNasNetRelayBundle MNasNetRelayBundleObjectGen tvm_runtime)
add_nestc_test(NAME MNasNetRelayBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/MNasNetRelayBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)



