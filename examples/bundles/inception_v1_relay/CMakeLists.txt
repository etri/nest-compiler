set(MODEL_INPUT_NAME "data_0")

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/inception.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/inception_v1.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/inception.onnx
)

add_custom_target(InceptionRelayBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/inception.onnx
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay__inception/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/inception.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        DEPENDS
        model-compiler Relay InceptionRelayBundleONNX
)

add_custom_target(InceptionRelayBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__inception/genCode.py)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__inception/inception.cpp
        COMMAND
        python3 relay__inception/genCode.py
        DEPENDS InceptionRelayBundleNet tvm
)

add_custom_target(InceptionRelayBundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__inception/inception.cpp)        

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__inception/inception.o
        ${CMAKE_CURRENT_BINARY_DIR}/inception_tvm.so
        COMMAND
        make
        COMMAND
        cp ${CMAKE_CURRENT_BINARY_DIR}/module__inception/inception_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/module__inception/
        DEPENDS InceptionRelayBundleCPPGen
)

add_custom_target(InceptionRelayBundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__inception/inception.o)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__inception/)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )

add_executable(InceptionRelayBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}//module__inception/inception.o)
set_target_properties(InceptionRelayBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(InceptionRelayBundle -fPIC -I. -pthread png tvm_runtime)
target_compile_features(InceptionRelayBundle PRIVATE cxx_std_17)
add_dependencies(InceptionRelayBundle InceptionRelayBundleObjectGen tvm_runtime)
add_nestc_test(NAME InceptionRelayBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/InceptionRelayBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)


