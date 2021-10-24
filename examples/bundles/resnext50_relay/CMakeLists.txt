set(MODEL_INPUT_NAME "input")

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/resnext50.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/resnext50.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/resnext50.onnx --no-verbose
)

add_custom_target(ResNext50RelayBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/resnext50.onnx
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay__resnext50/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/resnext50.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        DEPENDS
        model-compiler Relay ResNext50RelayBundleONNX
)


add_custom_target(ResNext50RelayBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__resnext50/genCode.py)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__resnext50/resnext50.cpp
        COMMAND
        python3 relay__resnext50/genCode.py
        DEPENDS ResNext50RelayBundleNet tvm
)

add_custom_target(ResNext50RelayBundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__resnext50/resnext50.cpp)        

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__resnext50/resnext50.o
        ${CMAKE_CURRENT_BINARY_DIR}/resnext50_tvm.so
        COMMAND
        make
        COMMAND
        cp ${CMAKE_CURRENT_BINARY_DIR}/module__resnext50/resnext50_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/module__resnext50/
        DEPENDS ResNext50RelayBundleCPPGen
)

add_custom_target(ResNext50RelayBundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__resnext50/resnext50.o)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__resnext50/)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )

add_executable(ResNext50RelayBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}//module__resnext50/resnext50.o)
set_target_properties(ResNext50RelayBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ResNext50RelayBundle -fPIC -I. -pthread png tvm_runtime)
target_compile_features(ResNext50RelayBundle PRIVATE cxx_std_17)
add_dependencies(ResNext50RelayBundle ResNext50RelayBundleObjectGen tvm_runtime)
add_nestc_test(NAME ResNext50RelayBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ResNext50RelayBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)



