add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/onnxzoo_mobilenet_v210.onnx
        COMMAND
        aws s3 cp s3://nestc-data-pub/models/onnxzoo_mobilenet_v210.onnx ${CMAKE_CURRENT_BINARY_DIR}/onnxzoo_mobilenet_v210.onnx --no-sign-request
)

add_custom_target(MobileNetRelayONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/onnxzoo_mobilenet_v210.onnx
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay__onnxzoo_mobilenet_v210/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/onnxzoo_mobilenet_v210.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=data,float,[1,3,224,224]
        DEPENDS
        model-compiler Relay MobileNetRelayONNX
)


add_custom_target(MobileNetRelayBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__onnxzoo_mobilenet_v210/genCode.py)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__onnxzoo_mobilenet_v210/onnxzoo_mobilenet_v210.cpp
        COMMAND
        python3 relay__onnxzoo_mobilenet_v210/genCode.py
        DEPENDS MobileNetRelayBundleNet tvm
)

add_custom_target(MobileNetRelayBundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__onnxzoo_mobilenet_v210/onnxzoo_mobilenet_v210.cpp)        

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__onnxzoo_mobilenet_v210/onnxzoo_mobilenet_v210.o
        ${CMAKE_CURRENT_BINARY_DIR}/onnxzoo_mobilenet_v210_tvm.so
        COMMAND
        make
        COMMAND
        cp ${CMAKE_CURRENT_BINARY_DIR}/module__onnxzoo_mobilenet_v210/onnxzoo_mobilenet_v210_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/module__onnxzoo_mobilenet_v210/
        DEPENDS MobileNetRelayBundleCPPGen
)

add_custom_target(MobileNetRelayBundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__onnxzoo_mobilenet_v210/onnxzoo_mobilenet_v210.o)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__onnxzoo_mobilenet_v210/)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )

add_executable(MobileNetRelayBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}//module__onnxzoo_mobilenet_v210/onnxzoo_mobilenet_v210.o)
set_target_properties(MobileNetRelayBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(MobileNetRelayBundle -fPIC -I. -pthread png tvm_runtime)
target_compile_features(MobileNetRelayBundle PRIVATE cxx_std_17)
add_dependencies(MobileNetRelayBundle MobileNetRelayBundleObjectGen tvm_runtime)
add_nestc_test(NAME MobileNetRelayBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/MobileNetRelayBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)



