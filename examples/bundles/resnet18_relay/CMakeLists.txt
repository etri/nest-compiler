
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
)

add_custom_target(resnet18RelayONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=data,float,[1,224,224,3]
        -dump-graph-DAG=opt.dot
        DEPENDS
        model-compiler Relay resnet18RelayONNX
)

add_custom_target(resnet18RelayBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module/mxnet_exported_resnet18.cpp
        COMMAND
        python3 relay/genCode.py
        DEPENDS resnet18RelayBundleNet tvm
)

add_custom_target(resnet18RelayBundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module/mxnet_exported_resnet18.cpp)


add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module/mxnet_exported_resnet18.o
        ${CMAKE_CURRENT_BINARY_DIR}/module/mxnet_exported_resnet18_tvm.so
        COMMAND
        make
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/module/
        DEPENDS resnet18RelayBundleCPPGen
)
add_custom_target(resnet18RelayBundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module/mxnet_exported_resnet18.o)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/module/)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )

add_executable(resnet18RelayBundle deploy.cc ${CMAKE_CURRENT_BINARY_DIR}/module/mxnet_exported_resnet18.o ${CMAKE_CURRENT_BINARY_DIR}/module/mxnet_exported_resnet18_tvm.so)
set_target_properties(resnet18RelayBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(resnet18RelayBundle -fPIC -I. -pthread png tvm_runtime)
target_compile_features(resnet18RelayBundle PRIVATE cxx_std_17)
add_dependencies(resnet18RelayBundle resnet18RelayBundleObjectGen tvm_runtime)
add_nestc_test(NAME resnet18RelayBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/resnet18RelayBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)