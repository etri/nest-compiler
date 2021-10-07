add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
)

add_custom_target(resnet18RelayQNNONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay__mxnet_exported_resnet18/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        #-model-input=data,int8q,4.0,0,[1,224,224,3]
        -model-input=data,float,[1,224,224,3]
        -dump-graph-DAG=opt.dot
        -load-profile="qnn_test.yaml"
        -quantization-schema="symmetric_with_power2_scale"              
        DEPENDS
        model-compiler Relay resnet18RelayQNNONNX
)
if(0)
    add_custom_target(resnet18RelayQNNBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__mxnet_exported_resnet18/genCode.py)

    include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/)
    include_directories(BEFORE
            ${NESTC_ROOT_DIR}/tvm/include
            ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
            ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
            )
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.cpp
            COMMAND
            python3 relay__mxnet_exported_resnet18/genCode.py
            DEPENDS resnet18RelayQNNBundleNet tvm
    )

    add_custom_target(resnet18RelayQNNBundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.cpp)

    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.o
            ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18_tvm.so
            COMMAND
            make
            WORKING_DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/
            DEPENDS resnet18RelayQNNBundleCPPGen
    )

    add_custom_target(resnet18RelayQNNBundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.o)
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/)
    include_directories(BEFORE
            ${NESTC_ROOT_DIR}/tvm/include
            ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
            ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
            )

    add_executable(resnet18RelayQNNBundle deploy.cc ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.o ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18_tvm.so)
    set_target_properties(resnet18RelayQNNBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(resnet18RelayQNNBundle -fPIC -I. -pthread png tvm_runtime)
    target_compile_features(resnet18RelayQNNBundle PRIVATE cxx_std_17)
    add_dependencies(resnet18RelayQNNBundle resnet18RelayQNNBundleObjectGen tvm_runtime)
endif()