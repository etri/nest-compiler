if(NESTC_CUDA_TEST)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx --no-verbose
    )

    add_custom_target(resnet18RelayCUDAONNX
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
            -model-input=data,float,[1,224,224,3]
            -dump-graph-DAG=opt.dot
            -relay-target="cuda"
            DEPENDS
            model-compiler Relay resnet18RelayCUDAONNX
    )

    add_custom_target(resnet18RelayCUDABundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__mxnet_exported_resnet18/genCode.py)
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
            DEPENDS resnet18RelayCUDABundleNet tvm
    )

    add_custom_target(resnet18RelayCUDABundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.cpp)


    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.o
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_tvm.so
            COMMAND
            make
            COMMAND
            cp ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
            WORKING_DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/
            DEPENDS resnet18RelayCUDABundleCPPGen
    )
    add_custom_target(resnet18RelayCUDABundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.o)
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/)
    include_directories(BEFORE
            ${NESTC_ROOT_DIR}/tvm/include
            ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
            ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
            )

    add_executable(resnet18RelayCUDABundle deploy.cc ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.o)
    set_target_properties(resnet18RelayCUDABundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(resnet18RelayCUDABundle -fPIC -I. -pthread png tvm_runtime)
    target_compile_features(resnet18RelayCUDABundle PRIVATE cxx_std_17)
    add_dependencies(resnet18RelayCUDABundle resnet18RelayCUDABundleObjectGen tvm_runtime)
endif()