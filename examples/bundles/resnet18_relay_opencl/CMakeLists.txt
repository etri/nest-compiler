if(NESTC_OPENCL_TEST)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
    )

    add_custom_target(resnet18RelayOpenCLONNX
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
            -relay-target="opencl"
            -relay-target-host="llvm"
            DEPENDS
            model-compiler Relay resnet18RelayOpenCLONNX
    )

    add_custom_target(resnet18RelayOpenCLBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__mxnet_exported_resnet18/genCode.py)
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
            DEPENDS resnet18RelayOpenCLBundleNet tvm
    )

    add_custom_target(resnet18RelayOpenCLBundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.cpp)


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
            DEPENDS resnet18RelayOpenCLBundleCPPGen
    )
    add_custom_target(resnet18RelayOpenCLBundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.o)
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/)
    include_directories(BEFORE
            ${NESTC_ROOT_DIR}/tvm/include
            ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
            ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
            )

    add_executable(resnet18RelayOpenCLBundle deploy.cc ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.o)
    set_target_properties(resnet18RelayOpenCLBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(resnet18RelayOpenCLBundle -fPIC -I. -pthread png tvm_runtime)
    target_compile_features(resnet18RelayOpenCLBundle PRIVATE cxx_std_17)
    add_dependencies(resnet18RelayOpenCLBundle resnet18RelayOpenCLBundleObjectGen tvm_runtime)
endif()