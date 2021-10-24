option(NESTC_USE_PRECOMPILED_BUNDLE_FROM_AWS "Download bundles from AWS S3" OFF)
option(NESTC_RELAY_TARGET_MALI "Generate bundles for publish" OFF)
if(NESTC_OPENCL_TEST)
    if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
                COMMAND
                wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx --no-verbose
        )

        add_custom_target(resnet18RelayOpenCLONNX
                DEPENDS
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18.onnx
                )
        if(NOT NESTC_RELAY_TARGET_MALI)
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
                    -relay-opt-level=3
                    DEPENDS
                    model-compiler Relay resnet18RelayOpenCLONNX
            )
        else()
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
                    -relay-target="mali"
                    -relay-target-host="aarch64"
                    -relay-export-option="aarch64"
                    -relay-opt-level=3
                    DEPENDS
                    model-compiler Relay resnet18RelayOpenCLONNX
            )
        endif()
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
    else()
        if(NESTC_USE_PRECOMPILED_BUNDLE_FROM_AWS)
            add_custom_command(
                    OUTPUT
                    ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.h
                    ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/Makefile
                    ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_tvm.so
                    COMMAND
                    mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18
                    COMMAND
                    aws s3 cp s3://$ENV{S3_BUCKET}/examples/bundles/resnet18_relay_opencl/module__mxnet_exported_resnet18 ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18 --recursive
                    COMMAND
                    cp ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
            )
            add_custom_target(resnet18RelayOpenCLBundleCPPGen DEPENDS
                    ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.h
                    ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/Makefile
                    ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_tvm.so
                    )
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

            add_definitions(-Wall -Wnon-virtual-dtor -g -fno-omit-frame-pointer -O0 -fPIE -fvisibility=hidden -fvisibility-inlines-hidden   -pthread -std=c++14)
            add_executable(resnet18RelayOpenCLBundle deploy.cc ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.o)
            set_target_properties(resnet18RelayOpenCLBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
            target_link_libraries(resnet18RelayOpenCLBundle -fPIC -I. -pthread png tvm_runtime)
            target_compile_features(resnet18RelayOpenCLBundle PRIVATE cxx_std_17)
            add_dependencies(resnet18RelayOpenCLBundle resnet18RelayOpenCLBundleObjectGen tvm_runtime)
            add_nestc_test(NAME resnet18RelayOpenCLBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/resnet18RelayOpenCLBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
        endif()
    endif()
endif()