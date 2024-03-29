option(NESTC_USE_PRECOMPILED_BUNDLE_FROM_AWS "Download bundles from AWS S3" OFF)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)

    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/relay__mxnet_exported_resnet18/genCode.py
            COMMAND
            model-compiler -g
            -model=${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
            -backend=Relay
            -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
            -bundle-api=dynamic
            -model-input=data,float,[1,224,224,3]
            -relay-target='aarch64'
            -relay-export-option='aarch64'
            -relay-opt-level=3
            -dump-graph-DAG=opt.dot
            DEPENDS
            model-compiler Relay MxnetResnet18ONNXDownload
    )

    add_custom_target(resnet18RelayAarch64BundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__mxnet_exported_resnet18/genCode.py)
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
            DEPENDS resnet18RelayAarch64BundleNet tvm
    )

    add_custom_target(resnet18RelayAarch64BundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.cpp)

elseif(NESTC_EVTA_RUN_ON_ZCU102)
    if(NOT NESTC_USE_PRECOMPILED_BUNDLE_FROM_AWS)
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.h
                ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/Makefile
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_tvm.so
                COMMAND
                mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18
                COMMAND
                aws s3 cp s3://nestc-data-pub/RelayBundleTest/Resnet18AarchTest/mxnet_exported_resnet18.cpp ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.cpp
                COMMAND
                aws s3 cp s3://nestc-data-pub/RelayBundleTest/Resnet18AarchTest/mxnet_exported_resnet18.h ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.h
               COMMAND
                aws s3 cp s3://nestc-data-pub/RelayBundleTest/Resnet18AarchTest/mxnet_exported_resnet18_tvm.so ${CMAKE_CURRENT_BINARY_DIR}//mxnet_exported_resnet18_tvm.so
               COMMAND
                aws s3 cp s3://nestc-data-pub/RelayBundleTest/Resnet18AarchTest/Makefile ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/Makefile
        )
    else()

        if(DEFINED $ENV{S3_BUCKET})
            set(AWS_LOC $ENV{S3_BUCKET})
        else()
            set(AWS_LOC "nestc-pub")
        endif()
        if(${AWS_LOC} STREQUAL "nestc-pub")
            set(AWS_PARAM "--no-sign-request")
        endif()

        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.h
                ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/Makefile
                ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_tvm.so
                COMMAND
                mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18
                COMMAND
                aws s3 cp s3://${AWS_LOC}/examples/bundles/resnet18_relay_aarch64/module__mxnet_exported_resnet18 ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18 --recursive
                ARGS ${AWS_PARAM}
                COMMAND
                cp ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/

        )
    endif()
    add_custom_target(resnet18RelayAarch64BundleCPPGen DEPENDS
            ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.h
            ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/Makefile
            ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_tvm.so
    )

    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.o
            COMMAND
            make
            WORKING_DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/
            DEPENDS resnet18RelayAarch64BundleCPPGen
    )
    add_custom_target(resnet18RelayAarch64BundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.o)
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/)

    include_directories(BEFORE
            ${NESTC_ROOT_DIR}/tvm/include
            ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
            ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
            )

    add_definitions(-Wall -Wnon-virtual-dtor -g -fno-omit-frame-pointer -O0 -fPIE -fvisibility=hidden -fvisibility-inlines-hidden   -pthread -std=c++14)
    add_executable(resnet18RelayAarch64Bundle deploy.cc ${CMAKE_CURRENT_BINARY_DIR}/module__mxnet_exported_resnet18/mxnet_exported_resnet18.o ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_tvm.so)
    set_target_properties(resnet18RelayAarch64Bundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(resnet18RelayAarch64Bundle -fPIC -I. -pthread png tvm_runtime)
    target_compile_features(resnet18RelayAarch64Bundle PRIVATE cxx_std_17)
    add_dependencies(resnet18RelayAarch64Bundle tvm_runtime resnet18RelayAarch64BundleObjectGen)
    add_nestc_test(ONLY_ZCU102 NAME resnet18RelayAarch64Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/resnet18RelayAarch64Bundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
endif()