option(NESTC_TARGET_AARCH64 "Generate bundles for publish" OFF)
option(VTA_RESNET18_WITH_SKIPQUANT0 "Build the LLVM-based JIT CPU backend" OFF)
option(NESTC_USE_PRECOMPILED_BUNDLE_FROM_AWS "Download bundles from AWS S3" OFF)

set(MODEL_INPUT_NAME "data")
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18_calib_1.yaml -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
)

add_custom_target(vtaMxnetResnet18RelayEVTAPartitionBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
)
if(NESTC_TARGET_AARCH64)
        add_custom_command(
                OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o
                ${CMAKE_CURRENT_BINARY_DIR}/p0.h ${CMAKE_CURRENT_BINARY_DIR}/p1.h ${CMAKE_CURRENT_BINARY_DIR}/p2.h ${CMAKE_CURRENT_BINARY_DIR}/p3.h ${CMAKE_CURRENT_BINARY_DIR}/p4.h ${CMAKE_CURRENT_BINARY_DIR}/p5.h ${CMAKE_CURRENT_BINARY_DIR}/p6.h ${CMAKE_CURRENT_BINARY_DIR}/p7.h ${CMAKE_CURRENT_BINARY_DIR}/p8.h ${CMAKE_CURRENT_BINARY_DIR}/p9.h ${CMAKE_CURRENT_BINARY_DIR}/p10.h ${CMAKE_CURRENT_BINARY_DIR}/p11.h ${CMAKE_CURRENT_BINARY_DIR}/p12.h ${CMAKE_CURRENT_BINARY_DIR}/p13.h ${CMAKE_CURRENT_BINARY_DIR}/p14.h ${CMAKE_CURRENT_BINARY_DIR}/p15.h ${CMAKE_CURRENT_BINARY_DIR}/p16.h ${CMAKE_CURRENT_BINARY_DIR}/p17.h
                ${CMAKE_CURRENT_BINARY_DIR}/p0.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p1.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p2.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p3.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p4.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p5.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p6.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p7.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p8.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p9.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p10.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p11.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p12.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p13.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p14.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p15.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p16.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/p17.weights.bin
                ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                COMMAND
                model-partition-tuner -model=${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
                -model-input=${MODEL_INPUT_NAME},float,[1,224,224,3]
                -target=aarch64 -mcpu=cortex-a53
                -load-device-configs=${NESTC_ROOT_DIR}/tests/runtime_test/VTATestDeviceConfigs.yaml
                -partition-plan=${NESTC_ROOT_DIR}/tests/runtime_test/partition_perform_profile/NESTOptimalFusedPlan.yaml
                -exe-type=0
                -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
                -bundle-api=dynamic
                -load-vta-profile=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
                DEPENDS
                model-partition-tuner VTA vtaMxnetResnet18RelayEVTAPartitionBundleONNX MxnetResnet18ONNXDownload)
        add_custom_target(vtaMxnetResnet18RelayEVTAPartitionBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o)
else()
        if(NESTC_USE_VTASIM)
                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/relay__p0/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/relay__p2/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/relay__p4/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/relay__p5/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/relay__p7/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/relay__p9/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/relay__p11/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/relay__p13/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/relay__p15/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/relay__p17/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
                        COMMAND
                        model-partition-tuner -model=${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
                        -model-input=${MODEL_INPUT_NAME},float,[1,224,224,3]
                        -load-device-configs=${NESTC_ROOT_DIR}/tests/runtime_test/RelayEVTATestDeviceConfigs.yaml
                        -partition-plan=${NESTC_ROOT_DIR}/tests/runtime_test/partition_perform_profile/NESTRelayEVTATestPlan.yaml
                        -exe-type=0
                        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
                        -bundle-api=dynamic
                        -load-vta-profile=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
#                        -relay-target=aarch64
#                        -relay-export-option=aarch64
#                        -relay-opt-level=3

                        DEPENDS
                        model-partition-tuner VTA vtaMxnetResnet18RelayEVTAPartitionBundleONNX)

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p0/p0.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p2/p2.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p4/p4.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p5/p5.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p7/p7.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p9/p9.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p11/p11.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p13/p13.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p15/p15.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p17/p17.cpp
                        COMMAND
                        python3 relay__p0/genCode.py
                        COMMAND
                        python3 relay__p2/genCode.py
                        COMMAND
                        python3 relay__p4/genCode.py
                        COMMAND
                        python3 relay__p5/genCode.py
                        COMMAND
                        python3 relay__p7/genCode.py
                        COMMAND
                        python3 relay__p9/genCode.py
                        COMMAND
                        python3 relay__p11/genCode.py
                        COMMAND
                        python3 relay__p13/genCode.py
                        COMMAND
                        python3 relay__p15/genCode.py
                        COMMAND
                        python3 relay__p17/genCode.py
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__p0/genCode.py tvm
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p0/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p0/p0.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p0_tvm.so
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/module__p0/p0_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p0/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p0/p0.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p0.h
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p0/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p0.o
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p0/p0.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p0.h
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p0/p0.o ${CMAKE_CURRENT_BINARY_DIR}/module__p0/p0.h
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p2/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p2/p2.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p2_tvm.so
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/module__p2/p2_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p2/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p2/p2.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p2.h
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p2/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p2.o
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p2/p2.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p2.h
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p2/p2.o ${CMAKE_CURRENT_BINARY_DIR}/module__p2/p2.h
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p4/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p4/p4.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p4_tvm.so
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/module__p4/p4_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p4/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p4/p4.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p4.h
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p4/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p4.o
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p4/p4.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p4.h
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p4/p4.o ${CMAKE_CURRENT_BINARY_DIR}/module__p4/p4.h
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p5/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p5/p5.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p5_tvm.so
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/module__p5/p5_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p5/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p5/p5.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p5.h
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p5/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p5.o
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p5/p5.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p5.h
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p5/p5.o ${CMAKE_CURRENT_BINARY_DIR}/module__p5/p5.h
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p7/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p7/p7.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p7_tvm.so
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/module__p7/p7_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p7/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p7/p7.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p7.h
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p7/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p7.o
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p7/p7.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p7.h
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p7/p7.o ${CMAKE_CURRENT_BINARY_DIR}/module__p7/p7.h
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p9/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p9/p9.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p9_tvm.so
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/module__p9/p9_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p9/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p9/p9.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p9.h
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p9/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p9.o
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p9/p9.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p9.h
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p9/p9.o ${CMAKE_CURRENT_BINARY_DIR}/module__p9/p9.h
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p11/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p11/p11.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p11_tvm.so
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/module__p11/p11_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p11/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p11/p11.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p11.h
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p11/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p11.o
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p11/p11.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p11.h
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p11/p11.o ${CMAKE_CURRENT_BINARY_DIR}/module__p11/p11.h
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p13/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p13/p13.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p13_tvm.so
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/module__p13/p13_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p13/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p13/p13.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p13.h
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p13/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p13.o
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p13/p13.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p13.h
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p13/p13.o ${CMAKE_CURRENT_BINARY_DIR}/module__p13/p13.h
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p15/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p15/p15.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p15_tvm.so
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/module__p15/p15_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p15/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p15/p15.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p15.h
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p15/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p15.o
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p15/p15.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p15.h
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p15/p15.o ${CMAKE_CURRENT_BINARY_DIR}/module__p15/p15.h
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p17/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p17/p17.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p17_tvm.so
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/module__p17/p17_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p17/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p17/p17.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p17.h
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p17/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/p17.o
                        COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/module__p17/p17.h
                        ${CMAKE_CURRENT_BINARY_DIR}/p17.h
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p17/p17.o ${CMAKE_CURRENT_BINARY_DIR}/module__p17/p17.h
                )

                add_custom_target(vtaMxnetResnet18RelayEVTAPartitionBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o)
                INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o)
                add_dependencies(vtaMxnetResnet18RelayEVTAPartitionBundle vtaMxnetResnet18RelayEVTAPartitionBundleNet ${BUNDLE_DEPENDS})

                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundle VTABundle ${BUNDLE_LINK_LIB} tvm_runtime)
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundle
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )

#                add_nestc_test(NAME vtaMxnetResnet18RelayEVTAPartitionBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
        endif()
endif()

if(NOT NESTC_USE_VTASIM)
        if(NESTC_USE_PRECOMPILED_BUNDLE_FROM_AWS)
                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o
                        COMMAND
                        aws s3 cp s3://$ENV{S3_BUCKET}/vta/bundles/Resnet18PartitionTest/ ${CMAKE_CURRENT_BINARY_DIR}/ --recursive
                )
                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o)
                add_dependencies(vtaMxnetResnet18RelayEVTAPartitionBundle ${BUNDLE_DEPENDS})

                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundle VTABundle ${BUNDLE_LINK_LIB})
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundle
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )
                INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
                add_nestc_test(ZCU102 NAME vtaMxnetResnet18RelayEVTAPartitionBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
        endif()
endif()