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
                -load-profile=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
                DEPENDS
                model-partition-tuner VTA vtaMxnetResnet18RelayEVTAPartitionBundleONNX MxnetResnet18ONNXDownload)
        add_custom_target(vtaMxnetResnet18RelayEVTAPartitionBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o)
else()
        if(NESTC_USE_VTASIM)

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/relay__p0/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/relay__p2/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/relay__p4/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/relay__p5/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/relay__p7/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/relay__p9/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/relay__p11/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/relay__p13/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/relay__p15/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/relay__p17/genCode.py
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p16.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p17.o

                        COMMAND
                        model-partition-tuner -model=${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
                        -model-input=${MODEL_INPUT_NAME},float,[1,224,224,3]
                        -load-device-configs=${NESTC_ROOT_DIR}/tests/runtime_test/RelayEVTATestDeviceConfigs.yaml
                        -partition-plan=${NESTC_ROOT_DIR}/tests/runtime_test/partition_perform_profile/NESTRelayEVTATestPlan.yaml
                        -exe-type=0
                        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}/Relay/
                        -bundle-api=dynamic
                        -load-profile=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
#                        -relay-target=aarch64
#                        -relay-export-option=aarch64
#                        -relay-opt-level=3

                        COMMAND
                        model-partition-tuner -model=${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
                        -model-input=${MODEL_INPUT_NAME},float,[1,224,224,3]
                        -load-device-configs=${NESTC_ROOT_DIR}/tests/runtime_test/VTATestDeviceConfigs.yaml
                        -partition-plan=${NESTC_ROOT_DIR}/tests/runtime_test/partition_perform_profile/NESTOptimalFusedPlan.yaml
                        -exe-type=0
                        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}/CPU/
                        -bundle-api=dynamic
                        -load-profile=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml

                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/CPU/*.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/
                        DEPENDS
                        model-partition-tuner VTA vtaMxnetResnet18RelayEVTAPartitionBundleONNX MxnetResnet18ONNXDownload)

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.cpp
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
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay
                        DEPENDS
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/relay__p0/genCode.py
                        tvm
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.o
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.o
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.o
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.o
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.o
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.o
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.o
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.o
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.o
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.cpp
                )

                add_custom_command(
                        OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.o
                        COMMAND
                        make
                        COMMAND
                        cp ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
                        WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/
                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.cpp
                )


                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundleTestNone mainNone.cpp
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p16.cpp
                        )

                target_include_directories(vtaMxnetResnet18RelayEVTAPartitionBundleTestNone
                        PUBLIC
                        AFTER
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/
                        )
                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundleTestNone VTABundle ${BUNDLE_LINK_LIB} tvm_runtime)
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundleTestNone
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )



                ## START vtaMxnetResnet18RelayEVTAPartitionBundleTest0
                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundleTest0 main0.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p16.cpp
                        )

                target_include_directories(vtaMxnetResnet18RelayEVTAPartitionBundleTest0
                        PUBLIC
                        AFTER
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/
                        )
                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundleTest0 VTABundle ${BUNDLE_LINK_LIB} tvm_runtime)
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundleTest0
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )
                ## END vtaMxnetResnet18RelayEVTAPartitionBundleTest0


                ## START vtaMxnetResnet18RelayEVTAPartitionBundleTest2


                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundleTest2 main2.cpp
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.o
                                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.o
                        #                        ${CMAKE_CURRENT_BIvtaMxnetResnet18RelayEVTAPartitionBundleNARY_DIR}/Relay/module__p4/p4.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/CPU/p0.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p16.cpp
                        )

                target_include_directories(vtaMxnetResnet18RelayEVTAPartitionBundleTest2
                        PUBLIC
                        AFTER
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/
                        )
                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundleTest2 VTABundle ${BUNDLE_LINK_LIB} tvm_runtime)
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundleTest2
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )
                ## END vtaMxnetResnet18RelayEVTAPartitionBundleTest2



## START vtaMxnetResnet18RelayEVTAPartitionBundleTest4
                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundleTest4 main4.cpp
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p2.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p16.cpp
                        )

                target_include_directories(vtaMxnetResnet18RelayEVTAPartitionBundleTest4
                        PUBLIC
                        AFTER
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/
                        )
                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundleTest4 VTABundle ${BUNDLE_LINK_LIB} tvm_runtime)
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundleTest4
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )
## END vtaMxnetResnet18RelayEVTAPartitionBundleTest4


                ## START vtaMxnetResnet18RelayEVTAPartitionBundleTest5
                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundleTest5 main5.cpp
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p2.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/CPU/p4.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p16.cpp
                        )

                target_include_directories(vtaMxnetResnet18RelayEVTAPartitionBundleTest5
                        PUBLIC
                        AFTER
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/
                        )
                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundleTest5 VTABundle ${BUNDLE_LINK_LIB} tvm_runtime)
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundleTest5
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )
                ## END vtaMxnetResnet18RelayEVTAPartitionBundleTest5




                ## START vtaMxnetResnet18RelayEVTAPartitionBundleTest7
                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundleTest7 main7.cpp
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p4.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/CPU/p5.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p16.cpp
                        )

                target_include_directories(vtaMxnetResnet18RelayEVTAPartitionBundleTest7
                        PUBLIC
                        AFTER
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/
                        )
                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundleTest7 VTABundle ${BUNDLE_LINK_LIB} tvm_runtime)
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundleTest7
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )
                ## END vtaMxnetResnet18RelayEVTAPartitionBundleTest7


                ## START vtaMxnetResnet18RelayEVTAPartitionBundleTest9
                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundleTest9 main9.cpp
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p5.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/CPU/p7.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p16.cpp
                        )

                target_include_directories(vtaMxnetResnet18RelayEVTAPartitionBundleTest9
                        PUBLIC
                        AFTER
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/
                        )
                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundleTest9 VTABundle ${BUNDLE_LINK_LIB} tvm_runtime)
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundleTest9
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )
                ## END vtaMxnetResnet18RelayEVTAPartitionBundleTest9



                ## START vtaMxnetResnet18RelayEVTAPartitionBundleTest11
                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundleTest11 main11.cpp
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p7.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/CPU/p9.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p16.cpp
                        )

                target_include_directories(vtaMxnetResnet18RelayEVTAPartitionBundleTest11
                        PUBLIC
                        AFTER
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/
                        )
                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundleTest11 VTABundle ${BUNDLE_LINK_LIB} tvm_runtime)
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundleTest11
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )
                ## END vtaMxnetResnet18RelayEVTAPartitionBundleTest11



                ## START vtaMxnetResnet18RelayEVTAPartitionBundleTest13
                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundleTest13 main13.cpp
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p9.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/CPU/p11.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p16.cpp
                        )

                target_include_directories(vtaMxnetResnet18RelayEVTAPartitionBundleTest13
                        PUBLIC
                        AFTER
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/
                        )
                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundleTest13 VTABundle ${BUNDLE_LINK_LIB} tvm_runtime)
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundleTest13
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )
                ## END vtaMxnetResnet18RelayEVTAPartitionBundleTest13




                ## START vtaMxnetResnet18RelayEVTAPartitionBundleTest15
                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundleTest15 main15.cpp
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p11.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/CPU/p13.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p16.cpp
                        )

                target_include_directories(vtaMxnetResnet18RelayEVTAPartitionBundleTest15
                        PUBLIC
                        AFTER
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/
                        )
                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundleTest15 VTABundle ${BUNDLE_LINK_LIB} tvm_runtime)
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundleTest15
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )
                ## END vtaMxnetResnet18RelayEVTAPartitionBundleTest15




                ## START vtaMxnetResnet18RelayEVTAPartitionBundleTest17
                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundleTest17 main17.cpp
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.o
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p13.o
                                                ${CMAKE_CURRENT_BINARY_DIR}/CPU/p15.o
#                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/p17.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p16.cpp
                        )

                target_include_directories(vtaMxnetResnet18RelayEVTAPartitionBundleTest17
                        PUBLIC
                        AFTER
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/
                        #                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/
#                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/
                                                ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/
                        )
                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundleTest17 VTABundle ${BUNDLE_LINK_LIB} tvm_runtime)
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundleTest17
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )
                ## END vtaMxnetResnet18RelayEVTAPartitionBundleTest17





                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundleTestAll mainAll.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/p0.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p1.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/p2.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p3.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/p4.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/p5.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p6.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/p7.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p8.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/p9.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p10.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/p11.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p12.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/p13.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p14.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/p15.o
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/p16.cpp
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/p17.o
                        )

                target_include_directories(vtaMxnetResnet18RelayEVTAPartitionBundleTestAll
                        PUBLIC
                        AFTER
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p0/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p2/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p4/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p5/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p7/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p9/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p11/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p13/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p15/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/module__p17/
                        ${CMAKE_CURRENT_BINARY_DIR}/Relay/
                        ${CMAKE_CURRENT_BINARY_DIR}/CPU/
                        )
                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundleTestAll VTABundle ${BUNDLE_LINK_LIB} tvm_runtime)
                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundleTestAll
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        )



                add_nestc_test(NAME vtaMxnetResnet18RelayEVTAPartitionBundleTestNone COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundleTestNone USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
                add_nestc_test(NAME vtaMxnetResnet18RelayEVTAPartitionBundleTest0 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundleTest0 USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
                add_nestc_test(NAME vtaMxnetResnet18RelayEVTAPartitionBundleTest2 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundleTest2 USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
                add_nestc_test(NAME vtaMxnetResnet18RelayEVTAPartitionBundleTest4 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundleTest4 USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
                add_nestc_test(NAME vtaMxnetResnet18RelayEVTAPartitionBundleTest5 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundleTest5 USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
                add_nestc_test(NAME vtaMxnetResnet18RelayEVTAPartitionBundleTest7 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundleTest7 USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
                add_nestc_test(NAME vtaMxnetResnet18RelayEVTAPartitionBundleTest9 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundleTest9 USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
                add_nestc_test(NAME vtaMxnetResnet18RelayEVTAPartitionBundleTest11 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundleTest11 USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
                add_nestc_test(NAME vtaMxnetResnet18RelayEVTAPartitionBundleTest13 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundleTest13 USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
                add_nestc_test(NAME vtaMxnetResnet18RelayEVTAPartitionBundleTest15 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundleTest15 USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
                add_nestc_test(NAME vtaMxnetResnet18RelayEVTAPartitionBundleTest17 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundleTest17 USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
                add_nestc_test(NAME vtaMxnetResnet18RelayEVTAPartitionBundleTestAll COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundleTestAll USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
        endif()
endif()

if(NOT NESTC_USE_VTASIM)
#        if(NESTC_USE_PRECOMPILED_BUNDLE_FROM_AWS)
#                add_custom_command(
#                        OUTPUT
#                        ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o
#                        COMMAND
#                        aws s3 cp s3://$ENV{S3_BUCKET}/vta/bundles/Resnet18PartitionTest/ ${CMAKE_CURRENT_BINARY_DIR}/ --recursive
#                )
#                add_executable(vtaMxnetResnet18RelayEVTAPartitionBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}/p0.o ${CMAKE_CURRENT_BINARY_DIR}/p1.cpp ${CMAKE_CURRENT_BINARY_DIR}/p2.o ${CMAKE_CURRENT_BINARY_DIR}/p3.cpp ${CMAKE_CURRENT_BINARY_DIR}/p4.o ${CMAKE_CURRENT_BINARY_DIR}/p5.o ${CMAKE_CURRENT_BINARY_DIR}/p6.cpp ${CMAKE_CURRENT_BINARY_DIR}/p7.o ${CMAKE_CURRENT_BINARY_DIR}/p8.cpp ${CMAKE_CURRENT_BINARY_DIR}/p9.o ${CMAKE_CURRENT_BINARY_DIR}/p10.cpp ${CMAKE_CURRENT_BINARY_DIR}/p11.o ${CMAKE_CURRENT_BINARY_DIR}/p12.cpp ${CMAKE_CURRENT_BINARY_DIR}/p13.o ${CMAKE_CURRENT_BINARY_DIR}/p14.cpp ${CMAKE_CURRENT_BINARY_DIR}/p15.o ${CMAKE_CURRENT_BINARY_DIR}/p16.cpp ${CMAKE_CURRENT_BINARY_DIR}/p17.o)
#                add_dependencies(vtaMxnetResnet18RelayEVTAPartitionBundle ${BUNDLE_DEPENDS})
#
#                target_link_libraries(vtaMxnetResnet18RelayEVTAPartitionBundle VTABundle ${BUNDLE_LINK_LIB})
#                set_target_properties(vtaMxnetResnet18RelayEVTAPartitionBundle
#                        PROPERTIES
#                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
#                        )
#                INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
#                add_nestc_test(ZCU102 NAME vtaMxnetResnet18RelayEVTAPartitionBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMxnetResnet18RelayEVTAPartitionBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
#        endif()
endif()