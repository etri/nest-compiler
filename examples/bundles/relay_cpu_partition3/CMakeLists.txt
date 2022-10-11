if(NESTC_WITH_TVM)
    option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
    set(MODEL_INPUT_NAME "data")
    #set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

    # Output directories for regular and quantized outputs
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
    set(RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_custom_target(ResNet18RelayPartition3BundleDir ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}
            )

    # Final Executables
    # =================
    # Regular

    add_executable(ResNet18RelayPartition3Bundle $<TARGET_OBJECTS:ResNet18RelayPartition3BundleMain>)
    set_target_properties(ResNet18RelayPartition3Bundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY})
    if(NOT NEST_BUILD_ON_MAC)
        target_link_libraries(ResNet18RelayPartition3Bundle ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p2.o ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p3.o png tvm_runtime)
    else()
        target_link_libraries(ResNet18RelayPartition3Bundle ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p2.o ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p3.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a tvm_runtime)
    endif()

    add_dependencies(ResNet18RelayPartition3Bundle ResNet18RelayPartition3BundleMain ResNet18RelayPartition3BundleNet)

    # Glow Bundles
    # ============
    # Regular Bundle
    add_custom_command(
            OUTPUT
            ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/relay__p0/genCode.py
            ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/relay__p1/genCode.py
            ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/relay__p2/genCode.py
    #        ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p1.o
    #        ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p2.o
            ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p3.o
            COMMAND
            model-partition-tuner -model=${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
            -model-input=${MODEL_INPUT_NAME},float,[1,224,224,3]
            -load-device-configs=${NESTC_SOURCE_DIR}/tests/runtime_test/RelayTestDeviceConfigs.yaml
            -partition-plan=${NESTC_SOURCE_DIR}/tests/runtime_test/partition_perform_profile/RelayTestPartitionPlan3.yaml
            -exe-type=0
            -emit-bundle=${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}
            -bundle-api=dynamic
    #        -relay-target=aarch64
    #        -relay-export-option=aarch64
    #        -relay-opt-level=3
            DEPENDS
            model-partition-tuner ResNet18RelayPartition3BundleDir MxnetResnet18ONNXDownload
    )

    add_custom_command(
            TARGET ResNet18RelayPartition3Bundle POST_BUILD
            COMMAND
            ls ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/main.cpp)

    add_custom_command(
            TARGET ResNet18RelayPartition3Bundle POST_BUILD
            COMMAND
            ls ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/CMakeLists.txt)

    if(GLOW_WITH_VTA_BUNDLE_TEST)
        add_custom_command(
                TARGET ResNet18RelayPartition3Bundle POST_BUILD
                COMMAND ${CMAKE_COMMAND}
                -E chdir ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY} ./ResNet18RelayPartition3Bundle ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png
        )
        add_custom_command(
                TARGET ResNet18RelayPartition3Bundle POST_BUILD
                COMMAND
                echo "ResNet18PartitionBundle Test Succeed")
    endif()

    add_custom_target(ResNet18RelayPartition3BundleNet DEPENDS ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p0.o ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p1.o ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p2.o ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/p3.o)

    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/module__p0/p0.cpp
            COMMAND
            python3 relay__p0/genCode.py
            DEPENDS ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/relay__p0/genCode.py tvm
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
            ${CMAKE_CURRENT_BINARY_DIR}/module__p1/p1.cpp
            COMMAND
            python3 relay__p1/genCode.py
            DEPENDS ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/relay__p1/genCode.py tvm
    )

    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/module__p1/p1.o
            ${CMAKE_CURRENT_BINARY_DIR}/module__p1/p1.h
            ${CMAKE_CURRENT_BINARY_DIR}/p1_tvm.so
            COMMAND
            make
            COMMAND
            cp ${CMAKE_CURRENT_BINARY_DIR}/module__p1/p1_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
            WORKING_DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}/module__p1/
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p1/p1.cpp
    )

    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/p1.o
            ${CMAKE_CURRENT_BINARY_DIR}/p1.h
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/module__p1/p1.o
            ${CMAKE_CURRENT_BINARY_DIR}/p1.o
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/module__p1/p1.h
            ${CMAKE_CURRENT_BINARY_DIR}/p1.h
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__p1/p1.o ${CMAKE_CURRENT_BINARY_DIR}/module__p1/p1.h
    )


    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/module__p2/p2.cpp
            COMMAND
            python3 relay__p2/genCode.py
            DEPENDS ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY}/relay__p2/genCode.py tvm
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

    include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__p0/)
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__p1)
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__p2)
    include_directories(BEFORES
            ${NESTC_ROOT_DIR}/tvm/include
            ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
            ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
            )

    # Other
    # =====
    # Driver program with main function for regular bundle
    add_library(ResNet18RelayPartition3BundleMain OBJECT main.cpp)
    if(NOT NEST_BUILD_ON_MAC)
        target_compile_options(ResNet18RelayPartition3BundleMain PRIVATE -std=c++11 -g)
    else()
        target_compile_options(ResNet18RelayPartition3BundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
    endif()
    target_include_directories(ResNet18RelayPartition3BundleMain PUBLIC ${RESNET18_RELAY_PARTITION3_BUNDLE_OUTPUT_DIRECTORY})
    add_dependencies(ResNet18RelayPartition3BundleMain ResNet18RelayPartition3BundleNet)

    add_nestc_test(NAME ResNet18RelayPartition3Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ResNet18RelayPartition3Bundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
endif()