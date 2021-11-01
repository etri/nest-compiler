INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/glow/tests/googletest/googletest/include)
add_executable(NESTVTABackendTest NESTVTABackendTest.cpp ../../lib/Backends/VTA/tests/VTABackendTestUtils.cpp)
target_include_directories(NESTVTABackendTest
        PUBLIC
        "${SYNAPSE_INCLUDE_DIR}")

target_link_libraries(NESTVTABackendTest
        PRIVATE

        Backends
        Base

        ExecutionEngine
        Graph
        LLVMSupport
        TestMain
        )
set_target_properties( NESTVTABackendTest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${NESTC_BINARY_DIR}/tests"
        )

add_nestc_test(NESTVTABackendTest ${NESTC_BINARY_DIR}/tests/NESTVTABackendTest --gtest_output=xml:VTASaverTest.xml)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnet18/mxnet_exported_resnet18_calib_1.yaml -O ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
)

add_custom_target(NESTVTAImageclassifierResnet18ONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/results.txt
        COMMAND
        image-classifier ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png
        -m=${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
        -backend=VTA
        -model-input-name=data
        -image-layout=NHWC
        -compute-softmax
        -topk=1
        -image-mode=0to255
        -load-profile=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
        -quantization-schema=symmetric_with_power2_scale
        -expected-labels=281 > results.txt
        COMMAND
        image-classifier ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png
        -m=${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
        -backend=VTAInterpreter
        -model-input-name=data
        -image-layout=NHWC
        -compute-softmax
        -topk=1
        -image-mode=0to255
        -load-profile=${CMAKE_CURRENT_BINARY_DIR}/mxnet_exported_resnet18_calib_1.yaml
        -quantization-schema=symmetric_with_power2_scale
        -expected-labels=281 >> results.txt
        COMMAND
        cat results.txt
        DEPENDS
        image-classifier NESTVTAImageclassifierResnet18ONNX MxnetResnet18ONNXDownload
)

add_custom_target(NESTVTAImageclassifierResnet18Test DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/results.txt)
