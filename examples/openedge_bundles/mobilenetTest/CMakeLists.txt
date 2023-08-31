add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mobilenetv2-7.yaml
        COMMAND
        model-compiler
        -m=${NESTC_BINARY_DIR}/models/mobilenetv2-7.onnx
        -model-input=input,float,[1,3,224,224]
        -onnx-define-symbol=batch_size,1
        -backend=Enlight
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -dump-graph-DAG=test.dot
        DEPENDS model-compiler openedgeMobilenetv2ONNXDownload
)

add_custom_target(enlightMobilenetBundle DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mobilenetv2-7.yaml)

#add_custom_target(mobilenet_enlightMobilenetBundle
#        DEPENDS
#        ${NESTC_ROOT_DIR}/examples/openedge_bundles/mobilenetTest/mobilenetv2-7.onnx
#)

add_custom_command(
        TARGET enlightMobilenetBundle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/mobilenetv2-7.yaml ${CMAKE_CURRENT_BINARY_DIR}/output.bin
        COMMAND diff ${CMAKE_CURRENT_BINARY_DIR}/output.bin ${NESTC_ROOT_DIR}/examples/openedge_bundles/mobilenetTest/mobilenetGoldenTest.yaml
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mobilenetv2-7.yaml model-compiler
)

#add_nestc_test(NAME enlightConvBundle
#        COMMAND ${CMAKE_CURRENT_BINARY_DIR} USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/output.bin
#        USE_DIFF 1 DIFF_TARGET ${NESTC_ROOT_DIR}/examples/openedge_bundles/ConvTest/GoldenTest.yaml
#        )

