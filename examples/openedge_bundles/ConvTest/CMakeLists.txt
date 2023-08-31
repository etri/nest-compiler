add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/resnet50_1_1.yaml
        COMMAND
        model-compiler
        -m=${NESTC_BINARY_DIR}/models/resnet50_1_1.onnx
        -model-input=data,float,[1,3,224,224]
        -backend=Enlight
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -dump-graph-DAG=test.dot
        DEPENDS model-compiler openedgeresnet50_1_1Download
)

add_custom_target(enlightConvBundle DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/resnet50_1_1.yaml)
#add_custom_target(resnet_enlightConvBundle
#        DEPENDS
#        ${NESTC_ROOT_DIR}/examples/openedge_bundles/ConvTest/resnet50_1_1.onnx
#        )

add_custom_command(
        TARGET enlightConvBundle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/resnet50_1_1.yaml ${CMAKE_CURRENT_BINARY_DIR}/output.bin
        COMMAND diff ${CMAKE_CURRENT_BINARY_DIR}/output.bin ${NESTC_ROOT_DIR}/examples/openedge_bundles/ConvTest/GoldenTest.yaml
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/resnet50_1_1.yaml model-compiler
        )

#add_nestc_test(NAME enlightConvBundle
#        COMMAND ${CMAKE_CURRENT_BINARY_DIR} USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/output.bin
#        USE_DIFF 1 DIFF_TARGET ${NESTC_ROOT_DIR}/examples/openedge_bundles/ConvTest/GoldenTest.yaml
#        )

