add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/resnet50-v2-7.onnx
        COMMAND
        echo "make resnet50 onnx from yaml and diff original onnx"
        COMMAND
        python3 ${NESTC_ROOT_DIR}/onnx_yaml-parser/extract_model/src/onnx_parser.py --onnx_file ${NESTC_BINARY_DIR}/models/resnet50-v2-7.onnx --yaml_dir ${CMAKE_BINARY_DIR}/examples/openedge_bundles/resnet50Test/ --save_dir ${CMAKE_CURRENT_BINARY_DIR}/ --debug
        DEPENDS
        enlightResnet50Bundle openedgeGoldenResnet50Download

)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mobilenetv2-7.onnx
        COMMAND
        echo "make mobilenet onnx from yaml and diff original onnx"
        COMMAND
        python3 ${NESTC_ROOT_DIR}/onnx_yaml-parser/extract_model/src/onnx_parser.py --onnx_file ${NESTC_BINARY_DIR}/models/mobilenetv2-7.onnx --yaml_dir ${CMAKE_BINARY_DIR}/examples/openedge_bundles/mobilenetTest/ --save_dir ${CMAKE_CURRENT_BINARY_DIR}/ --debug
        DEPENDS
        enlightMobilenetBundle openedgeGoldenMobilenetv2Download
)

add_custom_target(onnxYamlParserValTestBunddle DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mobilenetv2-7.onnx ${CMAKE_CURRENT_BINARY_DIR}/resnet50-v2-7.onnx)

add_custom_command(
        TARGET onnxYamlParserValTestBunddle POST_BUILD
        COMMAND diff ${CMAKE_CURRENT_BINARY_DIR}/resnet50-v2-7.onnx ${NESTC_BINARY_DIR}/models/Golden_resnet50-v2-7.onnx
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/resnet50-v2-7.onnx
)

add_custom_command(
        TARGET onnxYamlParserValTestBunddle POST_BUILD
        COMMAND diff ${CMAKE_CURRENT_BINARY_DIR}/mobilenetv2-7.onnx ${NESTC_BINARY_DIR}/models/Golden_mobilenetv2-7.onnx
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mobilenetv2-7.onnx
)

#add_nestc_test(NAME enlightConvBundle
#        COMMAND ${CMAKE_CURRENT_BINARY_DIR} USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/output.bin
#        USE_DIFF 1 DIFF_TARGET ${NESTC_ROOT_DIR}/examples/openedge_bundles/ConvTest/GoldenTest.yaml
#        )

