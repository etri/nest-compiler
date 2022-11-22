
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/openedge/resnet50_openedge.yaml
        COMMAND
        image-classifier ${NESTC_ROOT_DIR}/glow/tests/images/imagenet/cat_285.png
        -m=${NESTC_BINARY_DIR}/models/resnet50-v2-7.onnx
        -model-input-name=data
        -backend=CPU
        -compute-softmax
        -topk=5
        -dump-profile=${CMAKE_CURRENT_BINARY_DIR}/openedge/resnet50_openedge.yaml
        DEPENDS image-classifier openedgeResnet50ONNXDownload
)

add_custom_target(resnet50_openedgeBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/openedge/resnet50_openedge.yaml)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/openedge/resnet50_openedge.onnx
        COMMAND
        python3 ${NESTC_ROOT_DIR}/onnx_yaml-parser/extract_model/src/onnx_parser.py --onnx_file ${NESTC_BINARY_DIR}/models/resnet50-v2-7.onnx
        --yaml_dir ${CMAKE_CURRENT_BINARY_DIR}/openedge/
        --save_dir ${CMAKE_CURRENT_BINARY_DIR}/openedge/
        DEPENDS resnet50_openedgeBundleNet
)

add_custom_target(resnet50_openedgeBundle
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/openedge/resnet50_openedge.onnx
        )
