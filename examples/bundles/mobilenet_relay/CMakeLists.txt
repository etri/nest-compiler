add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/onnxzoo_mobilenet_v210.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/onnxzoo_mobilenet_v210.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/onnxzoo_mobilenet_v210.onnx
)

add_custom_target(MobileNetRelayONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/onnxzoo_mobilenet_v210.onnx
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/onnxzoo_mobilenet_v210.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=data,float,[1,3,224,224]
        DEPENDS
        model-compiler Relay MobileNetRelayONNX
)

add_custom_target(MobileNetRelayBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py)
