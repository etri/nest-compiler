set(MODEL_INPUT_NAME "gpu_0/data_0")

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/shufflenet.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/shufflenet.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/shufflenet.onnx
)

add_custom_target(ShufflenetBundleRelayONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/shufflenet.onnx
        )
        
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/shufflenet.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        DEPENDS
        model-compiler Relay ShufflenetBundleRelayONNX
)

add_custom_target(ShufflenetRelayBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py)
