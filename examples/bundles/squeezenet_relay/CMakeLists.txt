set(MODEL_INPUT_NAME "data_0")

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/squeezenet.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/squeezenet.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/squeezenet.onnx
)

add_custom_target(SqueezeNetBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/squeezenet.onnx
        )


add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/squeezenet.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        DEPENDS
        model-compiler Relay SqueezeNetBundleONNX
)

add_custom_target(SqueezeNetRelayBundle DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py)
