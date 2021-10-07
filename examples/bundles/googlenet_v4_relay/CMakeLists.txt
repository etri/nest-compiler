set(MODEL_INPUT_NAME "data_0")

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/googlenet.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/googlenet_v4.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/googlenet.onnx
)

add_custom_target(GoogleNetBundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/googlenet.onnx
        )
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/googlenet.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        DEPENDS
        model-compiler Relay GoogleNetBundleONNX
)

add_custom_target(GoogleNetRelayBundle DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py)
