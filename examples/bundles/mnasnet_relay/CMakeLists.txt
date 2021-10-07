set(MODEL_INPUT_NAME "input")

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mnasnet.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/mnasnet.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/mnasnet.onnx
)

add_custom_target(MNasNetONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/mnasnet.onnx
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/inception.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        DEPENDS
        model-compiler Relay MNasNetONNX
)

add_custom_target(MNasNetRelayBundle DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py)
