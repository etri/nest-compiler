set(MODEL_INPUT_NAME "input")


add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/resnext50.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/resnext50.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/resnext50.onnx
)

add_custom_target(ResNext50BundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/resnext50.onnx
        )

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/resnext50.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        DEPENDS
        model-compiler Relay ResNext50BundleONNX
)

add_custom_target(ResNext50RelayBundle DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py)
