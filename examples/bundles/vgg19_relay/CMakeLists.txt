set(MODEL_INPUT_NAME "data_0")

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/vgg19.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/vgg19-caffe2-9.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/vgg19.onnx
)

add_custom_target(VGG19BundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/vgg19.onnx
        )
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/vgg19.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        DEPENDS
        model-compiler Relay VGG19BundleONNX
)

add_custom_target(VGG19RelayBundle DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay/genCode.py)
