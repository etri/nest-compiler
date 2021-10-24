set(MODEL_INPUT_NAME "data_0")

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/vgg19.onnx
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/models/vgg19-caffe2-9.onnx -O ${CMAKE_CURRENT_BINARY_DIR}/vgg19.onnx --no-verbose
)

add_custom_target(VGG19BundleRelayONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/vgg19.onnx
        )


add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/relay__vgg19/genCode.py
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_BINARY_DIR}/vgg19.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
        DEPENDS
        model-compiler Relay VGG19BundleRelayONNX
)



add_custom_target(VGG19RelayBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relay__vgg19/genCode.py)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__vgg19/vgg19.cpp
        COMMAND
        python3 relay__vgg19/genCode.py
        DEPENDS VGG19RelayBundleNet tvm
)

add_custom_target(VGG19RelayBundleCPPGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__vgg19/vgg19.cpp)        

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/module__vgg19/vgg19.o
        ${CMAKE_CURRENT_BINARY_DIR}/vgg19_tvm.so
        COMMAND
        make
        COMMAND
        cp ${CMAKE_CURRENT_BINARY_DIR}/module__vgg19/vgg19_tvm.so ${CMAKE_CURRENT_BINARY_DIR}/
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/module__vgg19/
        DEPENDS VGG19RelayBundleCPPGen
)

add_custom_target(VGG19RelayBundleObjectGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/module__vgg19/vgg19.o)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/module__vgg19/)
include_directories(BEFORE
        ${NESTC_ROOT_DIR}/tvm/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dlpack/include
        ${NESTC_ROOT_DIR}/tvm/3rdparty/dmlc-core/include
        )

add_executable(VGG19RelayBundle main.cpp ${CMAKE_CURRENT_BINARY_DIR}//module__vgg19/vgg19.o)
set_target_properties(VGG19RelayBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(VGG19RelayBundle -fPIC -I. -pthread png tvm_runtime)
target_compile_features(VGG19RelayBundle PRIVATE cxx_std_17)
add_dependencies(VGG19RelayBundle VGG19RelayBundleObjectGen tvm_runtime)
add_nestc_test(NAME VGG19RelayBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/VGG19RelayBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png)
