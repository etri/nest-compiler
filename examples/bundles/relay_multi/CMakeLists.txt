add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/output.txt
        COMMAND
        model-compiler -g
        -model=${CMAKE_CURRENT_SOURCE_DIR}/ex_multi.onnx
        -backend=Relay
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=0,float,[1,5,5]
        -model-input=1,float,[1,5,5]
        -dump-graph-DAG=opt.dot > output.txt
        DEPENDS
	model-compiler Relay 
)

add_custom_target(relayMultiBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output.txt)
