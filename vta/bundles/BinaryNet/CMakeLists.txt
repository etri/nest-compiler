set(BINARYNET_ONNX ${CMAKE_SOURCE_DIR}/vta/bundles/BinaryNet)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/binarynet.onnx
        ${CMAKE_CURRENT_BINARY_DIR}/cat_3.png
        COMMAND
        aws s3 cp s3://nestc-data-pub/VTABundleTests/BinaryNet/binarynet.onnx ${CMAKE_CURRENT_BINARY_DIR}/binarynet.onnx --no-sign-request
        COMMAND
        aws s3 cp s3://nestc-data-pub/VTABundleTests/BinaryNet/cat_3.png ${CMAKE_CURRENT_BINARY_DIR}/cat_3.png --no-sign-request
        )

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
	add_custom_command(
		    OUTPUT
		    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.yaml
		    COMMAND
		    image-classifier ${CMAKE_CURRENT_BINARY_DIR}/cat_3.png -i=0to1 
		    -m=${CMAKE_CURRENT_BINARY_DIR}/binarynet.onnx
		    -onnx-define-symbol=batch_size,1
		    -model-input-name=input
		    -dump-profile=${CMAKE_CURRENT_BINARY_DIR}/binarynet.yaml
		    -use-cifar10-normalization
		    DEPENDS
		    image-classifier ${CMAKE_CURRENT_BINARY_DIR}/binarynet.onnx ${CMAKE_CURRENT_BINARY_DIR}/cat_3.png
		    )

	add_custom_command(
	    OUTPUT
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.cpp
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.weights.bin
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.h
	    ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
	    COMMAND
	    model-compiler
	    -model=${CMAKE_CURRENT_BINARY_DIR}/binarynet.onnx
	    -backend=VTA
	    -assert-all-nodes-quantized
	    -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
	    -bundle-api=dynamic
	    -model-input-name=input,float,[1,3,32,32]
	    -load-profile=${CMAKE_CURRENT_BINARY_DIR}/binarynet.yaml
            -quantization-schema=symmetric_with_power2_scale
	    -onnx-define-symbol=batch_size,1
	    DEPENDS
	    model-compiler VTA ${CMAKE_CURRENT_BINARY_DIR}/binarynet.yaml
	    )
else()
    add_custom_command(
	    OUTPUT
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.cpp
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.weights.bin
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.h
	    ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
	    COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/BinaryNet/binarynet.cpp ${CMAKE_CURRENT_BINARY_DIR}/binarynet.cpp --no-sign-request
	    COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/BinaryNet/binarynet.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/binarynet.weights.bin --no-sign-request
	    COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/BinaryNet/binarynet.h ${CMAKE_CURRENT_BINARY_DIR}/binarynet.h --no-sign-request
	    COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/BinaryNet/VTARuntime.h ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h --no-sign-request
            )
endif() 

add_custom_target(vtaBinarynetBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/binarynet.cpp ${CMAKE_CURRENT_BINARY_DIR}/binarynet.weights.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaBinarynetBundle ${BINARYNET_ONNX}/main.cpp ${CMAKE_CURRENT_BINARY_DIR}/binarynet.cpp)
add_dependencies(vtaBinarynetBundle vtaBinarynetBundleNet ${BUNDLE_DEPENDS})
target_link_libraries(vtaBinarynetBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaBinarynetBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )


add_nestc_test(ZCU102 NAME vtaBinarynetBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaBinarynetBundle USE_SH 1 PARAMS ${BINARYNET_ONNX}/cat_3.png)

