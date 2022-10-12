set(BIREAL_ONNX ${CMAKE_SOURCE_DIR}/vta/bundles/BiRealNet)

add_custom_command( 
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.onnx
        COMMAND
        aws s3 cp s3://nestc-data-pub/VTABundleTests/BiRealNet/BirealNet.onnx ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.onnx --no-sign-request
        )

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
	add_custom_command(
		    OUTPUT
		    ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.yaml
		    COMMAND
		    image-classifier
                    ${GLOW_SOURCE_DIR}/tests/images/imagenet/dog_207.png
		    -i=0to1
		    -compute-softmax
		    -topk=5
		    -m=${CMAKE_CURRENT_BINARY_DIR}/BirealNet.onnx
		    -onnx-define-symbol=batch_size,1
		    -model-input-name=input
		    -dump-profile=${CMAKE_CURRENT_BINARY_DIR}/BirealNet.yaml
		    -use-imagenet-normalization
		    DEPENDS
		    image-classifier ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.onnx
		    )

	add_custom_command(
		    OUTPUT
		    ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.cpp
		    ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.weights.bin
		    ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.h
		    ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
		    #COMMAND ${CMAKE_COMMAND} -E copy
		    #${CMAKE_CURRENT_SOURCE_DIR}/VTASkipQuantizeNodes.txt
		    #${CMAKE_CURRENT_BINARY_DIR}/VTASkipQuantizeNodes.txt
		    COMMAND
		    model-compiler
		    -model=${CMAKE_CURRENT_BINARY_DIR}/BirealNet.onnx
		    -keep-original-precision-for-nodes=SoftMax
		    -backend=VTA
		    -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
		    -bundle-api=dynamic
		    -model-input-name=input,float,[1,3,224,224]
		    -load-profile=${CMAKE_CURRENT_BINARY_DIR}/BirealNet.yaml
		    -quantization-schema=symmetric_with_power2_scale
		    -onnx-define-symbol=batch_size,1
		    -bnn-with-scale=true
		    DEPENDS
		    model-compiler VTA ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.yaml ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.onnx
		    )

    add_custom_target(vtaBirealBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.cpp ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.weights.bin)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
    add_executable(vtaBirealBundle ${BIREAL_ONNX}/main.cpp ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.cpp)
    add_dependencies(vtaBirealBundle vtaBirealBundleNet ${BUNDLE_DEPENDS})
    target_link_libraries(vtaBirealBundle VTABundle ${BUNDLE_LINK_LIB})
    set_target_properties(vtaBirealBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

else()
    add_custom_command(
	    OUTPUT
	    ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.cpp
	    ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.weights.bin
	    ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.h
	    ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
	    COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/BiRealNet/BirealNet.cpp ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.cpp --no-sign-request
	    COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/BiRealNet/BirealNet.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.weights.bin --no-sign-request
	    COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/BiRealNet/BirealNet.h ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.h --no-sign-request
	    COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/BiRealNet/VTARuntime.h ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h --no-sign-request
            )

    add_custom_target(vtaBirealBundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.cpp ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.weights.bin)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
    add_executable(vtaBirealBundle ${BIREAL_ONNX}/main.cpp ${CMAKE_CURRENT_BINARY_DIR}/BirealNet.cpp)
    add_dependencies(vtaBirealBundle vtaBirealBundleNet ${BUNDLE_DEPENDS})
    target_link_libraries(vtaBirealBundle VTABundle ${BUNDLE_LINK_LIB})
    set_target_properties(vtaBirealBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )


endif()

add_nestc_test(ZCU102 NAME vtaBirealBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaBirealBundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/dog_207.png)
