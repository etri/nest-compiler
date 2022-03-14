
set(BNN_ONNX ${CMAKE_SOURCE_DIR}/vta/bundles/bnn)

add_custom_command(
        OUTPUT
        #${CMAKE_CURRENT_BINARY_DIR}/profile.yml
        ${CMAKE_CURRENT_BINARY_DIR}/binarynet.onnx
        #${CMAKE_CURRENT_BINARY_DIR}/cat_3.png
        #COMMAND
        #aws s3 cp s3://nestc-data-pub/VTABundleTests/bnn/profile.yaml ${CMAKE_CURRENT_BINARY_DIR}/profile.yaml --no-sign-request
        COMMAND
        aws s3 cp s3://nestc-data-pub/VTABundleTests/bnn/binarynet.onnx ${CMAKE_CURRENT_BINARY_DIR}/binarynet.onnx --no-sign-request
        #COMMAND
        #aws s3 cp s3://nestc-data-pub/VTABundleTests/bnn/cat_3.png ${CMAKE_CURRENT_BINARY_DIR}/cat_3.png --no-sign-request
        )
#[[
if(NESTC_USE_VTASIM)
    add_custom_target(vtaBinarynetBundleONNX
	    DEPENDS
            ${CMAKE_CURRENT_BINARY_DIR}/binarynet.onnx
            ${CMAKE_CURRENT_BINARY_DIR}/profile.yml
	    )

    add_custom_command(
	    OUTPUT
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.cpp
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.weights.bin
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.h
	    ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
	    COMMAND
	    model-compiler -g    
            -model=${CMAKE_CURRENT_BINARY_DIR}/binarynet.onnx
            -backend=VTA
            -assert-all-nodes-quantized
            -bnn
            -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
            -bundle-api=dynamic
            -model-input-name=input,float,[1,3,32,32]
            -dump-graph-DAG=opt.dot
            -load-profile=${CMAKE_CURRENT_BINARY_DIR}/profile.yml
            -onnx-define-symbol=batch_size,1
	    -quantization-precision-bias=Int32
	    DEPENDS
	    model-compiler VTA vtaBinarynetBundleONNX
	    )
else()
    add_custom_command(
	    OUTPUT
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.cpp
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.weights.bin
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.h
	    ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
	    COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/bnn/binarynet.cpp ${CMAKE_CURRENT_BINARY_DIR}/binarynet.cpp --no-sign-request
	    COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/bnn/binarynet.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/binarynet.weights.bin --no-sign-request
	    COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/bnn/binarynet.h ${CMAKE_CURRENT_BINARY_DIR}/binarynet.h --no-sign-request
	    COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/bnn/VTARuntime.h ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h --no-sign-request
            )
endif() 

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaBinarynetBundle ${BNN_ONNX}/main.cpp ${CMAKE_CURRENT_BINARY_DIR}/binarynet.cpp)
add_dependencies(vtaBinarynetBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaBinarynetBundle VTABundle png ${BUNDLE_LINK_LIB})
set_target_properties(vtaBinarynetBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_RUN_WITH_GENERIC_BUNDLE)
    add_nestc_test(ZCU102 NAME vtaBinarynetBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaBinarynetBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/cat_3.png)
else()
    add_nestc_test(NAME vtaBinarynetBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaBinarynetBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/cat_3.png)
endif()
]]

#set(BNN_ONNX ${CMAKE_SOURCE_DIR}/vta/bundles/bnn)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/binarynet.yaml
	COMMAND
        wget -O ${CMAKE_CURRENT_BINARY_DIR}/binarynet.yaml https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/yaml/binarynet.yaml
        )

if(NESTC_USE_VTASIM)
    add_custom_target(vtaBinarynetBundleONNX
	    DEPENDS
            ${CMAKE_CURRENT_BINARY_DIR}/binarynet.onnx
            ${CMAKE_CURRENT_BINARY_DIR}/binarynet.yaml
	    )

    add_custom_command(
	    OUTPUT
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.cpp
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.weights.bin
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.h
	    ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
	    COMMAND
	    model-compiler -g    
            -model=${CMAKE_CURRENT_BINARY_DIR}/binarynet.onnx
            -backend=VTA
            -assert-all-nodes-quantized
            -bnn
            -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
            -bundle-api=dynamic
            -model-input-name=input,float,[1,3,32,32]
            -dump-graph-DAG=opt.dot
            -load-profile=${CMAKE_CURRENT_BINARY_DIR}/binarynet.yaml
            -onnx-define-symbol=batch_size,1
	    -quantization-precision-bias=Int32
	    DEPENDS
	    model-compiler VTA vtaBinarynetBundleONNX
	    )
else()
    add_custom_command(
	    OUTPUT
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.cpp
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.weights.bin
	    ${CMAKE_CURRENT_BINARY_DIR}/binarynet.h
	    ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
	    COMMAND
            wget -O ${CMAKE_CURRENT_BINARY_DIR}/binarynet.cpp https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/bnn/binarynet.cpp
	    COMMAND
            wget -O ${CMAKE_CURRENT_BINARY_DIR}/binarynet.weights.bin https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/bnn/binarynet.weights.bin
	    COMMAND
            wget -O ${CMAKE_CURRENT_BINARY_DIR}/binarynet.h https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/bnn/binarynet.h
	    COMMAND
            wget -O ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/bnn/VTARuntime.h
            )
endif()

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaBinarynetBundle ${BNN_ONNX}/main.cpp ${CMAKE_CURRENT_BINARY_DIR}/binarynet.cpp)
add_dependencies(vtaBinarynetBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaBinarynetBundle VTABundle png ${BUNDLE_LINK_LIB})
set_target_properties(vtaBinarynetBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_RUN_WITH_GENERIC_BUNDLE)
    add_nestc_test(ZCU102 NAME vtaBinarynetBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaBinarynetBundle USE_SH 1 PARAMS ${BNN_ONNX}/cat_3.png)
else()
    add_nestc_test(NAME vtaBinarynetBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaBinarynetBundle USE_SH 1 PARAMS ${BNN_ONNX}/cat_3.png)
endif()


