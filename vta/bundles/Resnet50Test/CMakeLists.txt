if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/caffe2_resnet50.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/caffe2_resnet50.h
            ${CMAKE_CURRENT_BINARY_DIR}/caffe2_resnet50.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            model-compiler -g
            -keep-original-precision-for-nodes=SoftMax
            -model=${GLOW_SOURCE_DIR}/tests/models/nestModels/caffe2resnet50/caffe2_resnet50.onnx
            -backend=VTA
            -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
            -bundle-api=dynamic
            -model-input-name=gpu_0/data_0,float,[1,3,224,224]
            -dump-graph-DAG=opt.dot
            -load-profile=${GLOW_SOURCE_DIR}/tests/models/nestModels/caffe2resnet50/2_caffe2_resnet50_CPU.yaml
            -quantization-schema=symmetric_with_power2_scale
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/caffe2_resnet50.weights.bin
            ${CMAKE_CURRENT_SOURCE_DIR}/caffe2_resnet50.weights_temp.bin
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/caffe2_resnet50.h
            ${CMAKE_CURRENT_SOURCE_DIR}/caffe2_resnet50_temp.h
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            ${CMAKE_CURRENT_SOURCE_DIR}/VTARuntime_temp.h
            DEPENDS
            model-compiler VTA
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/caffe2_resnet50.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/caffe2_resnet50.h
            ${CMAKE_CURRENT_BINARY_DIR}/caffe2_resnet50.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/caffe2_resnet50.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/caffe2_resnet50.cpp
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/caffe2_resnet50_temp.h
            ${CMAKE_CURRENT_BINARY_DIR}/caffe2_resnet50.h
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/VTARuntime_temp.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/caffe2_resnet50.weights_temp.bin
            ${CMAKE_CURRENT_BINARY_DIR}/caffe2_resnet50.weights.bin)

endif()
add_custom_target(vtaCaffe2Resnet50BundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/caffe2_resnet50.cpp )
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaCaffe2Resnet50Bundle caffe2_resnet50Main.cpp ${CMAKE_CURRENT_BINARY_DIR}/caffe2_resnet50.cpp)
add_dependencies(vtaCaffe2Resnet50Bundle vtaCaffe2Resnet50BundleNet)
target_link_libraries(vtaCaffe2Resnet50Bundle VTABundle png Miniz ${BUNDLE_LINK_LIB})
set_target_properties(vtaCaffe2Resnet50Bundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaCaffe2Resnet50Bundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            #${CMAKE_CURRENT_SOURCE_DIR}/cat_285.png
            #${CMAKE_CURRENT_BINARY_DIR}/cat_285.png)
            ${CMAKE_CURRENT_SOURCE_DIR}/data0001.bin
            ${CMAKE_CURRENT_BINARY_DIR}/data0001.bin)

    add_custom_command(
            TARGET vtaCaffe2Resnet50Bundle POST_BUILD
            COMMAND
            mkdir -p debug)
    add_custom_command(
                TARGET vtaCaffe2Resnet50Bundle POST_BUILD
                COMMAND
            vtaCaffe2Resnet50Bundle data0001.bin)
    add_custom_command(
            TARGET vtaCaffe2Resnet50Bundle POST_BUILD
            COMMAND
            echo "vtaCaffe2Resnet50Bundle Test Succeed")
endif()