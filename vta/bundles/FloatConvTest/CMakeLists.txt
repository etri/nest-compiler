if(NESTC_USE_VTASIM)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestMainEntry.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.conv_gpu_0_conv1
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestGolden.bin
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/FloatConvTest/vtaFloatConvTestBundle.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.cpp --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/FloatConvTest/vtaFloatConvTestInput.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestInput.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/FloatConvTest/vtaFloatConvTestBundle.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.weights.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/FloatConvTest/vtaFloatConvTestGolden.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestGolden.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/FloatConvTest/vtaFloatConvTestMainEntry.h ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestMainEntry.h --no-sign-request
    )
endif()
add_custom_target(vtaFloatConvTestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestGolden.bin)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaFloatConvTestBundle vtaFloatConvTestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.cpp)
add_dependencies(vtaFloatConvTestBundle vtaFloatConvTestBundleNET ${BUNDLE_DEPENDS})
target_link_libraries(vtaFloatConvTestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaFloatConvTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaFloatConvTestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestGolden.bin)