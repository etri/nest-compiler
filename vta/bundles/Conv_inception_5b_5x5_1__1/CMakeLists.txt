
if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Bundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Input.bin
            ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Bundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Golden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1MainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.Conv_inception_5b_5x5_1__1
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Bundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Input.bin
            ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Bundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Golden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1MainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/Conv_inception_5b_5x5_1__1/Conv_inception_5b_5x5_1__1Bundle.cpp ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Bundle.cpp --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/Conv_inception_5b_5x5_1__1/Conv_inception_5b_5x5_1__1Input.bin ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Input.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/Conv_inception_5b_5x5_1__1/Conv_inception_5b_5x5_1__1Bundle.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Bundle.weights.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/Conv_inception_5b_5x5_1__1/Conv_inception_5b_5x5_1__1Golden.bin  ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Golden.bin  --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/Conv_inception_5b_5x5_1__1/Conv_inception_5b_5x5_1__1MainEntry.h ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1MainEntry.h --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/Conv_inception_5b_5x5_1__1/VTARuntime.h ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h --no-sign-request

    )
endif()
add_custom_target(Conv_inception_5b_5x5_1__1BundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Bundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Input.bin
        ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Bundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Golden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(Conv_inception_5b_5x5_1__1Bundle Conv_inception_5b_5x5_1__1BundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Bundle.cpp)
add_dependencies(Conv_inception_5b_5x5_1__1Bundle ${BUNDLE_DEPENDS})
target_link_libraries(Conv_inception_5b_5x5_1__1Bundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( Conv_inception_5b_5x5_1__1Bundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME Conv_inception_5b_5x5_1__1Bundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Bundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Input.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/Conv_inception_5b_5x5_1__1Golden.bin)