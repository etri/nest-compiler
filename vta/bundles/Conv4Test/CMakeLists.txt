
if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.convTest4
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv4Test/vtaConv4TestBundle.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestBundle.cpp --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv4Test/vtaConv4TestInput.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestInput.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv4Test/vtaConv4TestBundle.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestBundle.weights.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv4Test/vtaConv4TestGolden.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestGolden.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv4Test/vtaConv4TestMainEntry.h ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestMainEntry.h --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv4Test/VTARuntime.h ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h --no-sign-request

    )
endif()
add_custom_target(vtaConv4TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vtaConv4TestBundle vtaConv4TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestBundle.cpp)
add_dependencies(vtaConv4TestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaConv4TestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaConv4TestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaConv4TestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaConv4TestGolden.bin)