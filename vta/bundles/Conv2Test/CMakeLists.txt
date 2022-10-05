if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.convTest2
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv2Test/vtaConv2TestBundle.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.cpp --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv2Test/vtaConv2TestInput.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestInput.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv2Test/vtaConv2TestBundle.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.weights.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv2Test/vtaConv2TestGolden.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestGolden.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv2Test/vtaConv2TestMainEntry.h ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestMainEntry.h --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv2Test/VTARuntime.h ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h --no-sign-request

    )
endif()
add_custom_target(vtaConv2TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vtaConv2TestBundle vtaConv2TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.cpp)
add_dependencies(vtaConv2TestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaConv2TestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaConv2TestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaConv2TestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestGolden.bin)