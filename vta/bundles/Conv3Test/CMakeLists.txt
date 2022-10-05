
if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.convTest3
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv3Test/vtaConv3TestBundle.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.cpp --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv3Test/vtaConv3TestInput.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestInput.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv3Test/vtaConv3TestBundle.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.weights.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv3Test/vtaConv3TestGolden.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestGolden.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv3Test/vtaConv3TestMainEntry.h ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestMainEntry.h --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/VTAConv3Test/VTARuntime.h ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h --no-sign-request

    )
endif()
add_custom_target(vtaConv3TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vtaConv3TestBundle vtaConv3TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.cpp)
add_dependencies(vtaConv3TestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaConv3TestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaConv3TestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaConv3TestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestGolden.bin)