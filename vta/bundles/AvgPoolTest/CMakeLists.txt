if(NESTC_USE_VTASIM)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestMainEntry.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.avgPoolTest
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestMainEntry.h
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/AvgPoolTest/vtaAvgPoolTestBundle.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.cpp --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/AvgPoolTest/vtaAvgPoolTestInput.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestInput.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/AvgPoolTest/vtaAvgPoolTestBundle.weights.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.weights.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/AvgPoolTest/vtaAvgPoolTestGolden.bin ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestGolden.bin --no-sign-request
            COMMAND
            aws s3 cp s3://nestc-data-pub/VTABundleTests/AvgPoolTest/vtaAvgPoolTestMainEntry.hp ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestMainEntry.h --no-sign-request
    )
endif()

add_custom_target(vtaAvgPoolTestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaAvgPoolTestBundle vtaAvgPoolTestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.cpp)
add_dependencies(vtaAvgPoolTestBundle  ${BUNDLE_DEPENDS})
target_link_libraries(vtaAvgPoolTestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaAvgPoolTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaAvgPoolTestBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestGolden.bin)
