if(GLOW_WITH_VTASIM)
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
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/AvgPoolTest/vtaAvgPoolTestBundle.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.cpp
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/AvgPoolTest/vtaAvgPoolTestInput.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestInput.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/AvgPoolTest/vtaAvgPoolTestBundle.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.weights.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/AvgPoolTest/vtaAvgPoolTestGolden.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestGolden.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/AvgPoolTest/vtaAvgPoolTestMainEntry.h -O ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestMainEntry.h
    )
endif()

add_custom_target(vtaAvgPoolTestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaAvgPoolTestBundle vtaAvgPoolTestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaAvgPoolTestBundle.cpp)
target_link_libraries(vtaAvgPoolTestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaAvgPoolTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaAvgPoolTestBundle POST_BUILD
            COMMAND
            vtaAvgPoolTestBundle vtaAvgPoolTestInput.bin)

    add_custom_command(
            TARGET vtaAvgPoolTestBundle POST_BUILD
            COMMAND
            diff output.bin vtaAvgPoolTestGolden.bin)

    add_custom_command(
            TARGET vtaAvgPoolTestBundle POST_BUILD
            COMMAND
            echo "AvgPoolTestBundle Test Succeed")
endif()