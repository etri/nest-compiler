if(GLOW_WITH_VTASIM)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestMainEntry.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.convTest
            DEPENDS
            VTASaverTest
    )

    add_custom_target(vtaConvTestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestGolden.bin)

    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

    add_executable(vtaConvTestBundle vtaConvTestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestBundle.cpp)

    target_link_libraries(vtaConvTestBundle VTABundle ${BUNDLE_LINK_LIB})
    set_target_properties( vtaConvTestBundle
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            )
    if(NESTC_EVTA_BUNDLE_TEST)
        add_custom_command(
                TARGET vtaConvTestBundle POST_BUILD
                COMMAND
                vtaConvTestBundle vtaConvTestInput.bin)

        add_custom_command(
                TARGET vtaConvTestBundle POST_BUILD
                COMMAND
                diff output.bin vtaConvTestGolden.bin)

        add_custom_command(
                TARGET vtaConvTestBundle POST_BUILD
                COMMAND
                echo "AvgPoolTestBundle Test Succeed")
    endif()
endif()
