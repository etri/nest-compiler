if(NESTC_USE_VTASIM)
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

    add_nestc_test(NAME vtaConvTestBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestInput.bin
            USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestGolden.bin)
endif()
