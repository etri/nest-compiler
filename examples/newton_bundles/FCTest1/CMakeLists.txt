add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest1Bundle.cpp
        COMMAND
        NewtonSaverTest --gtest_filter=NewtonSaverTest.fcTest1
        DEPENDS
        NewtonSaverTest
)


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(newtonFCTest1Bundle newtonFCTest1BundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest1Bundle.cpp)

target_link_libraries(newtonFCTest1Bundle ${NESTC_ROOT_DIR}/external_libs/newton/aimddk/libaim/lib/libaim.so)
set_target_properties(newtonFCTest1Bundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )


add_nestc_test(NEWTON NAME newtonFCTest1Bundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest1Bundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest1Input.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest1Golden.bin)
