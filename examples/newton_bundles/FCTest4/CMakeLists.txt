add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest4Bundle.cpp
        COMMAND
        NewtonSaverTest --gtest_filter=NewtonSaverTest.fcTest4
        DEPENDS
        NewtonSaverTest
)


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(newtonFCTest4Bundle newtonFCTest4BundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest4Bundle.cpp)

target_link_libraries(newtonFCTest4Bundle ${NESTC_ROOT_DIR}/external_libs/newton/aimddk/libaim/lib/libaim.so)
set_target_properties(newtonFCTest4Bundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(NEWTON NAME newtonFCTest4Bundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest4Bundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest4Input.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest4Golden.bin)
