add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest2_3Bundle.cpp
        COMMAND
        NewtonSaverTest --gtest_filter=NewtonSaverTest.fcTest2_3
        DEPENDS
        NewtonSaverTest
)


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(newtonFCTest2_3Bundle newtonFCTest2_3BundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest2_3Bundle.cpp)

target_link_libraries(newtonFCTest2_3Bundle ${NESTC_ROOT_DIR}/external_libs/newton/aimddk/libaim/lib/libaim.so)
set_target_properties(newtonFCTest2_3Bundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(NEWTON NAME newtonFCTest2_3Bundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest2_3Bundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest2_3Input.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest2_3Golden.bin)
