add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest2Bundle.cpp
        COMMAND
        NewtonSaverTest --gtest_filter=NewtonSaverTest.fcTest2
        DEPENDS
        NewtonSaverTest
)


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(newtonFCTest2Bundle newtonFCTest2BundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest2Bundle.cpp)

target_link_libraries(newtonFCTest2Bundle ${NESTC_ROOT_DIR}/external_libs/newton/aimddk/libaim/lib/libaim.so)
set_target_properties(newtonFCTest2Bundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(NEWTON NAME newtonFCTest2Bundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest2Bundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest2Input.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/newtonFCTest2Golden.bin)
