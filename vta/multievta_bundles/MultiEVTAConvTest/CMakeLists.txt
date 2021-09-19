option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle0.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle1.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle2.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle3.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestInput0.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestInput1.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestInput2.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestInput3.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle0.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle1.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle2.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle3.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestGolden0.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestGolden1.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestGolden2.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestGolden3.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestMainEntry0.h
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestMainEntry1.h
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestMainEntry2.h
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestMainEntry3.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.multievtaConvTest
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/MultiEVTAConvTest.tar
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/MultiEVTAConvTest/MultiEVTAConvTest.tar -O ${CMAKE_CURRENT_BINARY_DIR}/MultiEVTAConvTest.tar
    )
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle0.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle1.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle2.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle3.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestInput0.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestInput1.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestInput2.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestInput3.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle0.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle1.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle2.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle3.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestGolden0.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestGolden1.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestGolden2.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestGolden3.bin
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestMainEntry0.h
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestMainEntry1.h
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestMainEntry2.h
            ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestMainEntry3.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            ${CMAKE_COMMAND} -E tar "xvf" ${CMAKE_CURRENT_BINARY_DIR}/MultiEVTAConvTest.tar
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/MultiEVTAConvTest.tar
    )
endif()
add_custom_target(multievtaConvTestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(multievtaConvTestBundle multievtaConvTestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle0.cpp ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle1.cpp ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle2.cpp ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle3.cpp)
add_dependencies(multievtaConvTestBundle ${BUNDLE_DEPENDS})
target_link_libraries(multievtaConvTestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( multievtaConvTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 MULTIEVTA NAME multievtaConvTestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/multievtaConvTestGolden0.bin)