option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.ResConv2Test
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTACResConv2Test/vtaResConv2TestBundle.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestBundle.cpp
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTACResConv2Test/vtaResConv2TestInput.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestInput.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTACResConv2Test/vtaResConv2TestBundle.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestBundle.weights.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTACResConv2Test/vtaResConv2TestGolden.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestGolden.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTACResConv2Test/vtaResConv2TestMainEntry.h -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestMainEntry.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTACResConv2Test/VTARuntime.h -O ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h

    )
endif()
add_custom_target(vtaResConv2TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vtaResConv2TestBundle vtaResConv2TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestBundle.cpp)
add_dependencies(vtaResConv2TestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaResConv2TestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaResConv2TestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaResConv2TestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv2TestGolden.bin)

