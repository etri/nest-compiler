option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.convTest5
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv5Test/vtaConv5TestBundle.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestBundle.cpp
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv5Test/vtaConv5TestInput.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestInput.bin --no-verbose
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv5Test/vtaConv5TestBundle.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestBundle.weights.bin --no-verbose
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv5Test/vtaConv5TestGolden.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestGolden.bin --no-verbose
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv5Test/vtaConv5TestMainEntry.h -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestMainEntry.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv5Test/VTARuntime.h -O ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h

    )
endif()
add_custom_target(vtaConv5TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vtaConv5TestBundle vtaConv5TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestBundle.cpp)
add_dependencies(vtaConv5TestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaConv5TestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaConv5TestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaConv5TestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaConv5TestGolden.bin)