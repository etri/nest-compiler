option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.ResConv5Test
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv5Test/vtaResConv5TestBundle.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestBundle.cpp
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv5Test/vtaResConv5TestInput.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestInput.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv5Test/vtaResConv5TestBundle.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestBundle.weights.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv5Test/vtaResConv5TestGolden.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestGolden.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv5Test/vtaResConv5TestMainEntry.h -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestMainEntry.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv5Test/VTARuntime.h -O ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h

    )
endif()
add_custom_target(vtaResConv5TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vtaResConv5TestBundle vtaResConv5TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestBundle.cpp)
add_dependencies(vtaResConv5TestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaResConv5TestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaResConv5TestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaResConv5TestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv5TestGolden.bin)

