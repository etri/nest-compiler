option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.ResConv7Test
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv7Test/vtaResConv7TestBundle.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestBundle.cpp
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv7Test/vtaResConv7TestInput.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestInput.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv7Test/vtaResConv7TestBundle.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestBundle.weights.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv7Test/vtaResConv7TestGolden.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestGolden.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv7Test/vtaResConv7TestMainEntry.h -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestMainEntry.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv7Test/VTARuntime.h -O ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h

    )
endif()
add_custom_target(vtaResConv7TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vtaResConv7TestBundle vtaResConv7TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestBundle.cpp)
add_dependencies(vtaResConv7TestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaResConv7TestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaResConv7TestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaResConv7TestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestGolden.bin)

