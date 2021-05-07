option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)
option(NESTC_EVTA_BUNDLE_TEST "Test bundles after build executable" ON)
if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.convTest2
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv2Test/vtaConv2TestBundle.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.cpp
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv2Test/vtaConv2TestInput.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestInput.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv2Test/vtaConv2TestBundle.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.weights.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv2Test/vtaConv2TestGolden.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestGolden.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv2Test/vtaConv2TestMainEntry.h -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestMainEntry.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv2Test/VTARuntime.h -O ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h

    )
endif()
add_custom_target(vtaConv2TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vtaConv2TestBundle vtaConv2TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaConv2TestBundle.cpp)

target_link_libraries(vtaConv2TestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaConv2TestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaConv2TestBundle POST_BUILD
            COMMAND
            vtaConv2TestBundle vtaConv2TestInput.bin)

    add_custom_command(
            TARGET vtaConv2TestBundle POST_BUILD
            COMMAND
            diff output.bin vtaConv2TestGolden.bin)

    add_custom_command(
            TARGET vtaConv2TestBundle POST_BUILD
            COMMAND
            echo "vtaConv2TestBundle Test Succeed")
endif()
