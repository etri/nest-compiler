option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)
option(NESTC_EVTA_BUNDLE_TEST "Test bundles after build executable" ON)
if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.convTest3
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv3Test/vtaConv3TestBundle.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.cpp
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv3Test/vtaConv3TestInput.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestInput.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv3Test/vtaConv3TestBundle.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.weights.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv3Test/vtaConv3TestGolden.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestGolden.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv3Test/vtaConv3TestMainEntry.h -O ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestMainEntry.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAConv3Test/VTARuntime.h -O ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h

    )
endif()
add_custom_target(vtaConv3TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vtaConv3TestBundle vtaConv3TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaConv3TestBundle.cpp)

target_link_libraries(vtaConv3TestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaConv3TestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaConv3TestBundle POST_BUILD
            COMMAND
            vtaConv3TestBundle vtaConv3TestInput.bin)

    add_custom_command(
            TARGET vtaConv3TestBundle POST_BUILD
            COMMAND
            diff output.bin vtaConv3TestGolden.bin)

    add_custom_command(
            TARGET vtaConv3TestBundle POST_BUILD
            COMMAND
            echo "vtaConv3TestBundle Test Succeed")
endif()
