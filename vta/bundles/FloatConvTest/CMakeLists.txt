if(GLOW_WITH_VTASIM)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestMainEntry.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.conv_gpu_0_conv1
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestGolden.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/FloatConvTest/vtaFloatConvTestBundle.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.cpp
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/FloatConvTest/vtaFloatConvTestInput.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestInput.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/FloatConvTest/vtaFloatConvTestBundle.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.weights.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/FloatConvTest/vtaFloatConvTestGolden.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestGolden.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/FloatConvTest/vtaFloatConvTestMainEntry.h -O ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestMainEntry.h

            #            COMMAND
            #            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/FloatConvTest/ -O ${CMAKE_CURRENT_BINARY_DIR}/
    )
endif()
add_custom_target(vtaFloatConvTestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestGolden.bin)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaFloatConvTestBundle vtaFloatConvTestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaFloatConvTestBundle.cpp)
add_dependencies(vtaFloatConvTestBundle vtaFloatConvTestBundleNET)
target_link_libraries(vtaFloatConvTestBundle VTABundle )
set_target_properties(vtaFloatConvTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaFloatConvTestBundle POST_BUILD
            COMMAND
            vtaFloatConvTestBundle vtaFloatConvTestInput.bin)
    add_custom_command(
            TARGET vtaFloatConvTestBundle POST_BUILD
            COMMAND
            echo "vtaFloatConvTestBundle Test Succeed")
endif()