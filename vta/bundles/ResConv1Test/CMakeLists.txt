option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.ResConv1Test
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv1Test/vtaResConv1TestBundle.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestBundle.cpp
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv1Test/vtaResConv1TestInput.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestInput.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv1Test/vtaResConv1TestBundle.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestBundle.weights.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv1Test/vtaResConv1TestGolden.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestGolden.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv1Test/vtaResConv1TestMainEntry.h -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestMainEntry.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv1Test/VTARuntime.h -O ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h

    )
endif()
add_custom_target(vtaResConv1TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vtaResConv1TestBundle vtaResConv1TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestBundle.cpp)
add_dependencies(vtaResConv1TestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaResConv1TestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaResConv1TestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaResConv1TestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestGolden.bin)

