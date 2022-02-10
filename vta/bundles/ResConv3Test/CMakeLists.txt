option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.ResConv3Test
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv3Test/vtaResConv3TestBundle.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestBundle.cpp
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv3Test/vtaResConv3TestInput.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestInput.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv3Test/vtaResConv3TestBundle.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestBundle.weights.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv3Test/vtaResConv3TestGolden.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestGolden.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv3Test/vtaResConv3TestMainEntry.h -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestMainEntry.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv3Test/VTARuntime.h -O ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h

    )
endif()
add_custom_target(vtaResConv3BundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vtaResConv3TestBundle vtaResConv3TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestBundle.cpp)
add_dependencies(vtaResConv3TestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaResConv3TestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaResConv3TestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaResConv3TestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestGolden.bin)

