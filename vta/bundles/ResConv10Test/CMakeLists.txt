option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.ResConv10Test
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv10Test/vtaResConv10TestBundle.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestBundle.cpp
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv10Test/vtaResConv10TestInput.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestInput.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv10Test/vtaResConv10TestBundle.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestBundle.weights.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv10Test/vtaResConv10TestGolden.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestGolden.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv10Test/vtaResConv10TestMainEntry.h -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestMainEntry.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/VTAResConv10Test/VTARuntime.h -O ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h

    )
endif()
add_custom_target(vtaResConv10TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vtaResConv10TestBundle vtaResConv10TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestBundle.cpp)
add_dependencies(vtaResConv10TestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaResConv10TestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaResConv10TestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaResConv10TestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv10TestGolden.bin)

