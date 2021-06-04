option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.resnetv10S1C0Test
            DEPENDS
            VTASaverTest
    )
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnetv10S1C0Test/vtaResnetv10S1C0TestBundle.cpp -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestBundle.cpp
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnetv10S1C0Test/vtaResnetv10S1C0TestInput.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestInput.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnetv10S1C0Test/vtaResnetv10S1C0TestBundle.weights.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestBundle.weights.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnetv10S1C0Test/vtaResnetv10S1C0TestGolden.bin -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestGolden.bin
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnetv10S1C0Test/vtaResnetv10S1C0TestMainEntry.h -O ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestMainEntry.h
            COMMAND
            wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/VTABundleTests/Resnetv10S1C0Test/VTARuntime.h -O ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h

    )
endif()
add_custom_target(vtaResnetv10S1C0TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestBundle.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestInput.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestGolden.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(vtaResnetv10S1C0TestBundle vtaResnetv10S1C0TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestBundle.cpp)

target_link_libraries(vtaResnetv10S1C0TestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties( vtaResnetv10S1C0TestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaResnetv10S1C0TestBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaResnetv10S1C0TestInput.bin)