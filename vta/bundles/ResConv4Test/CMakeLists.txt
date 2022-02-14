option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.ResConv4Test
            DEPENDS
            VTASaverTest
    )
    add_custom_target(vtaResConv4TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestGolden.bin)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

    add_executable(vtaResConv4TestBundle vtaResConv4TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestBundle.cpp)
    add_dependencies(vtaResConv4TestBundle ${BUNDLE_DEPENDS})
    target_link_libraries(vtaResConv4TestBundle VTABundle ${BUNDLE_LINK_LIB})
    set_target_properties( vtaResConv4TestBundle
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            )

    add_nestc_test(ZCU102 NAME vtaResConv4TestBundle
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestInput.bin
            USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestGolden.bin)

    if(NESTC_EVTA_PROFILE_AUTOTUNE)
        add_executable(vtaResConv4TestBundleProfiling vtaResConv4TestBundleProfilingMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv4TestBundle.cpp)
        add_dependencies(vtaResConv4TestBundleProfiling ${BUNDLE_DEPENDS})
        target_link_libraries(vtaResConv4TestBundleProfiling VTABundle ${BUNDLE_LINK_LIB})
        set_target_properties( vtaResConv4TestBundleProfiling
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                )
    endif()
endif()

