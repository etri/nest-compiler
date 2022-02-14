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

    if(NESTC_EVTA_PROFILE_AUTOTUNE)
        add_executable(vtaResConv3TestBundleProfiling vtaResConv3TestBundleProfilingMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv3TestBundle.cpp)
        add_dependencies(vtaResConv3TestBundleProfiling ${BUNDLE_DEPENDS})
        target_link_libraries(vtaResConv3TestBundleProfiling VTABundle ${BUNDLE_LINK_LIB})
        set_target_properties( vtaResConv3TestBundleProfiling
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                )
    endif()
endif()
