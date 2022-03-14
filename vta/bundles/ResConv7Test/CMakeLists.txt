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

    if(NESTC_EVTA_PROFILE_AUTOTUNE)
        add_executable(vtaResConv7TestBundleProfiling vtaResConv7TestBundleProfilingMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestBundle.cpp)
        add_dependencies(vtaResConv7TestBundleProfiling ${BUNDLE_DEPENDS})
        target_link_libraries(vtaResConv7TestBundleProfiling VTABundle ${BUNDLE_LINK_LIB})
        set_target_properties( vtaResConv7TestBundleProfiling
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                )
    endif()
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestBundle.cpp
            COMMAND
            aws s3 cp s3://nestc-pub/vta/bundles/ResConv7Test/ ${CMAKE_CURRENT_BINARY_DIR}/ --no-sign-request --recursive
    )
    if(NESTC_EVTA_PROFILE_AUTOTUNE)
        INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
        add_executable(vtaResConv7TestBundleProfiling vtaResConv7TestBundleProfilingMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv7TestBundle.cpp)
        add_dependencies(vtaResConv7TestBundleProfiling ${BUNDLE_DEPENDS})
        target_link_libraries(vtaResConv7TestBundleProfiling VTABundle ${BUNDLE_LINK_LIB})
        set_target_properties( vtaResConv7TestBundleProfiling
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                )
    endif()
endif()

