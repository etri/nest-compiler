option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.ResConv6Test
            DEPENDS
            VTASaverTest
    )
    add_custom_target(vtaResConv6TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestGolden.bin)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

    add_executable(vtaResConv6TestBundle vtaResConv6TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestBundle.cpp)
    add_dependencies(vtaResConv6TestBundle ${BUNDLE_DEPENDS})
    target_link_libraries(vtaResConv6TestBundle VTABundle ${BUNDLE_LINK_LIB})
    set_target_properties( vtaResConv6TestBundle
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            )

    add_nestc_test(ZCU102 NAME vtaResConv6TestBundle
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestInput.bin
            USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestGolden.bin)

    if(NESTC_EVTA_PROFILE_AUTOTUNE)
        add_executable(vtaResConv6TestBundleProfiling vtaResConv6TestBundleProfilingMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestBundle.cpp)
        add_dependencies(vtaResConv6TestBundleProfiling ${BUNDLE_DEPENDS})
        target_link_libraries(vtaResConv6TestBundleProfiling VTABundle ${BUNDLE_LINK_LIB})
        set_target_properties( vtaResConv6TestBundleProfiling
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                )
    endif()
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestBundle.cpp
            COMMAND
            aws s3 cp s3://nestc-pub/vta/bundles/ResConv6Test/ ${CMAKE_CURRENT_BINARY_DIR}/ --no-sign-request --recursive
    )
    if(NESTC_EVTA_PROFILE_AUTOTUNE)
        INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
        add_executable(vtaResConv6TestBundleProfiling vtaResConv6TestBundleProfilingMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv6TestBundle.cpp)
        add_dependencies(vtaResConv6TestBundleProfiling ${BUNDLE_DEPENDS})
        target_link_libraries(vtaResConv6TestBundleProfiling VTABundle ${BUNDLE_LINK_LIB})
        set_target_properties( vtaResConv6TestBundleProfiling
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                )
    endif()
endif()

