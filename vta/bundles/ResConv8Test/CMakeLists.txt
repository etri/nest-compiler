
if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.ResConv8Test
            DEPENDS
            VTASaverTest
    )
    add_custom_target(vtaResConv8TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestGolden.bin)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

    add_executable(vtaResConv8TestBundle vtaResConv8TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestBundle.cpp)
    add_dependencies(vtaResConv8TestBundle ${BUNDLE_DEPENDS})
    target_link_libraries(vtaResConv8TestBundle VTABundle ${BUNDLE_LINK_LIB})
    set_target_properties( vtaResConv8TestBundle
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            )

    add_nestc_test(ZCU102 NAME vtaResConv8TestBundle
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestInput.bin
            USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestGolden.bin)

    if(NESTC_EVTA_PROFILE_AUTOTUNE)
        add_executable(vtaResConv8TestBundleProfiling vtaResConv8TestBundleProfilingMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestBundle.cpp)
        add_dependencies(vtaResConv8TestBundleProfiling ${BUNDLE_DEPENDS})
        target_link_libraries(vtaResConv8TestBundleProfiling VTABundle ${BUNDLE_LINK_LIB})
        set_target_properties( vtaResConv8TestBundleProfiling
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                )
    endif()
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestBundle.cpp
            COMMAND
            aws s3 cp s3://nestc-pub/vta/bundles/ResConv8Test/ ${CMAKE_CURRENT_BINARY_DIR}/ --no-sign-request --recursive
    )
    if(NESTC_EVTA_PROFILE_AUTOTUNE)
        INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
        add_executable(vtaResConv8TestBundleProfiling vtaResConv8TestBundleProfilingMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv8TestBundle.cpp)
        add_dependencies(vtaResConv8TestBundleProfiling ${BUNDLE_DEPENDS})
        target_link_libraries(vtaResConv8TestBundleProfiling VTABundle ${BUNDLE_LINK_LIB})
        set_target_properties( vtaResConv8TestBundleProfiling
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                )
    endif()
endif()

