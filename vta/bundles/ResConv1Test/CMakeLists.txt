
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

    if(NESTC_EVTA_PROFILE_AUTOTUNE)
        add_executable(vtaResConv1TestBundleProfiling vtaResConv1TestBundleProfilingMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestBundle.cpp)
        add_dependencies(vtaResConv1TestBundleProfiling ${BUNDLE_DEPENDS})
        target_link_libraries(vtaResConv1TestBundleProfiling VTABundle ${BUNDLE_LINK_LIB})
        set_target_properties( vtaResConv1TestBundleProfiling
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                )
    endif()
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestBundle.cpp
            COMMAND
            aws s3 cp s3://nestc-pub/vta/bundles/ResConv1Test/ ${CMAKE_CURRENT_BINARY_DIR}/ --no-sign-request --recursive
    )
    if(NESTC_EVTA_PROFILE_AUTOTUNE)
        INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
        add_executable(vtaResConv1TestBundleProfiling vtaResConv1TestBundleProfilingMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv1TestBundle.cpp)
        add_dependencies(vtaResConv1TestBundleProfiling ${BUNDLE_DEPENDS})
        target_link_libraries(vtaResConv1TestBundleProfiling VTABundle ${BUNDLE_LINK_LIB})
        set_target_properties( vtaResConv1TestBundleProfiling
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                )
    endif()
endif()


