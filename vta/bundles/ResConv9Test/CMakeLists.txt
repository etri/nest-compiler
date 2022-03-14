option(NESTC_USE_PRECOMPILED_BUNDLE "Use pre-compiled bundle" OFF)

if(NOT NESTC_USE_PRECOMPILED_BUNDLE)
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestMainEntry.h
            ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
            COMMAND
            VTASaverTest --gtest_filter=VTASaverTest.ResConv9Test
            DEPENDS
            VTASaverTest
    )
    add_custom_target(vtaResConv9TestBundleNET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestBundle.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestGolden.bin)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

    add_executable(vtaResConv9TestBundle vtaResConv9TestBundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestBundle.cpp)
    add_dependencies(vtaResConv9TestBundle ${BUNDLE_DEPENDS})
    target_link_libraries(vtaResConv9TestBundle VTABundle ${BUNDLE_LINK_LIB})
    set_target_properties( vtaResConv9TestBundle
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            )

    add_nestc_test(ZCU102 NAME vtaResConv9TestBundle
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestInput.bin
            USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestGolden.bin)

    if(NESTC_EVTA_PROFILE_AUTOTUNE)
        add_executable(vtaResConv9TestBundleProfiling vtaResConv9TestBundleProfilingMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestBundle.cpp)
        add_dependencies(vtaResConv9TestBundleProfiling ${BUNDLE_DEPENDS})
        target_link_libraries(vtaResConv9TestBundleProfiling VTABundle ${BUNDLE_LINK_LIB})
        set_target_properties( vtaResConv9TestBundleProfiling
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                )
    endif()
else()
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestBundle.cpp
            COMMAND
            aws s3 cp s3://nestc-pub/vta/bundles/ResConv9Test/ ${CMAKE_CURRENT_BINARY_DIR}/ --no-sign-request --recursive
    )
    if(NESTC_EVTA_PROFILE_AUTOTUNE)
        INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
        add_executable(vtaResConv9TestBundleProfiling vtaResConv9TestBundleProfilingMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/vtaResConv9TestBundle.cpp)
        add_dependencies(vtaResConv9TestBundleProfiling ${BUNDLE_DEPENDS})
        target_link_libraries(vtaResConv9TestBundleProfiling VTABundle ${BUNDLE_LINK_LIB})
        set_target_properties( vtaResConv9TestBundleProfiling
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                )
    endif()
endif()
