INCLUDE_DIRECTORIES(../)
if(NOT NESTC_USE_PRECOMPILED_EVTA_LIBRARY)
    add_subdirectory(vtalib)
endif()

if(NOT NESTC_BNN)
    add_subdirectory(tests)
    if(NOT NESTC_USE_PRECOMPILED_EVTA_LIBRARY)
        add_nestc_test(name VTABackendSimulatorTest ${NESTC_BINARY_DIR}/tests/VTABackendSimulatorTest --gtest_output=xml:VTABackendSimulatorTest.xml)
    endif()
endif()

add_subdirectory(bundles)
if(NESTC_EVTA_MULTI)
    if(NOT NESTC_BNN)
        add_subdirectory(multievta_bundles)
    endif()
endif()

