INCLUDE_DIRECTORIES(../)
add_subdirectory(vtalib)
add_subdirectory(tests)
add_test(VTABackendSimulatorTest ${NESTC_BINARY_DIR}/tests/VTABackendSimulatorTest --gtest_output=xml:VTABackendSimulatorTest.xml)
add_subdirectory(bundles)

