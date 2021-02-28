INCLUDE_DIRECTORIES(../)
add_subdirectory(vtalib)
add_subdirectory(tests)
add_glow_test(VTABackendSimulatorTest ${GLOW_BINARY_DIR}/tests/VTABackendSimulatorTest --gtest_output=xml:VTABackendSimulatorTest.xml)
add_subdirectory(bundles)

