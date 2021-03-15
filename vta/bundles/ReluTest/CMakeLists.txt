add_executable(vtaReluTestBundle vtaReluTestBundleMain.cpp vtaReluTestBundle.cpp)
target_link_libraries(vtaReluTestBundle
       VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaReluTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaReluTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaReluTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaReluTestInput.bin)

    add_custom_command(
            TARGET vtaReluTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaReluTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaReluTestBundle.weights.bin)

    add_custom_command(
            TARGET vtaReluTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaReluTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaReluTestGolden.bin)



    add_custom_command(
            TARGET vtaReluTestBundle POST_BUILD
            COMMAND
            vtaReluTestBundle vtaReluTestInput.bin)
    add_custom_command(
            TARGET vtaReluTestBundle POST_BUILD
            COMMAND
            diff output.bin vtaReluTestGolden.bin)
    add_custom_command(
            TARGET vtaReluTestBundle POST_BUILD
            COMMAND
            echo "vtaReluTestBundle Test Succeed")
endif()