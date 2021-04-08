add_executable(vtaTransposeTestBundle vtaTransposeTestBundleMain.cpp vtaTransposeTestBundle.cpp)
target_link_libraries(vtaTransposeTestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaTransposeTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaTransposeTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaTransposeTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaTransposeTestInput.bin)

    add_custom_command(
            TARGET vtaTransposeTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaTransposeTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaTransposeTestBundle.weights.bin)

    add_custom_command(
            TARGET vtaTransposeTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaTransposeTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaTransposeTestGolden.bin)



    add_custom_command(
            TARGET vtaTransposeTestBundle POST_BUILD
            COMMAND
            vtaTransposeTestBundle vtaTransposeTestInput.bin)
    add_custom_command(
            TARGET vtaTransposeTestBundle POST_BUILD
            COMMAND
            diff output.bin vtaTransposeTestGolden.bin)
    add_custom_command(
            TARGET vtaTransposeTestBundle POST_BUILD
            COMMAND
            echo "vtaTransposeTestBundle Test Succeed")
endif()