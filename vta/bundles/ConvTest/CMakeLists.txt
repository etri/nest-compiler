add_executable(vtaConvTestBundle vtaConvTestBundleMain.cpp vtaConvTestBundle.cpp)
target_link_libraries(vtaConvTestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaConvTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaConvTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaConvTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestInput.bin)

    add_custom_command(
            TARGET vtaConvTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaConvTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestBundle.weights.bin)

    add_custom_command(
            TARGET vtaConvTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaConvTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaConvTestGolden.bin)

    add_custom_command(
            TARGET vtaConvTestBundle POST_BUILD
            COMMAND
            ls)


    add_custom_command(
        TARGET vtaConvTestBundle POST_BUILD
        COMMAND
        vtaConvTestBundle vtaConvTestInput.bin)
    add_custom_command(
        TARGET vtaConvTestBundle POST_BUILD
        COMMAND
        diff output.bin vtaConvTestGolden.bin)
    add_custom_command(
            TARGET vtaConvTestBundle POST_BUILD
            COMMAND
            echo "vtaConvTestBundle Test Succeed")
endif()