add_executable(vtaQuantizeTestBundle vtaQuantizeTestBundleMain.cpp vtaQuantizeTestBundle.cpp)
target_link_libraries(vtaQuantizeTestBundle VTABundle  )
set_target_properties(vtaQuantizeTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaQuantizeTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaQuantizeTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaQuantizeTestInput.bin)


    add_custom_command(
            TARGET vtaQuantizeTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaQuantizeTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaQuantizeTestGolden.bin)



    add_custom_command(
            TARGET vtaQuantizeTestBundle POST_BUILD
            COMMAND
            vtaQuantizeTestBundle vtaQuantizeTestInput.bin)
    add_custom_command(
            TARGET vtaQuantizeTestBundle POST_BUILD
            COMMAND
            diff output.bin vtaQuantizeTestGolden.bin)
    add_custom_command(
            TARGET vtaQuantizeTestBundle POST_BUILD
            COMMAND
            echo "vtaQuantizeTestBundle Test Succeed")
endif()