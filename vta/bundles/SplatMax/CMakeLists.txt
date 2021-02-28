add_executable(vtaSplatMaxTestBundle vtaSplatMaxTestBundleMain.cpp vtaSplatMaxTestBundle.cpp)
target_link_libraries(vtaSplatMaxTestBundle VTABundle )
set_target_properties(vtaSplatMaxTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaSplatMaxTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaSplatMaxTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaSplatMaxTestInput.bin)

    add_custom_command(
            TARGET vtaSplatMaxTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaSplatMaxTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaSplatMaxTestBundle.weights.bin)

    add_custom_command(
            TARGET vtaSplatMaxTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaSplatMaxTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaSplatMaxTestGolden.bin)



    add_custom_command(
            TARGET vtaSplatMaxTestBundle POST_BUILD
            COMMAND
            vtaSplatMaxTestBundle vtaSplatMaxTestInput.bin)
    add_custom_command(
            TARGET vtaSplatMaxTestBundle POST_BUILD
            COMMAND
            diff output.bin vtaSplatMaxTestGolden.bin)
    add_custom_command(
            TARGET vtaSplatMaxTestBundle POST_BUILD
            COMMAND
            echo "vtaSplatMaxTestBundle Test Succeed")
endif()