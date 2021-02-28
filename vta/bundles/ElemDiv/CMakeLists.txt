add_executable(vtaElemDivTestBundle vtaElemDivTestBundleMain.cpp vtaElemDivTestBundle.cpp)
target_link_libraries(vtaElemDivTestBundle
        VTABundle )

set_target_properties(vtaElemDivTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaElemDivTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemDivTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaElemDivTestInput.bin)

    add_custom_command(
            TARGET vtaElemDivTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemDivTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaElemDivTestBundle.weights.bin)

    add_custom_command(
            TARGET vtaElemDivTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemDivTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaElemDivTestGolden.bin)



    add_custom_command(
            TARGET vtaElemDivTestBundle POST_BUILD
            COMMAND
            vtaElemDivTestBundle vtaElemDivTestInput.bin)
    add_custom_command(
            TARGET vtaElemDivTestBundle POST_BUILD
            COMMAND
            diff output.bin vtaElemDivTestGolden.bin)
    add_custom_command(
            TARGET vtaElemDivTestBundle POST_BUILD
            COMMAND
            echo "vtaElemDivTestBundle Test Succeed")
endif()