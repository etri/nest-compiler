add_executable(vtaElemSubTestBundle vtaElemSubTestBundleMain.cpp vtaElemSubTestBundle.cpp)
target_link_libraries(vtaElemSubTestBundle
        VTABundle )

set_target_properties(vtaElemSubTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaElemSubTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemSubTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaElemSubTestInput.bin)

    add_custom_command(
            TARGET vtaElemSubTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemSubTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaElemSubTestBundle.weights.bin)

    add_custom_command(
            TARGET vtaElemSubTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemSubTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaElemSubTestGolden.bin)



    add_custom_command(
            TARGET vtaElemSubTestBundle POST_BUILD
            COMMAND
            vtaElemSubTestBundle vtaElemSubTestInput.bin)
    add_custom_command(
            TARGET vtaElemSubTestBundle POST_BUILD
            COMMAND
            diff output.bin vtaElemSubTestGolden.bin)
    add_custom_command(
            TARGET vtaElemSubTestBundle POST_BUILD
            COMMAND
            echo "vtaElemSubTestBundle Test Succeed")
endif()