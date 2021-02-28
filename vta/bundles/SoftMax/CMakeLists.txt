add_executable(vtaSoftMaxTestBundle vtaSoftMaxTestBundleMain.cpp vtaSoftMaxTestBundle.cpp)
target_link_libraries(vtaSoftMaxTestBundle VTABundle )
set_target_properties(vtaSoftMaxTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaSoftMaxTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaSoftMaxTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaSoftMaxTestInput.bin)

    add_custom_command(
            TARGET vtaSoftMaxTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaSoftMaxTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaSoftMaxTestBundle.weights.bin)

    add_custom_command(
            TARGET vtaSoftMaxTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaSoftMaxTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaSoftMaxTestGolden.bin)



    add_custom_command(
            TARGET vtaSoftMaxTestBundle POST_BUILD
            COMMAND
            vtaSoftMaxTestBundle vtaSoftMaxTestInput.bin)
    add_custom_command(
            TARGET vtaSoftMaxTestBundle POST_BUILD
            COMMAND
            diff output.bin vtaSoftMaxTestGolden.bin)
    add_custom_command(
            TARGET vtaSoftMaxTestBundle POST_BUILD
            COMMAND
            echo "vtaSoftMaxTestBundle Test Succeed")
endif()