add_executable(vtaMaxPoolTestBundle vtaMaxPoolTestBundleMain.cpp vtaMaxPoolTestBundle.cpp)
target_link_libraries(vtaMaxPoolTestBundle
       VTABundle )
set_target_properties(vtaMaxPoolTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaMaxPoolTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaMaxPoolTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaMaxPoolTestInput.bin)

    add_custom_command(
            TARGET vtaMaxPoolTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaMaxPoolTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaMaxPoolTestBundle.weights.bin)

    add_custom_command(
            TARGET vtaMaxPoolTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaMaxPoolTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaMaxPoolTestGolden.bin)



    add_custom_command(
            TARGET vtaMaxPoolTestBundle POST_BUILD
            COMMAND
            vtaMaxPoolTestBundle vtaMaxPoolTestInput.bin)
    add_custom_command(
            TARGET vtaMaxPoolTestBundle POST_BUILD
            COMMAND
            diff output.bin vtaMaxPoolTestGolden.bin)
    add_custom_command(
            TARGET vtaMaxPoolTestBundle POST_BUILD
            COMMAND
            echo "vtaMaxPoolTestBundle Test Succeed")
endif()