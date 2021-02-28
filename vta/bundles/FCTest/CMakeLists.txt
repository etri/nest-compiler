add_executable(vtaFCTestBundle vtaFCTestBundleMain.cpp vtaFCTestBundle.cpp)
target_link_libraries(vtaFCTestBundle VTABundle )
set_target_properties(vtaFCTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaFCTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaFCTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFCTestInput.bin)

    add_custom_command(
            TARGET vtaFCTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaFCTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFCTestBundle.weights.bin)

    add_custom_command(
            TARGET vtaFCTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaFCTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaFCTestGolden.bin)



    add_custom_command(
            TARGET vtaFCTestBundle POST_BUILD
            COMMAND
            vtaFCTestBundle vtaFCTestInput.bin)
    add_custom_command(
            TARGET vtaFCTestBundle POST_BUILD
            COMMAND
            diff output.bin vtaFCTestGolden.bin)
    add_custom_command(
            TARGET vtaFCTestBundle POST_BUILD
            COMMAND
            echo "vtaFCTestBundle Test Succeed")
endif()