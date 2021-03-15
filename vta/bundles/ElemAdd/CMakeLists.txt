add_executable(vtaElemAddTestBundle vtaElemAddTestBundleMain.cpp vtaElemAddTestBundle.cpp)
target_link_libraries(vtaElemAddTestBundle
        VTABundle ${BUNDLE_LINK_LIB})

set_target_properties(vtaElemAddTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaElemAddTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemAddTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaElemAddTestInput.bin)

    add_custom_command(
            TARGET vtaElemAddTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemAddTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaElemAddTestBundle.weights.bin)

    add_custom_command(
            TARGET vtaElemAddTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemAddTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaElemAddTestGolden.bin)



    add_custom_command(
            TARGET vtaElemAddTestBundle POST_BUILD
            COMMAND
            vtaElemAddTestBundle vtaElemAddTestInput.bin)
    add_custom_command(
            TARGET vtaElemAddTestBundle POST_BUILD
            COMMAND
            diff output.bin vtaElemAddTestGolden.bin)
    add_custom_command(
            TARGET vtaElemAddTestBundle POST_BUILD
            COMMAND
            echo "vtaElemAddTestBundle Test Succeed")
endif()