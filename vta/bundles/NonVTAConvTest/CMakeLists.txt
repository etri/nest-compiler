add_executable(vtaNonVTAConvTestBundle vtaNonVTAConvTestBundleMain.cpp vtaNonVTAConvTestBundle.cpp)
target_link_libraries(vtaNonVTAConvTestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaNonVTAConvTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaNonVTAConvTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaNonVTAConvTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaNonVTAConvTestInput.bin)

    add_custom_command(
            TARGET vtaNonVTAConvTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaNonVTAConvTestBundle.weights.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaNonVTAConvTestBundle.weights.bin)

    add_custom_command(
            TARGET vtaNonVTAConvTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaNonVTAConvTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaNonVTAConvTestGolden.bin)



    add_custom_command(
            TARGET vtaNonVTAConvTestBundle POST_BUILD
            COMMAND
            vtaNonVTAConvTestBundle vtaNonVTAConvTestInput.bin)
    add_custom_command(
            TARGET vtaNonVTAConvTestBundle POST_BUILD
            COMMAND
            diff output.bin vtaNonVTAConvTestGolden.bin)
    add_custom_command(
            TARGET vtaNonVTAConvTestBundle POST_BUILD
            COMMAND
            echo "vtaNonVTAConvTestBundle Test Succeed")
endif()