add_executable(vtaDequantizeTestBundle vtaDequantizeTestBundleMain.cpp vtaDequantizeTestBundle.cpp)
target_link_libraries(vtaDequantizeTestBundle
         VTABundle  )

set_target_properties(vtaDequantizeTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

if(NESTC_EVTA_BUNDLE_TEST)
    add_custom_command(
            TARGET vtaDequantizeTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaDequantizeTestInput.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaDequantizeTestInput.bin)


    add_custom_command(
            TARGET vtaDequantizeTestBundle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/vtaDequantizeTestGolden.bin
            ${CMAKE_CURRENT_BINARY_DIR}/vtaDequantizeTestGolden.bin)



    add_custom_command(
            TARGET vtaDequantizeTestBundle POST_BUILD
            COMMAND
            vtaDequantizeTestBundle vtaDequantizeTestInput.bin)
    add_custom_command(
            TARGET vtaDequantizeTestBundle POST_BUILD
            COMMAND
            diff output.bin vtaDequantizeTestGolden.bin)
    add_custom_command(
            TARGET vtaDequantizeTestBundle POST_BUILD
            COMMAND
            echo "vtaDequantizeTestBundle Test Succeed")
endif()