add_executable(vtaTransposeTestBundle vtaTransposeTestBundleMain.cpp vtaTransposeTestBundle.cpp)
target_link_libraries(vtaTransposeTestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaTransposeTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_custom_command(
        TARGET vtaTransposeTestBundle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/vtaTransposeTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaTransposeTestBundle.weights.bin)

add_nestc_test(ZCU102 NAME vtaTransposeTestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaTransposeTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_SOURCE_DIR}/vtaTransposeTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/vtaTransposeTestGolden.bin)