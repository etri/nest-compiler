add_executable(vtaReluTestBundle vtaReluTestBundleMain.cpp vtaReluTestBundle.cpp)
target_link_libraries(vtaReluTestBundle
       VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaReluTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_custom_command(
        TARGET vtaReluTestBundle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/vtaReluTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaReluTestBundle.weights.bin)


add_nestc_test(ZCU102 NAME vtaReluTestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaReluTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_SOURCE_DIR}/vtaReluTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/vtaReluTestGolden.bin)