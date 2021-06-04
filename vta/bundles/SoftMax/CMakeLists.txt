add_executable(vtaSoftMaxTestBundle vtaSoftMaxTestBundleMain.cpp vtaSoftMaxTestBundle.cpp)
target_link_libraries(vtaSoftMaxTestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaSoftMaxTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_custom_command(
        TARGET vtaSoftMaxTestBundle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/vtaSoftMaxTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaSoftMaxTestBundle.weights.bin)

add_nestc_test(ZCU102 NAME vtaSoftMaxTestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaSoftMaxTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_SOURCE_DIR}/vtaSoftMaxTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/vtaSoftMaxTestGolden.bin)