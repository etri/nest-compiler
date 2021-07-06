add_executable(vtaElemDivTestBundle vtaElemDivTestBundleMain.cpp vtaElemDivTestBundle.cpp)
add_dependencies(vtaElemDivTestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaElemDivTestBundle
        VTABundle ${BUNDLE_LINK_LIB})

set_target_properties(vtaElemDivTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_custom_command(
        TARGET vtaElemDivTestBundle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemDivTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaElemDivTestBundle.weights.bin)

add_nestc_test(ZCU102 NAME vtaElemDivTestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaElemDivTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemDivTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemDivTestGolden.bin)