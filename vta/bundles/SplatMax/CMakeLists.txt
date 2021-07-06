add_executable(vtaSplatMaxTestBundle vtaSplatMaxTestBundleMain.cpp vtaSplatMaxTestBundle.cpp)
add_dependencies(vtaSplatMaxTestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaSplatMaxTestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaSplatMaxTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_custom_command(
        TARGET vtaSplatMaxTestBundle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/vtaSplatMaxTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaSplatMaxTestBundle.weights.bin)

add_nestc_test(ZCU102 NAME vtaSplatMaxTestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaSplatMaxTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_SOURCE_DIR}/vtaSplatMaxTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/vtaSplatMaxTestGolden.bin)