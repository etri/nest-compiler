add_executable(vtaFCTestBundle vtaFCTestBundleMain.cpp vtaFCTestBundle.cpp)
add_dependencies(vtaFCTestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaFCTestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaFCTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_custom_command(
        TARGET vtaFCTestBundle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/vtaFCTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaFCTestBundle.weights.bin)

add_nestc_test(ZCU102 NAME vtaFCTestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaFCTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_SOURCE_DIR}/vtaFCTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/vtaFCTestGolden.bin)