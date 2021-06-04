add_executable(vtaElemSubTestBundle vtaElemSubTestBundleMain.cpp vtaElemSubTestBundle.cpp)
target_link_libraries(vtaElemSubTestBundle
        VTABundle ${BUNDLE_LINK_LIB})

set_target_properties(vtaElemSubTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_custom_command(
        TARGET vtaElemSubTestBundle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemSubTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaElemSubTestBundle.weights.bin)

add_nestc_test(ZCU102 NAME vtaElemSubTestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaElemSubTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemSubTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemSubTestGolden.bin)