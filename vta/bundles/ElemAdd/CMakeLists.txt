add_executable(vtaElemAddTestBundle vtaElemAddTestBundleMain.cpp vtaElemAddTestBundle.cpp)
target_link_libraries(vtaElemAddTestBundle
        VTABundle ${BUNDLE_LINK_LIB})

set_target_properties(vtaElemAddTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_custom_command(
        TARGET vtaElemAddTestBundle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemAddTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaElemAddTestBundle.weights.bin)

add_nestc_test(ZCU102 NAME vtaElemAddTestBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaElemAddTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemAddTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/vtaElemAddTestGolden.bin)