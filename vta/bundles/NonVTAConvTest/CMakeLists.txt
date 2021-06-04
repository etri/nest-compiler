add_executable(vtaNonVTAConvTestBundle vtaNonVTAConvTestBundleMain.cpp vtaNonVTAConvTestBundle.cpp)
target_link_libraries(vtaNonVTAConvTestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaNonVTAConvTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_custom_command(
        TARGET vtaNonVTAConvTestBundle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/vtaNonVTAConvTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaNonVTAConvTestBundle.weights.bin)

add_nestc_test(ZCU102 NAME vtaNonVTAConvTestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaNonVTAConvTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_SOURCE_DIR}/vtaNonVTAConvTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/vtaNonVTAConvTestGolden.bin)