add_executable(vtaQuantizeTestBundle vtaQuantizeTestBundleMain.cpp vtaQuantizeTestBundle.cpp)
target_link_libraries(vtaQuantizeTestBundle VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaQuantizeTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaQuantizeTestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaQuantizeTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_SOURCE_DIR}/vtaQuantizeTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/vtaQuantizeTestGolden.bin)