add_executable(vtaDequantizeTestBundle vtaDequantizeTestBundleMain.cpp vtaDequantizeTestBundle.cpp)
target_link_libraries(vtaDequantizeTestBundle
         VTABundle  ${BUNDLE_LINK_LIB})

set_target_properties(vtaDequantizeTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaDequantizeTestBundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaDequantizeTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_SOURCE_DIR}/vtaDequantizeTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/vtaDequantizeTestGolden.bin)