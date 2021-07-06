add_executable(vtaMaxPoolTestBundle vtaMaxPoolTestBundleMain.cpp vtaMaxPoolTestBundle.cpp)
add_dependencies(vtaMaxPoolTestBundle ${BUNDLE_DEPENDS})
target_link_libraries(vtaMaxPoolTestBundle
       VTABundle ${BUNDLE_LINK_LIB})
set_target_properties(vtaMaxPoolTestBundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_custom_command(
        TARGET vtaMaxPoolTestBundle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/vtaMaxPoolTestBundle.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/vtaMaxPoolTestBundle.weights.bin)

add_nestc_test(ZCU102 NAME vtaMaxPoolTestBundle
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaMaxPoolTestBundle USE_SH 1 PARAMS ${CMAKE_CURRENT_SOURCE_DIR}/vtaMaxPoolTestInput.bin
        USE_DIFF 1 DIFF_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/vtaMaxPoolTestGolden.bin)