INCLUDE_DIRECTORIES(Enlight public ${NESTC_ROOT_DIR}/include/Backends/Enlight)

set(CMAKE_CXX_STANDARD 17)
add_library(Enlight
        Enlight.cpp
        EnlightFunction.cpp
        EnlightNodes.cpp
        EnlightDeviceManager.cpp
        EnlightFactory.cpp
        EnlightSave.cpp
        )

target_include_directories(Enlight
        PUBLIC
        "${SYNAPSE_INCLUDE_DIR}")
target_link_libraries(Enlight
                      PRIVATE
                        Backend
                        Base
                        CodeGen
                        Graph
                        IR
                        LLVMCore
                        IROptimizer
                        GraphOptimizerPipeline
                        QuantizationBase
                        Runtime
        )

set(linked_backends ${linked_backends} Enlight PARENT_SCOPE)


INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/lib)
add_executable(EnlightSaverTest
        tests/EnlightSaverTest.cpp
        tests/EnlightBackendTestUtils.cpp)

target_include_directories(EnlightSaverTest PUBLIC BEFORE ${NESTC_ROOT_DIR}/include/glow/Base/)

target_link_libraries(EnlightSaverTest
        PRIVATE
        Backends
        Graph
        Support
        Base
        HostManager
        GraphOptimizer
        ExecutionEngine
        Quantization
        gtest
        TestMain)
set_target_properties( EnlightSaverTest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${NESTC_BINARY_DIR}/tests"
        )

INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/glow/tests/googletest/googletest/include ${NESTC_ROOT_DIR}/include/Backends/Enlight)
add_executable(EnlightBackendTest tests/EnlightBackendTest.cpp tests/EnlightBackendTestUtils.cpp)
target_link_libraries(EnlightBackendTest
        PRIVATE
        Backends
        Base
        ExecutionEngine
        Graph
        LLVMSupport
        TestMain
        )
set_target_properties(EnlightBackendTest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${NESTC_BINARY_DIR}/tests"
        )

add_nestc_test(ENLIGHT EnlightBackendTest ${NESTC_BINARY_DIR}/tests/EnlightBackendTest --gtest_output=xml:EnlightBackendTest.xml)