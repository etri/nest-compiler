INCLUDE_DIRECTORIES(Newton public ${NESTC_ROOT_DIR}/include/Backends/Newton)
INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/external_libs/newton/aimddk/libaim
        ${NESTC_ROOT_DIR}/external_libs/newton/aimddk/libaim/include
        ${NESTC_ROOT_DIR}/external_libs/newton/aimddk/libaim/external/eigen
        )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
add_library(Newton
        Newton.cpp
        NewtonFunction.cpp
        NewtonNodes.cpp
        NewtonDeviceManager.cpp
        NewtonFactory.cpp
        NewtonSave.cpp
        )

target_include_directories(Newton
        PUBLIC
        "${SYNAPSE_INCLUDE_DIR}")
target_link_libraries(Newton
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
                        ${NESTC_ROOT_DIR}/external_libs/newton/aimddk/libaim/lib/libaim.so
        )

set(linked_backends ${linked_backends} Newton PARENT_SCOPE)


INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/lib)
add_executable(NewtonSaverTest
        tests/NewtonSaverTest.cpp
        tests/NewtonBackendTestUtils.cpp)

target_include_directories(NewtonSaverTest PUBLIC BEFORE ${NESTC_ROOT_DIR}/include/glow/Base/)

target_link_libraries(NewtonSaverTest
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
set_target_properties( NewtonSaverTest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${NESTC_BINARY_DIR}/tests"
        )

INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/glow/tests/googletest/googletest/include ${NESTC_ROOT_DIR}/include/Backends/Newton)
add_executable(NewtonBackendTest tests/NewtonBackendTest.cpp tests/NewtonBackendTestUtils.cpp )
target_link_libraries(NewtonBackendTest
        PRIVATE
        Backends
        Base
        ExecutionEngine
        Graph
        LLVMSupport
        TestMain
        ${NESTC_ROOT_DIR}/external_libs/newton/aimddk/libaim/lib/libaim.so
        )
set_target_properties(NewtonBackendTest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${NESTC_BINARY_DIR}/tests"
        )

add_nestc_test(NEWTON NewtonBackendTest ${NESTC_BINARY_DIR}/tests/NewtonBackendTest --gtest_output=xml:NewtonBackendTest.xml)