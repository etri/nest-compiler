add_subdirectory(ClassGen)

add_library(VTAInterpreter
        VTAInterpreter.cpp
        VTAInterpreterFunction.cpp
        VTAInterpreterNodes.cpp
        VTAInterpreterDeviceManager.cpp
        VTAInterpreterFactory.cpp)
target_link_libraries(VTAInterpreter
                      PRIVATE
                        Backend
                        Base
                        CodeGen
                        Graph
                        IR
                        LLVMCore
                        IROptimizer
                        IROptimizerPipeline
                        GraphOptimizerPipeline
                        QuantizationBase
                        Runtime)

set(linked_backends ${linked_backends} VTAInterpreter PARENT_SCOPE)

INCLUDE_DIRECTORIES(include ../../../tests/googletest/googletest/include/)
INCLUDE_DIRECTORIES(../VTA)
add_executable(NESTVTAInterpreterBackendTest tests/NESTVTAInterpreterBackendTest.cpp tests/VTAInterpreterBackendTestUtils.cpp)
target_include_directories(NESTVTAInterpreterBackendTest
        PUBLIC
        "${SYNAPSE_INCLUDE_DIR}")

target_link_libraries(NESTVTAInterpreterBackendTest
        PRIVATE
        Backend
        Backends
        Base
        Converter
        ExecutionEngine
        Graph
        GraphOptimizer
        HostManager
        Quantization
        QuantizationBase
        LLVMSupport
        gtest
        TestMain
        IR)
set_target_properties( NESTVTAInterpreterBackendTest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${GLOW_BINARY_DIR}/tests"
        )

add_nestc_test(NESTVTAInterpreterBackendTest ${GLOW_BINARY_DIR}/tests/NESTVTAInterpreterBackendTest --gtest_output=xml:NESTVTAInterpreterBackendTest.xml)

option(GLOW_VTA_INTERPRETER_FUSION "VTA Interpreter Fusion Enable" ON)

if (GLOW_VTA_INTERPRETER_FUSION)
    add_definitions(-DVTA_INTERPRETER_FUSION)
    message(STATUS "VTA Interpreter Fusion ON")
else()
    message(STATUS "VTA Interpreter Fusion OFF")
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DVTA_DEBUG_MODE)
endif()