add_subdirectory(ClassGen)

INCLUDE_DIRECTORIES(include simulator/3rdparty/dlpack/include)
INCLUDE_DIRECTORIES(simulator/3rdparty/dmlc-core/include)
INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/glow/tests/googletest/googletest/include)
INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/vta/vtalib/include/Bundle)
add_library(VTA
        VTA.cpp
        VTAFunction.cpp
        VTANodes.cpp
        VTADeviceManager.cpp
        VTAFactory.cpp
        VTASave.cpp VTASchedules.h)

target_include_directories(VTA
        PUBLIC
        "${SYNAPSE_INCLUDE_DIR}")
target_link_libraries(VTA
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
        VTABundle
        vta_runtime
        )

set(linked_backends ${linked_backends} VTA PARENT_SCOPE)


INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/lib)
add_executable(VTASaverTest
        tests/VTASaverTest.cpp
        tests/VTABackendTestUtils.cpp)

target_link_libraries(VTASaverTest
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
set_target_properties( VTASaverTest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${NESTC_BINARY_DIR}/tests"
        )

add_nestc_test(VTASaverTest ${NESTC_BINARY_DIR}/tests/VTASaverTest --gtest_output=xml:VTASaverTest.xml)

INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/glow/tests/googletest/googletest/include)


option(GLOW_VTA_FUSION "VTA Fusion Option" ON)

if (GLOW_VTA_FUSION)
    add_definitions(-DVTA_FUSION)
    message(STATUS "VTA Fusion ON")
else()
    message(STATUS "VTA Fusion OFF")
endif ()
