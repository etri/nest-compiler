option(NESTC_EVTA_PROFILE_AUTOTUNE "Generate code for auto-tuning" OFF)

if(NESTC_EVTA_PROFILE_AUTOTUNE)
    add_definitions(-DNESTC_EVTA_PROFILE_AUTOTUNE=1)
endif()
add_subdirectory(ClassGen)

INCLUDE_DIRECTORIES(include simulator/3rdparty/dlpack/include)
INCLUDE_DIRECTORIES(simulator/3rdparty/dmlc-core/include)
INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/glow/tests/googletest/googletest/include)
INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/vta/vtalib/include/Bundle)
INCLUDE_DIRECTORIES(VTA public ${NESTC_ROOT_DIR}/include/Backends/VTA)

add_library(VTA
        VTA.cpp
        VTAFunction.cpp
        VTANodes.cpp
        VTADeviceManager.cpp
        VTAFactory.cpp
        VTASave.cpp)

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

target_include_directories(VTASaverTest PUBLIC BEFORE ${NESTC_ROOT_DIR}/include/glow/Base/)

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
if (NOT NESTC_BNN)
    add_nestc_test(VTASaverTest ${NESTC_BINARY_DIR}/tests/VTASaverTest --gtest_output=xml:VTASaverTest.xml)
endif()

INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/glow/tests/googletest/googletest/include)


option(GLOW_VTA_FUSION "VTA Fusion Option" ON)

if (GLOW_VTA_FUSION)
    add_definitions(-DVTA_FUSION)
    message(STATUS "VTA Fusion ON")
else()
    message(STATUS "VTA Fusion OFF")
endif ()

option(NESTC_EVTA_GRAPH_OPT "VTAConv graph optimization Enable" OFF)

if (NESTC_EVTA_GRAPH_OPT)
    add_definitions(-DNESTC_EVTA_GRAPH_OPT)
    message(STATUS "VTAConv graph optimization Enable ON")
else()
    message(STATUS "VTAConv graph optimization Enable OFF")
endif ()

if (NESTC_BNN)
    add_definitions(-DVTA_BNN)
    message(STATUS "VTA BNN ON")
else()
    message(STATUS "VTA BNN OFF")
endif ()
