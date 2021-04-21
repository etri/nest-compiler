add_library(Relay
        Relay.cpp
        RelayFunction.cpp
        RelayNodes.cpp
        RelayDeviceManager.cpp
        RelayFactory.cpp
        RelaySave.cpp)
target_link_libraries(Relay
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

set(linked_backends ${linked_backends} Relay PARENT_SCOPE)

INCLUDE_DIRECTORIES(include ../../../tests/googletest/googletest/include/)
