add_library(STC
        STC.cpp
        STCFunction.cpp
        STCNodes.cpp
        STCDeviceManager.cpp
        STCFactory.cpp)
target_link_libraries(STC
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
                        Runtime)

set(linked_backends ${linked_backends} STC PARENT_SCOPE)
