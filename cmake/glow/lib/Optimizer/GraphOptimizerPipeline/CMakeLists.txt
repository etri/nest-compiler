add_library(GraphOptimizerPipeline
              ${NESTC_ROOT_DIR}/lib/Optimizer/GraphOptimizerPipeline/FunctionPassPipeline.cpp)

target_link_libraries(GraphOptimizerPipeline
                      PRIVATE
                        Graph
                        LLVMCore
                        PassManager)

if (NESTC_BNN)
    add_definitions(-DNEST_BNN)
endif()
