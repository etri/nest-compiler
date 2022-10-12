add_library(Interpreter
              ${NESTC_ROOT_DIR}/lib/Interpreter/Interpreter.cpp
              InterpreterFunction.cpp
              ${NESTC_ROOT_DIR}/lib/Interpreter/InterpreterNodes.cpp
              InterpreterDeviceManager.cpp
              InterpreterFactory.cpp)
target_link_libraries(Interpreter
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

set(linked_backends ${linked_backends} Interpreter PARENT_SCOPE)
