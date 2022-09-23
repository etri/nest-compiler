include_directories(BEFORE
        ${NESTC_ROOT_DIR}/include/tools/loader
        )

add_executable(ccode-generator
        ${NESTC_ROOT_DIR}/tools/loader/Loader.cpp
        ${NESTC_ROOT_DIR}/tools/loader/CCodeGenerator.cpp)

target_link_libraries(ccode-generator
        PRIVATE
        Backends
        Base
        Converter
        ExecutionEngine
        HostManager
        Graph
        Importer
        GraphOptimizer
        Quantization
        LLVMSupport)

add_executable(model-partition-tuner
        ${NESTC_ROOT_DIR}/tools/loader/Loader.cpp
        ${NESTC_ROOT_DIR}/tools/loader/ModelPartitionTuner.cpp)


target_link_libraries(model-partition-tuner
        PRIVATE
        Backends
        Base
        Converter
        Graph
        HostManager
        Importer
        ExecutionEngine
        GraphOptimizer
        Quantization
        LLVMSupport)

INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/glow/tests/googletest/googletest/include)
INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/lib)
add_executable(NestPartitionerTest
        ${NESTC_ROOT_DIR}/lib/Partitioner/tests/NestPartitionerTest.cpp)

target_link_libraries(NestPartitionerTest
        PRIVATE
        Backends
        Base
        Converter
        Graph
        HostManager
        Importer
        ExecutionEngine
        GraphOptimizer
        Quantization
        LLVMSupport
        gtest
        TestMain)

set_target_properties(NestPartitionerTest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${NESTC_BINARY_DIR}/tests"
        )

add_nestc_test(NestPartitionerTest ${NESTC_BINARY_DIR}/tests/NestPartitionerTest  --gtest_output=xml:NestPartitionerTest.xml)


add_executable(nestos-partitioner
        ${NESTC_ROOT_DIR}/tools/loader/Loader.cpp
        ${NESTC_ROOT_DIR}/tools/loader/nestos-partitioner.cpp)

target_link_libraries(nestos-partitioner
        PRIVATE
        Backends
        Base
        Converter
        Graph
        HostManagerNestos
        Importer
        ExecutionEngine
        GraphOptimizer
        Quantization
        LLVMSupport)

add_executable(image-classifier
  ${NESTC_ROOT_DIR}/tools/loader/Loader.cpp
  ImageClassifier.cpp
  ExecutorCore.cpp
  ExecutorCoreHelperFunctions.cpp)

target_link_libraries(image-classifier
                      PRIVATE
                        Backends
                        Base
                        Converter
                        ExecutionEngine
                        HostManager
                        Graph
                        Importer
                        GraphOptimizer
                        Quantization
                        LLVMSupport)

add_executable(object-detector
               ${NESTC_ROOT_DIR}/tools/loader/Loader.cpp
               ObjectDetector.cpp
               ExecutorCore.cpp
               ExecutorCoreHelperFunctions.cpp)

target_link_libraries(object-detector
                      PRIVATE
                      Backends
                      Base
                      Converter
                      ExecutionEngine
                      HostManager
                      Graph
                      Importer
                      GraphOptimizer
                      Quantization
                      LLVMSupport)

add_executable(text-translator
  ${NESTC_ROOT_DIR}/tools/loader/Loader.cpp
  TextTranslator.cpp)

target_link_libraries(text-translator
                      PRIVATE
                        Backends
                        Base
                        Converter
                        Graph
                        HostManager
                        Importer
                        ExecutionEngine
                        GraphOptimizer
                        Quantization
                        LLVMSupport)

add_executable(model-runner
  ${NESTC_ROOT_DIR}/tools/loader/Loader.cpp
  ModelRunner.cpp)

target_link_libraries(model-runner
                      PRIVATE
                        Backends
                        Base
                        Converter
                        Graph
                        HostManager
                        Importer
                        ExecutionEngine
                        GraphOptimizer
                        Quantization
                        LLVMSupport)

add_executable(x-model-builder
  ${NESTC_ROOT_DIR}/tools/loader/Loader.cpp
  XModelBuilder.cpp)

target_link_libraries(x-model-builder
                      PRIVATE
                        Backends
                        Base
                        Converter
                        Graph
                        HostManager
                        Importer
                        ExecutionEngine
                        GraphOptimizer
                        Quantization
                        LLVMSupport)

add_executable(model-compiler
  ${NESTC_ROOT_DIR}/tools/loader/Loader.cpp
  ModelCompiler.cpp)

target_link_libraries(model-compiler
                      PRIVATE
                        Backends
                        Base
                        Converter
                        Graph
                        HostManager
                        Importer
                        ExecutionEngine
                        GraphOptimizer
                        Quantization
                        LLVMSupport)

add_executable(model-profiler
  ${NESTC_ROOT_DIR}/tools/loader/Loader.cpp
  LoaderUtils.cpp
  ModelProfiler.cpp)

target_link_libraries(model-profiler
                      PRIVATE
                        Backends
                        Base
                        Converter
                        Graph
                        HostManager
                        Importer
                        ExecutionEngine
                        GraphOptimizer
                        Quantization
                        LLVMSupport)

add_executable(model-tuner
  ${NESTC_ROOT_DIR}/tools/loader/Loader.cpp
  LoaderUtils.cpp
  ModelTuner.cpp)

target_link_libraries(model-tuner
                      PRIVATE
                        Backends
                        Base
                        Converter
                        Graph
                        HostManager
                        Importer
                        ExecutionEngine
                        GraphOptimizer
                        Quantization
                        LLVMSupport)
