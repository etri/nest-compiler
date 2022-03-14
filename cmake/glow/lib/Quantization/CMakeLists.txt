add_subdirectory(Base)

add_library(Quantization
              Serialization.cpp
              ${NESTC_ROOT_DIR}/lib/Quantization/Quantization.cpp)

target_link_libraries(Quantization
                      PRIVATE
                        Converter
                        Graph
                        Backend
                        QuantizationBase
                        LLVMSupport)
