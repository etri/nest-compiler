add_library(Converter
            Float16Converter.cpp
            FusedRowwiseConverter.cpp
            ${NESTC_ROOT_DIR}/lib/Converter/FunctionConverter.cpp
            TypeAToTypeBFunctionConverter.cpp)

target_link_libraries(Converter
                      PRIVATE
                        Base
                        Graph)
