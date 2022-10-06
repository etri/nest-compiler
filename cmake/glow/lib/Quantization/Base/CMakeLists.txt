add_library(QuantizationBase
              Profile.cpp
              ${NESTC_ROOT_DIR}/lib/Quantization/Base/Base.cpp
              Calibration.cpp)
target_link_libraries(QuantizationBase
  PRIVATE
  LLVMSupport
  Base)

add_dependencies(QuantizationBase AutoGen)
