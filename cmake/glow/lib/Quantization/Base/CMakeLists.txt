if(NESTC_WITH_NMP)
    add_library(QuantizationBase
              Profile.cpp
              ${NESTC_ROOT_DIR}/lib/Quantization/Base/Base.cpp
              Calibration.cpp)
else()
    add_library(QuantizationBase
            Profile.cpp
            ${NESTC_ROOT_DIR}/glow/lib/Quantization/Base/Base.cpp
            Calibration.cpp)
endif()

target_link_libraries(QuantizationBase
  PRIVATE
  LLVMSupport
  Base)

add_dependencies(QuantizationBase AutoGen)
