add_subdirectory(ClassGen)

include(GlowAddLibjit)
include(GlowSerialize)

# LIBJIT NMP specific source files.
set(LIBJIT_NMP_SOURCE_FILES)
list(APPEND LIBJIT_NMP_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_nmp/lib_nmp.cpp
)

# LIBJIT NMP compile options.
set(LIBJIT_NMP_COMPILE_OPTIONS
  --sysroot=${RISCV_NMP_TOOLCHAIN}/riscv32-unknown-elf
  --gcc-toolchain=${RISCV_NMP_TOOLCHAIN}
  # --target=<arch><sub>-<vendor>-<sys>-<abi>
  --target=riscv32
  # Disable implicit builtin knowledge of functions
  -fno-builtin
  # Compile common globals like normal definitions
  -fno-common
  # Enable LTO in 'full' mode
  -flto
  # Optimization level
  -O2
  ${LLVMCPURuntimeExtraFlags}
  # Include LIBJIT external functions (callbacks).
  -DGLOW_LIBJIT_EXTERNAL_FUNCTIONS
)

# Add NMP LIBJIT.
glow_add_libjit(
  NAME "libjit_nmp"
  SOURCE_FILES ${LIBJIT_NMP_SOURCE_FILES}
  COMPILE_OPTIONS ${LIBJIT_NMP_COMPILE_OPTIONS}
)

# Add native NMP LIBJIT library used for testing.
if (NOT MSVC)
    add_library(NMPRuntimeNative
              ${LIBJIT_NMP_SOURCE_FILES})
endif(NOT MSVC)

add_library(NMPBackend
            ${libjit_nmp_INCLUDE_FILE}
            CommandLine.cpp
            NMPBundleSaver.cpp
            NMPBackend.cpp
            NMPDeviceManager.cpp
            NMPFactory.cpp
            NMPFunction.cpp
            NMPLLVMIRGen.cpp
            NMPTensorAnalysis.cpp
            NMPTensorLayout.cpp
            Transforms.cpp)

target_link_libraries(NMPBackend
                      PUBLIC
                        Backend
                        Base
                        CodeGen
                        Flags
                        Graph
                        IR
                        IROptimizer
                        IROptimizerPipeline
                        GraphOptimizerPipeline
                        QuantizationBase
                        Runtime
                        LLVMIRCodeGen)

add_dependencies(NMPBackend libjit_nmp_TARGET)

set(linked_backends ${linked_backends} NMPBackend PARENT_SCOPE)

# List the object files from "libobj_nmp" directory.
FILE(GLOB objects "${CMAKE_CURRENT_LIST_DIR}/libobj_nmp/*.o")

# Add object file for testing.
if (GLOW_WITH_BUNDLES)
  list(APPEND objects "${GLOW_SOURCE_DIR}/examples/bundles/bundle_with_extra_objects/test.o")
endif()

# Serialize external object files.
glow_serialize(
  INP_FILES ${objects}
  OUT_DIR "${GLOW_BINARY_DIR}/libobj/libobj_nmp"
  OUT_FILE "${GLOW_BINARY_DIR}/libobj/libobj_nmp/nmpObjectRegistry.h"
  OUT_VAR "nmpObjectRegistry"
  OUT_TARGET nmpObjectRegistryTarget
)
add_dependencies(NMPBackend nmpObjectRegistryTarget)
include_directories(${GLOW_BINARY_DIR}/libobj/libobj_nmp)
