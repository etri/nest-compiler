INCLUDE_DIRECTORIES(${NESTC_ROOT_DIR}/external_libs/newton/aimddk/libaim
        ${NESTC_ROOT_DIR}/external_libs/newton/aimddk/libaim/include
        ${NESTC_ROOT_DIR}/external_libs/newton/aimddk/libaim/external/eigen
        )
set(CMAKE_CXX_STANDARD 17)

# library version
set(VERSION 0)
set(SUBVERSION 0)
set(PATCHLEVEL 0)

# compile flags
set(DEBUG 0)
set(DEBUG_DATA 0)
set(OPENMP 1)
set(FASTCOPY 1)
set(MINI_BATCH_SIZE 16)
set(CAST_BF16_FP32 1)
set(MEASURE_SWOH 0)
set(ISR_GDB 0)
set(TEST_AIMDMA 1)
set(TEST_AIMDMA_DA 1)
set(USE_MMAP_STOREWEIGHT 1)
set(ENABLE_MEM_TEST_THREAD 0)

# Validation options
# 0: Nothing
# 1: Show ISA command streams
# 2: Show ISA command streams & BitFlip validation
set(DATA_VALIDATION_CHECK 0)

set(BRIEF_INFO 0)
set(BREAK_DOWN 0)
set(ROW_MAJOR 0)
# AiM DMA configuration
# 0: ISR 16KB, CFR 16KB, GPR 64KB
# 1: ISR 2MB, CFR 16KB, GPR 512KB
# 2: ISR 2MB, CFR 16KB, GPR 8MB
set(AIM_DMA_CONFIG 1)
set(RNN_CPU_OP 0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG=${DEBUG_DATA} -DFASTCOPY=${FASTCOPY} -DMINI_BATCH_SIZE=${MINI_BATCH_SIZE}  ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCAST_BF16_FP32=${CAST_BF16_FP32} -DMEASURE_SWOH=${MEASURE_SWOH} -DISR_GDB=${ISR_GDB} ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTEST_AIMDMA=${TEST_AIMDMA} -DTEST_AIMDMA_DA=${TEST_AIMDMA_DA} -DUSE_MMAP_STOREWEIGHT=${USE_MMAP_STOREWEIGHT} ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAIM_DMA_CONFIG=${AIM_DMA_CONFIG} -DDATA_VALIDATION_CHECK=${DATA_VALIDATION_CHECK} -DBRIEF_INFO=${BRIEF_INFO}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBREAK_DOWN=${BREAK_DOWN} -DROW_MAJOR=${ROW_MAJOR} -DRNN_CPU_OP=${RNN_CPU_OP}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_MEM_TEST_THREAD=${ENABLE_MEM_TEST_THREAD} -mavx2  -std=c++17 -fPIC -Wall -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -I. -I./include -I./external/eigen")


set(CMAKE_CXX_STANDARD 17)
FILE(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(object ${subdirs})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${object})
    if(NOT ${object} STREQUAL bnn)
      message(STATUS "Adding ${object} Newton Bundles.")
      add_subdirectory(${object})
    endif()
  endif()
endforeach()
