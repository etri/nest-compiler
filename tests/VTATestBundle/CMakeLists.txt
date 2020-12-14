cmake_minimum_required(VERSION 3.10)
project(vtalib_example)


Option(VTADEBUGMODE 0)
Option(ZCU102 "cross-compile target for ultra96" 1)
Option(ONBOARD 0)

add_definitions(-DDMLC_USE_FOPEN64=0
        -DVTA_TARGET=zcu102
        -DVTA_HW_VER=0.0.1
        -DVTA_LOG_INP_WIDTH=3
        -DVTA_LOG_WGT_WIDTH=3
        -DVTA_LOG_ACC_WIDTH=5
        -DVTA_LOG_BATCH=0
        -DVTA_LOG_BLOCK=4
        -DVTA_LOG_UOP_BUFF_SIZE=16
        -DVTA_LOG_INP_BUFF_SIZE=16
        -DVTA_LOG_WGT_BUFF_SIZE=19
        -DVTA_LOG_ACC_BUFF_SIZE=18
        -DVTA_LOG_BLOCK_IN=4
        -DVTA_LOG_BLOCK_OUT=4
        -DVTA_LOG_OUT_WIDTH=3
        -DVTA_LOG_OUT_BUFF_SIZE=16
        -DVTA_LOG_BUS_WIDTH=7
        -DVTA_IP_REG_MAP_RANGE=0x1000
        -DVTA_FETCH_ADDR=0xA0000000
        -DVTA_LOAD_ADDR=0xA0001000
        -DVTA_COMPUTE_ADDR=0xA0002000
        -DVTA_STORE_ADDR=0xA0003000
        -DVTA_FETCH_INSN_COUNT_OFFSET=16
        -DVTA_FETCH_INSN_ADDR_OFFSET=24
        -DVTA_LOAD_INP_ADDR_OFFSET=16
        -DVTA_LOAD_WGT_ADDR_OFFSET=24
        -DVTA_COMPUTE_DONE_WR_OFFSET=16
        -DVTA_COMPUTE_DONE_RD_OFFSET=24
        -DVTA_COMPUTE_UOP_ADDR_OFFSET=32
        -DVTA_COMPUTE_BIAS_ADDR_OFFSET=40
        -DVTA_STORE_OUT_ADDR_OFFSET=16
        -DVTA_COHERENT_ACCESSES=true
        -Dvta_EXPORTS)



if(ONBOARD)
    find_library(LIBCMA
            NAMES libcma.so
            HINTS /usr/lib/
            REQUIRED)
    message(STATUS "On board ZCU102 compilation")
elseif(ZCU102)
    message(STATUS "Target is ZCU102")
    SET(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
    SET(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
    SET(CMAKE_LINKER /usr/bin/aarch64-linux-gnu-ld)
    SET(CMAKE_NM /usr/bin/aarch64-linux-gnu-nm)
    SET(CMAKE_OBJCOPY /usr/bin/aarch64-linux-gnu-objcopy)
    SET(CMAKE_OBJDUMP /usr/bin/aarch64-linux-gnu-objdump)
    SET(CMAKE_RANLIB /usr/bin/aarch64-linux-gnu-ranlib)
    find_library(LIBCMA
            NAMES libcma.so
            HINTS lib/aarch64
            REQUIRED)
else()
    message(STATUS "Target is NONE")
endif()

include_directories(include)
include_directories(include/include)




if(VTADEBUGMODE)
    add_definitions(-DVTADEBUGMODE=1)
endif()

add_compile_options(-std=c++11 -faligned-new -O0 -g -Wall -fPIC  -fPIC  -Wno-psabi)


#add_library(npu SHARED lib/vta/src/api.cc lib/vta/src/pynq_driver.cc)
add_library(npu lib/vta/src/api.cc lib/vta/src/pynq_driver.cc)

add_executable(vtaTestBundle vtaTestBundleMain.cpp vtaTestBundle.cpp)
target_link_libraries(vtaTestBundle
        npu ${LIBCMA} pthread)
