cmake_minimum_required(VERSION 3.5)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-fno-limit-debug-info)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/glow/cmake/modules")

project(NESTC C CXX)

option(NESTC_WITH_EVTA "Build the VTA backend" OFF)
option(NESTC_WITH_NEWTON "Build the Newton backend" OFF)
option(NESTC_WITH_ENLIGHT "Build the Enlight backend" OFF)
option(NESTC_WITH_OPENEDGE "Build the openedge backend" OFF)
option(NESTC_USE_VTASIM "Build the VTASIM" ON)
option(NESTC_EVTA_RUN_ON_ZCU102 "Run on ZCU102 board" OFF)
option(NESTC_EVTA_MULTI "use Multi-EVTA" OFF)
option(NESTC_USE_PRECOMPILED_EVTA_LIBRARY "use pre-compiled EVTA library" OFF)
option(NESTC_USE_PRECOMPILED_BUNDLE "use pre-compiled bundle code from nestc-data" OFF)
option(NESTC_OPENCL_TEST "Enable OpenCL TEST" OFF)
option(NESTC_CUDA_TEST "Enable CUDA TEST" OFF)
option(NESTC_BNN "Enable BNN mode" OFF)
option(NESTC_WITH_TVM "Build TVM backend" ON)
option(NESTC_DEMO "location of Demo project" OFF )
option(NESTC_WITH_NMP "Build the NMP backend" OFF)
option(NESTC_USE_NMP_FPGA "Build the NMP backend" OFF)


include(CTest)
include(NESTCTestSupport)

set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(NESTC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NESTC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if(EXISTS ${NESTC_ROOT_DIR}/glow/tools/loader/Loader.h)
    message(STATUS "Delete ${NESTC_ROOT_DIR}/glow/tools/loader/Loader.h")
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove -f ${NESTC_ROOT_DIR}/glow/tools/loader/Loader.h)
endif()

if(EXISTS ${NESTC_ROOT_DIR}/glow/include/glow/Quantization/Base/Base.h)
    message(STATUS "Delete ${NESTC_ROOT_DIR}/glow/include/glow/Quantization/Base/Base.h")
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove -f ${NESTC_ROOT_DIR}/glow/include/glow/Quantization/Base/Base.h)

endif()

if(EXISTS ${NESTC_ROOT_DIR}/glow/include/glow/Base/Image.h)
    message(STATUS "Delete ${NESTC_ROOT_DIR}/glow/include/glow/Base/Image.h")
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove -f ${NESTC_ROOT_DIR}/glow/include/glow/Base/Image.h)

endif()

if(EXISTS ${NESTC_ROOT_DIR}/include/glow/Optimizer/GraphOptimizer/FunctionPasses.def)
    message(STATUS "Delete ${NESTC_ROOT_DIR}/glow/include/glow/Optimizer/GraphOptimizer/FunctionPasses.def")
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove -f ${NESTC_ROOT_DIR}/glow/include/glow/Optimizer/GraphOptimizer/FunctionPasses.def)
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${NESTC_ROOT_DIR}/include/glow/Optimizer/GraphOptimizer/FunctionPasses.def 
						     ${NESTC_ROOT_DIR}/glow/include/glow/Optimizer/GraphOptimizer/FunctionPasses.def)
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${NESTC_ROOT_DIR}/cmake/glow ${NESTC_ROOT_DIR}/glow)

if (NESTC_WITH_EVTA)
    add_definitions(-DNESTC_WITH_VTAINTERPRETER=1)
    set(NESTC_WITH_VTA ON)
else()
    option(NESTC_WITH_VTAINTERPRETER "Build the VTAInterpreter backend" OFF)
    set(NESTC_WITH_VTA OFF)
endif()

if (NESTC_WITH_EVTA)
    message("NESTC WITH EVTA in graph cmake")
    add_definitions(-DGLOW_WITH_VTA)
endif()

if(NESTC_WITH_NMP)
    message("NESTC WITH NMP")
    add_definitions(-DGLOW_WITH_NMP=1)
    option(TENSOR_DIMS_32_BITS "Set the max bitwidth of the tensor dimension and related indices to 32b instead of 64b." ON)
    set(RISCV_NMP_TOOLCHAIN $ENV{RISCV_NMP_TOOLCHAIN})
    if (NESTC_USE_NMP_FPGA)
        add_definitions(-DNESTC_USE_NMP_FPGA=1)
    endif()

endif()

if (NESTC_WITH_NEWTON)
    message("NESTC WITH NEWTON in graph cmake")
    add_definitions(-DGLOW_WITH_NEWTON)
endif()

include_directories(BEFORE
        ${NESTC_ROOT_DIR}/
        ${NESTC_ROOT_DIR}/include
        ${NESTC_ROOT_DIR}/include/tools/loader
	${NESTC_ROOT_DIR}/tools/ClassGen/
)

if(NESTC_WITH_TVM)
    message(STATUS "START CMAKE TVM")
    SET(USE_LLVM ON)
    if(NESTC_OPENCL_TEST)
        SET(USE_OPENCL ON)
    endif()

    if(NESTC_CUDA_TEST)
        SET(USE_CUDA ON)
    endif()
    add_subdirectory(tvm)
endif()

if(NESTC_USE_PRECOMPILED_EVTA_LIBRARY)
    include(FetchContent)
    if(NESTC_EVTA_RUN_ON_ZCU102)
        FetchContent_Populate(
                VTALIB
                URL https://nestc-data-pub.s3.ap-northeast-2.amazonaws.com/lib/VTA/x86_sim/21.12/multi/vta.tar.gz
        )
        FetchContent_Populate(
                ACL
                URL https://nestc-data-pub.s3.ap-northeast-2.amazonaws.com/lib/VTA/acl/acl.tar.gz
        )
        if(NESTC_EVTA_MULTI)
            if(NESTC_EVTA_RUN_WITH_GENERIC_BUNDLE)
                set(vtalib_location ${vtalib_SOURCE_DIR}/multi/generic)
            else()
                set(vtalib_location ${vtalib_SOURCE_DIR}/multi/acl)
            endif()
        else()
            if(NESTC_EVTA_RUN_WITH_GENERIC_BUNDLE)
                set(vtalib_location ${vtalib_SOURCE_DIR}/single/generic)
            else()
                set(vtalib_location ${vtalib_SOURCE_DIR}/single/acl)
            endif()
        endif()
        link_directories(${acl_SOURCE_DIR})
    else()
        if(NESTC_EVTA_MULTI)
            FetchContent_Populate(
                    VTALIB
                    URL https://nestc-data-pub.s3.ap-northeast-2.amazonaws.com/lib/VTA/x86_sim/21.12/multi/vta.tar.gz
            )
        else()
            FetchContent_Populate(
                    VTALIB
                    URL https://nestc-data-pub.s3.ap-northeast-2.amazonaws.com/lib/VTA/x86_sim/21.12/single/vta.tar.gz
            )
        endif()
        set(vtalib_location ${vtalib_SOURCE_DIR} )
    endif()
    link_directories(${vtalib_SOURCE_DIR})
    add_library(VTABundle STATIC IMPORTED)
    set_property(TARGET VTABundle PROPERTY IMPORTED_LOCATION
            ${vtalib_location}/libVTABundle.a )
    add_library(vta_runtime STATIC IMPORTED)
    set_target_properties( vta_runtime PROPERTIES IMPORTED_LOCATION
            ${vtalib_location}/libvta_runtime.a )
    INCLUDE_DIRECTORIES(${vtalib_SOURCE_DIR}/include/Bundle/ ${vtalib_SOURCE_DIR}/include/simulator ${vtalib_SOURCE_DIR}/include/simulator/vta)
endif()

if(NESTC_WITH_NEWTON)
    add_subdirectory(examples/newton_bundles)
endif()

if(NESTC_WITH_OPENEDGE)
    add_subdirectory(examples/openedge_bundles)
endif()

message(STATUS "START CMAKE GLOW")
add_subdirectory(glow)

get_property(NESTC_TEST_DEPENDS GLOBAL PROPERTY NESTC_TEST_DEPENDS)
get_property(ZCU102_TEST_DEPENDS GLOBAL PROPERTY ZCU102_TEST_DEPENDS)
get_property(NEWTON_TEST_DEPENDS GLOBAL PROPERTY NEWTON_TEST_DEPENDS)
get_property(MULTIEVTA_TEST_DEPENDS GLOBAL PROPERTY MULTIEVTA_TEST_DEPENDS)
add_custom_target(check_nestc COMMAND ${CMAKE_CTEST_COMMAND} -L NESTC
        DEPENDS ${NESTC_TEST_DEPENDS} USES_TERMINAL)
add_custom_target(check_zcu102 COMMAND ${CMAKE_CTEST_COMMAND} -L ZCU102
        DEPENDS ${ZCU102_TEST_DEPENDS} USES_TERMINAL)
add_custom_target(check_newton COMMAND ${CMAKE_CTEST_COMMAND} -L NEWTON
        DEPENDS ${NEWTON_TEST_DEPENDS} USES_TERMINAL)
add_custom_target(check_enlight COMMAND ${CMAKE_CTEST_COMMAND} -L ENLIGHT
        DEPENDS ${ENLIGHT_TEST_DEPENDS} USES_TERMINAL)
add_custom_target(check_multievta COMMAND ${CMAKE_CTEST_COMMAND} -L MULTIEVTA
        DEPENDS ${MULTIEVTA_TEST_DEPENDS} USES_TERMINAL)


if(NESTC_DEMO)
    add_subdirectory(${NESTC_DEMO})
endif()