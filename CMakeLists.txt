cmake_minimum_required(VERSION 3.5)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-fno-limit-debug-info)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/glow/cmake/modules")

project(NESTC C CXX)

option(NESTC_WITH_EVTA "Build the VTA backend" OFF)
option(NESTC_USE_VTASIM "Build the VTASIM" ON)
option(NESTC_EVTA_RUN_ON_ZCU102 "Run on ZCU102 board" OFF)
option(NESTC_EVTA_MULTI "use Multi-EVTA" OFF)
option(NESTC_USE_PRECOMPILED_EVTA_LIBRARY "use pre-compiled EVTA library" OFF)
option(NESTC_USE_PRECOMPILED_BUNDLE "use pre-compiled bundle code from nestc-data" OFF)
option(NESTC_OPENCL_TEST "Enable OpenCL TEST" OFF)
option(NESTC_CUDA_TEST "Enable CUDA TEST" OFF)
option(NESTC_BNN "Enable BNN mode" OFF)

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

if(EXISTS ${NESTC_ROOT_DIR}/glow/tools/ClassGen/NodeBuilder.h)
    message(STATUS "Delete ${NESTC_ROOT_DIR}/glow/tools/ClassGen/NodeBuilder.h")
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove -f ${NESTC_ROOT_DIR}/glow/tools/ClassGen/NodeBuilder.h)

endif()

if(EXISTS ${NESTC_ROOT_DIR}/glow/tools/ClassGen/MemberType.h)
    message(STATUS "Delete ${NESTC_ROOT_DIR}/glow/tools/ClassGen/MemberType.h")
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove -f ${NESTC_ROOT_DIR}/glow/tools/ClassGen/MemberType.h)

endif()

if(EXISTS ${NESTC_ROOT_DIR}/glow/include/glow/Quantization/Base/Base.h)
    message(STATUS "Delete ${NESTC_ROOT_DIR}/glow/include/glow/Quantization/Base/Base.h")
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove -f ${NESTC_ROOT_DIR}/glow/include/glow/Quantization/Base/Base.h)

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

include_directories(BEFORE
        ${NESTC_ROOT_DIR}/
        ${NESTC_ROOT_DIR}/include
        ${NESTC_ROOT_DIR}/include/tools/loader
        ${NESTC_ROOT_DIR}/tools/ClassGen/
)
message(STATUS "START CMAKE TVM")
SET(USE_LLVM ON)
if(NESTC_OPENCL_TEST)
    SET(USE_OPENCL ON)
endif()

if(NESTC_CUDA_TEST)
    SET(USE_CUDA ON)
endif()
add_subdirectory(tvm)

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

message(STATUS "START CMAKE GLOW")
add_subdirectory(glow)

get_property(NESTC_TEST_DEPENDS GLOBAL PROPERTY NESTC_TEST_DEPENDS)
get_property(ZCU102_TEST_DEPENDS GLOBAL PROPERTY ZCU102_TEST_DEPENDS)
get_property(MULTIEVTA_TEST_DEPENDS GLOBAL PROPERTY MULTIEVTA_TEST_DEPENDS)
add_custom_target(check_nestc COMMAND ${CMAKE_CTEST_COMMAND} -L NESTC
        DEPENDS ${NESTC_TEST_DEPENDS} USES_TERMINAL)
add_custom_target(check_zcu102 COMMAND ${CMAKE_CTEST_COMMAND} -L ZCU102
        DEPENDS ${ZCU102_TEST_DEPENDS} USES_TERMINAL)
add_custom_target(check_multievta COMMAND ${CMAKE_CTEST_COMMAND} -L MULTIEVTA
        DEPENDS ${MULTIEVTA_TEST_DEPENDS} USES_TERMINAL)
