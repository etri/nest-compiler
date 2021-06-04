cmake_minimum_required(VERSION 3.5)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/glow/cmake/modules")

project(NESTC C CXX)

option(NESTC_WITH_EVTA "Build the VTA backend" OFF)
option(NESTC_USE_VTASIM "Build the VTASIM" ON)
option(NESTC_VTA_RUN_ON_AARCH64 "Run VTA on AArch64" OFF)

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
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${NESTC_ROOT_DIR}/cmake ${NESTC_ROOT_DIR})

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
        ${NESTC_ROOT_DIR}/include
        ${NESTC_ROOT_DIR}/include/tools/loader
)


message(STATUS "START CMAKE GLOW")
add_subdirectory(glow)
add_subdirectory(examples/bundles/resnet18_relay)

get_property(NESTC_TEST_DEPENDS GLOBAL PROPERTY NESTC_TEST_DEPENDS)
get_property(ZCU102_TEST_DEPENDS GLOBAL PROPERTY ZCU102_TEST_DEPENDS)
add_custom_target(check_nestc COMMAND ${CMAKE_CTEST_COMMAND} -L NESTC
        DEPENDS ${NESTC_TEST_DEPENDS} USES_TERMINAL)
add_custom_target(check_zcu102 COMMAND ${CMAKE_CTEST_COMMAND} -L ZCU102
        DEPENDS ${ZCU102_TEST_DEPENDS} USES_TERMINAL)
message("YIKWON_TEMP0 ${NESTC_TEST_DEPENDS}")