cmake_minimum_required(VERSION 3.5)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/glow/cmake/modules")

project(NESTC C CXX)

option(NESTC_WITH_EVTA "Build the VTA backend" OFF)
option(NESTC_EVTA_BUNDLE_TEST "Build with VTA bundle test" OFF)
option(GLOW_WITH_VTASIM "Build the VTASIM" ON)
option(NESTC_VTA_RUN_ON_AARCH64 "Run VTA on AArch64" OFF)

include(CTest)

set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(NESTC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NESTC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${NESTC_ROOT_DIR}/cmake ${NESTC_ROOT_DIR})

if (NESTC_WITH_EVTA)
    add_definitions(-DNESTC_WITH_VTAINTERPRETER=1)
    set(NESTC_WITH_VTA ON)
else()
    option(NESTC_WITH_VTAINTERPRETER "Build the VTAInterpreter backend" OFF)
    set(NESTC_WITH_VTA OFF)
endif()

message(STATUS "START CMAKE GLOW")
add_subdirectory(glow)
