cmake_minimum_required(VERSION 3.17)
project(CCodeGenTest)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(CCodeGenTest
        src/CCodeGenTest.cpp
        src/library.cpp
        src/library.h
        src/Image.cpp
        src/Image.h
        src/network.cpp
        src/network.h)

target_link_libraries(CCodeGenTest png pthread)