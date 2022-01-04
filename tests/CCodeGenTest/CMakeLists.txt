cmake_minimum_required(VERSION 3.17)
project(CCodeGenTest)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_SOURCE_DIR}/src/data.bin
        COMMAND
        wget https://gitlab.com/yongin.kwon/nestc-data/-/raw/master/ccodegen-data/data.bin -O ${CMAKE_CURRENT_SOURCE_DIR}/src/data.bin
)
set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(CCodeGenTest
        ${CMAKE_CURRENT_SOURCE_DIR}/src/CCodeGenTest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/library.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/library.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Image.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Image.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/network.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/network.h
        )

target_link_libraries(CCodeGenTest png pthread)