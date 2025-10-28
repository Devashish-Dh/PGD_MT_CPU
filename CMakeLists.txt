cmake_minimum_required(VERSION 3.10)

project(PGDCPU
    VERSION 1.0
    DESCRIPTION "NA"
    LANGUAGES CXX
)

add_executable(pgdcpu
    src/main.cpp
    src/funcs.cpp
)

target_include_directories(pgdcpu
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(pgdcpu PUBLIC cxx_std_17)
