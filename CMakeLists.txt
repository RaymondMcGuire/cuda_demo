# CMake version
cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)

# Project name
project(cuda_demo LANGUAGES CXX CUDA)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
endif()


# Extral library
set(EXTLIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extlib)

# Add sub project
add_subdirectory(cuda_lib)
add_subdirectory(main)