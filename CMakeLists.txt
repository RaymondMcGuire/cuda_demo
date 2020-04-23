# CMake version
cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)

# Project name
project(cuda_demo)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(GLOBAL_DIR ${PROJECT_SOURCE_DIR})

if(MSVC)
	string(REPLACE "/W3" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    add_compile_options(
        $<$<CONFIG:>:/MT>
        $<$<CONFIG:Debug>:/MTd> 
        $<$<CONFIG:Release>:/MT> 
    )

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++ -O3")
endif()

# Add sub project
add_subdirectory(cuda)
add_subdirectory(cuda_ray_tracer)
add_subdirectory(main)