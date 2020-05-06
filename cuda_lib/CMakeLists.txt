cmake_minimum_required(VERSION 3.17.0)

set(PROJECT_NAME cuda_lib)
set(CUDA_MATH_LIB_INLCUDE "../cuda_math/include")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
endif()

file(GLOB SOURCES "src/*.cu" "src/*.cpp" "src/*.c")
file(GLOB HEADERS "include/*.cuh" "include/*.hpp" "include/*.h")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ../bin/Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ../bin/Debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ../lib/Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ../lib/Debug)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ../lib/Release)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ../lib/Debug)

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME_DEBUG ${PROJECT_NAME}-debug
    OUTPUT_NAME_RELEASE ${PROJECT_NAME}
)
target_include_directories(${PROJECT_NAME} PUBLIC
    include 
    ${CUDA_MATH_LIB_INLCUDE}
    ${EXTLIB_DIR}
)
