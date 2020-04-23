cmake_minimum_required(VERSION 3.17.0)

# CUDA config
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
message("Cuda version: " ${CUDA_VERSION})
	set(PROJECT_NAME cuda)
	if(MSVC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
	else()
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++14")
    endif()
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
	set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -O0 -G -g")
    set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -lineinfo")
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
	CUDA_ADD_LIBRARY(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})
	set_target_properties(
		${PROJECT_NAME} PROPERTIES
		OUTPUT_NAME_DEBUG ${PROJECT_NAME}-debug
		OUTPUT_NAME_RELEASE ${PROJECT_NAME}
	)
	target_include_directories(${PROJECT_NAME} PUBLIC
		include
	)
endif()