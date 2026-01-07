cmake_minimum_required(VERSION 3.20)
project(optix_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find OptiX SDK
if(NOT DEFINED OPTIX_PATH)
    set(OPTIX_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0" CACHE PATH "OptiX SDK path")
endif()

if(NOT EXISTS "${OPTIX_PATH}/include/optix.h")
    message(FATAL_ERROR "OptiX SDK not found at ${OPTIX_PATH}. Set OPTIX_PATH correctly.")
endif()

message(STATUS "Using OptiX SDK: ${OPTIX_PATH}")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
    ${OPTIX_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compile CUDA to PTX
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/optix_benchmark.ptx
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
        -ptx
        -I${OPTIX_PATH}/include
        -I${CMAKE_CURRENT_SOURCE_DIR}
        --use_fast_math
        -O3
        -o ${CMAKE_CURRENT_BINARY_DIR}/optix_benchmark.ptx
        ${CMAKE_CURRENT_SOURCE_DIR}/optix_benchmark.cu
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/optix_benchmark.cu
    COMMENT "Compiling CUDA to PTX"
)

add_custom_target(ptx_compile ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/optix_benchmark.ptx)

# Main executable
add_executable(optix_benchmark optix_benchmark.cpp)

target_link_libraries(optix_benchmark
    CUDA::cuda_driver
    CUDA::cudart
)

add_dependencies(optix_benchmark ptx_compile)

# Copy PTX to output directory
add_custom_command(TARGET optix_benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/optix_benchmark.ptx
        $<TARGET_FILE_DIR:optix_benchmark>/optix_benchmark.ptx
    COMMENT "Copying PTX to output directory"
)

# Install
install(TARGETS optix_benchmark DESTINATION bin)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/optix_benchmark.ptx DESTINATION bin)
