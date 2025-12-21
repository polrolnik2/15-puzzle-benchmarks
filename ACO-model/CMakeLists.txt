cmake_minimum_required(VERSION 3.18)

if(USE_CUDA)
    project(ACO-model LANGUAGES CXX CUDA)
    
    # Set CUDA architectures (adjust based on your GPU)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
else()
    message(FATAL_ERROR "ACO-model requires CUDA. Enable with -DUSE_CUDA=ON or exclude this directory.")
endif()

# ACO solver library 
add_library(acomodel SHARED
    15-puzzle-aco-solver.cu
    aco_api.cpp
)

target_include_directories(acomodel PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/15-puzzle-state
)

# Link against CUDA state library for device code
# and host state library for host code conversion functions
target_link_libraries(acomodel PUBLIC 15-puzzle-state-cuda 15-puzzle-state)

target_compile_features(acomodel PUBLIC cxx_std_20)

# Set CUDA compiler flags
set_target_properties(acomodel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# ACO benchmark executable
add_executable(benchmark_aco
    benchmark.cpp
)

target_link_libraries(benchmark_aco PRIVATE acomodel 15-puzzle-state)
target_compile_features(benchmark_aco PRIVATE cxx_std_20)
install(TARGETS benchmark_aco RUNTIME DESTINATION bin)
