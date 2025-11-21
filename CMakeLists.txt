cmake_minimum_required(VERSION 3.14)
project(ok-projekt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable the vendor fixed-size allocator (can be turned off for debugging)
option(USE_FSA_MEMORY "Enable fixed-size allocator in A* vendor (stlastar)" ON)


add_subdirectory(A-star-model)
add_subdirectory(benchmarks)

enable_testing()
add_subdirectory(tests)
