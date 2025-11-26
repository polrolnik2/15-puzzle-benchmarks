cmake_minimum_required(VERSION 3.14)

add_executable(benchmark-a-star
    benchmark-a-star.cpp
)

target_include_directories(benchmark-a-star PRIVATE ${CMAKE_SOURCE_DIR}/generate_sample_state.hpp)

target_link_libraries(benchmark-a-star PRIVATE astarmodel)

target_link_libraries(benchmark-a-star PUBLIC bfsmodel)

target_link_libraries(benchmark-a-star PUBLIC 15-puzzle-state)

# headers from A-star-model are needed
# solver headers are provided transitively by linking the solver libraries

# Optional: set C++ standard (inherited from root) and any compile options
target_compile_features(benchmark-a-star PRIVATE cxx_std_20)

if(DEFINED BFS_GENERATION)
    target_compile_definitions(15-puzzle-state PUBLIC BFS_SAMPLE_GENERATION=1)
elseif(DEFINED RANDOM_WALK_GENERATION)
    target_compile_definitions(15-puzzle-state PUBLIC RANDOM_WALK_GENERATION=1)
else()
    target_compile_definitions(15-puzzle-state PUBLIC RANDOM_WALK_GENERATION=1)
endif()

if(DEFINED A_STAR_SOLVER)
    target_compile_definitions(benchmark-a-star PUBLIC A_STAR_SOLVER=1)
elseif(DEFINED BFS_SOLVER)
    target_compile_definitions(benchmark-a-star PUBLIC BFS_SOLVER=1)
else()
    target_compile_definitions(benchmark-a-star PUBLIC A_STAR_SOLVER=1)
endif()

install(TARGETS benchmark-a-star RUNTIME DESTINATION bin)
