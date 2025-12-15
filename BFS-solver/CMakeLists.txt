add_library(bfsmodel SHARED
    15-puzzle-bfs-solver.cpp
    bfs_api.cpp
)

target_include_directories(bfsmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link against the standalone state library
target_link_libraries(bfsmodel PUBLIC 15-puzzle-state)

target_compile_features(bfsmodel PUBLIC cxx_std_20)

set_target_properties(bfsmodel PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Model benchmark executable
add_executable(benchmark_bfs
    benchmark.cpp
)

target_link_libraries(benchmark_bfs PRIVATE bfsmodel 15-puzzle-state)
target_compile_features(benchmark_bfs PRIVATE cxx_std_20)
install(TARGETS benchmark_bfs RUNTIME DESTINATION bin)
