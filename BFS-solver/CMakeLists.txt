add_library(bfsmodel
    15-puzzle-bfs-solver.cpp
)

target_include_directories(bfsmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link against the standalone state library
target_link_libraries(bfsmodel PUBLIC 15-puzzle-state)

target_compile_features(bfsmodel PUBLIC cxx_std_20)
