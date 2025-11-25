cmake_minimum_required(VERSION 3.14)

add_executable(benchmark-a-star
    benchmark-a-star.cpp
)

target_link_libraries(benchmark-a-star PRIVATE astarmodel)

target_link_libraries(benchmark-a-star PUBLIC 15-puzzle-state)

# headers from A-star-model are needed
target_include_directories(benchmark-a-star PRIVATE ${CMAKE_SOURCE_DIR}/A-star-model)

# Optional: set C++ standard (inherited from root) and any compile options
target_compile_features(benchmark-a-star PRIVATE cxx_std_20)

install(TARGETS benchmark-a-star RUNTIME DESTINATION bin)
