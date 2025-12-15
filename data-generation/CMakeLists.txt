cmake_minimum_required(VERSION 3.14)

# Data generation executables
add_executable(generate_sample_bfs
    generate_sample_bfs.cpp
)
target_include_directories(generate_sample_bfs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/15-puzzle-state)
target_link_libraries(generate_sample_bfs PRIVATE 15-puzzle-state)
target_compile_features(generate_sample_bfs PRIVATE cxx_std_20)
install(TARGETS generate_sample_bfs RUNTIME DESTINATION bin)

add_executable(generate_sample_random_walk
    generate_sample_random_walk.cpp
)
target_include_directories(generate_sample_random_walk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/15-puzzle-state)
target_link_libraries(generate_sample_random_walk PRIVATE 15-puzzle-state)
target_compile_features(generate_sample_random_walk PRIVATE cxx_std_20)
install(TARGETS generate_sample_random_walk RUNTIME DESTINATION bin)

# Shared library for Python/ctypes integration
add_library(datagen SHARED
    c_api.cpp
)
target_include_directories(datagen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/15-puzzle-state)
target_link_libraries(datagen PUBLIC 15-puzzle-state)
set_target_properties(datagen PROPERTIES POSITION_INDEPENDENT_CODE ON)
