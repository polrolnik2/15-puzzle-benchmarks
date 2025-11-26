add_library(weightedastarmodel
    15-puzzle-weighted-a-star-solver.cpp
)

target_include_directories(weightedastarmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Link against the standalone state library
target_link_libraries(weightedastarmodel PUBLIC 15-puzzle-state)

target_compile_features(weightedastarmodel PUBLIC cxx_std_20)

# Model benchmark executable
add_executable(benchmark_weightedastar
    benchmark.cpp
)

target_link_libraries(benchmark_weightedastar PRIVATE weightedastarmodel 15-puzzle-state)
target_compile_features(benchmark_weightedastar PRIVATE cxx_std_20)
install(TARGETS benchmark_weightedastar RUNTIME DESTINATION bin)