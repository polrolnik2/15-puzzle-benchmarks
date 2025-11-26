add_library(astarmodel
    15-puzzle-a-star-solver.cpp
)

target_include_directories(astarmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/dependencies/a-star/cpp)

# Link against the standalone state library
target_link_libraries(astarmodel PUBLIC 15-puzzle-state)

target_compile_features(astarmodel PUBLIC cxx_std_20)

# Model benchmark executable
add_executable(benchmark_astar
    benchmark.cpp
)

target_link_libraries(benchmark_astar PRIVATE astarmodel 15-puzzle-state)
target_compile_features(benchmark_astar PRIVATE cxx_std_20)
install(TARGETS benchmark_astar RUNTIME DESTINATION bin)

# Propagate the USE_FSA_MEMORY option to the vendor code so it can be toggled
if (DEFINED USE_FSA_MEMORY)
    target_compile_definitions(astarmodel PUBLIC USE_FSA_MEMORY=${USE_FSA_MEMORY})
else()
    # default to ON if the option isn't set
    target_compile_definitions(astarmodel PUBLIC USE_FSA_MEMORY=0)
endif()
