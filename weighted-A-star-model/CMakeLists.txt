add_library(weightedastarmodel SHARED
    15-puzzle-weighted-a-star-solver.cpp
    15_puzzle_api.cpp
)

target_include_directories(weightedastarmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/dependencies/a-star/cpp)
# Link against the standalone state library
target_link_libraries(weightedastarmodel PUBLIC 15-puzzle-state)

target_compile_features(weightedastarmodel PUBLIC cxx_std_20)

set_target_properties(weightedastarmodel PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Propagate the USE_FSA_MEMORY option to the vendor code so it can be toggled
if (DEFINED USE_FSA_MEMORY)
    target_compile_definitions(weightedastarmodel PUBLIC USE_FSA_MEMORY=${USE_FSA_MEMORY})
else()
    target_compile_definitions(weightedastarmodel PUBLIC USE_FSA_MEMORY=0)
endif()

# Model benchmark executable
add_executable(benchmark_weightedastar
    benchmark.cpp
)

target_link_libraries(benchmark_weightedastar PRIVATE weightedastarmodel 15-puzzle-state)
target_compile_features(benchmark_weightedastar PRIVATE cxx_std_20)
install(TARGETS benchmark_weightedastar RUNTIME DESTINATION bin)