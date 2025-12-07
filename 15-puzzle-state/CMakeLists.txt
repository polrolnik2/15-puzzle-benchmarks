# Host-only C++ library (used by A*, BFS, benchmarks)
add_library(15-puzzle-state
    state.cpp
    distance.cpp
)

target_include_directories(15-puzzle-state PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/dependencies/a-star/cpp)

target_compile_features(15-puzzle-state PUBLIC cxx_std_20)

# Propagate the USE_FSA_MEMORY option to the vendor code so it can be toggled
if (DEFINED USE_FSA_MEMORY)
    target_compile_definitions(15-puzzle-state PUBLIC USE_FSA_MEMORY=${USE_FSA_MEMORY})
else()
    # default to ON if the option isn't set
    target_compile_definitions(15-puzzle-state PUBLIC USE_FSA_MEMORY=0)
endif()

# CUDA-compatible library (header-only for device code; used by GPU solvers)
add_library(15-puzzle-state-cuda INTERFACE)
target_include_directories(15-puzzle-state-cuda INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(15-puzzle-state-cuda INTERFACE cxx_std_20)
