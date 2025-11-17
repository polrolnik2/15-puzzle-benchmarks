add_library(astarmodel
    distance.cpp
    state.cpp
    15-puzzle-a-star-solver.cpp
)

target_include_directories(astarmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/dependencies/a-star/cpp)

target_compile_features(astarmodel PUBLIC cxx_std_20)
