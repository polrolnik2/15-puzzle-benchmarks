add_library(astarmodel
    a-star.cpp
    distance.cpp
    model.cpp
    state.cpp
)

target_include_directories(astarmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(astarmodel PUBLIC cxx_std_20)
