add_library(astarmodel
    distance.cpp
    state.cpp
    15-puzzle-a-star-solver.cpp
)

target_include_directories(astarmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/dependencies/a-star/cpp)

target_compile_features(astarmodel PUBLIC cxx_std_20)

# Propagate the USE_FSA_MEMORY option to the vendor code so it can be toggled
if (DEFINED USE_FSA_MEMORY)
    target_compile_definitions(astarmodel PUBLIC USE_FSA_MEMORY=${USE_FSA_MEMORY})
else()
    # default to ON if the option isn't set
    target_compile_definitions(astarmodel PUBLIC USE_FSA_MEMORY=0)
endif()
