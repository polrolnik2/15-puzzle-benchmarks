add_library(wastarmodel
    15-puzzle-weighted-a-star-solver.cpp
)

target_include_directories(wastarmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/dependencies/a-star/cpp)
# Link against the standalone state library
target_link_libraries(wastarmodel PUBLIC 15-puzzle-state)

target_compile_features(wastarmodel PUBLIC cxx_std_20)
# Propagate the USE_FSA_MEMORY option to the vendor code so it can be toggled
if (DEFINED USE_FSA_MEMORY)
    target_compile_definitions(wastarmodel PUBLIC USE_FSA_MEMORY=${USE_FSA_MEMORY})
else()
    # default to ON if the option isn't set
    target_compile_definitions(wastarmodel PUBLIC USE_FSA_MEMORY=0)
endif()
