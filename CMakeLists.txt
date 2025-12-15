cmake_minimum_required(VERSION 3.18)

# Option to enable/disable CUDA support
option(USE_CUDA "Enable CUDA support for ACO solver" ON)

if(USE_CUDA)
    project(ok-projekt LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    project(ok-projekt LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable the vendor fixed-size allocator (can be turned off for debugging)
option(USE_FSA_MEMORY "Enable fixed-size allocator in A* vendor (stlastar)" ON)


add_subdirectory(15-puzzle-state)
add_subdirectory(A-star-model)
add_subdirectory(BFS-solver)
add_subdirectory(weighted-A-star-model)

if(USE_CUDA)
    add_subdirectory(ACO-model)
else()
    message(STATUS "CUDA disabled - ACO-model will not be built")
endif()
add_subdirectory(data-generation)

enable_testing()
add_subdirectory(tests)

# Doxygen documentation target (optional)
find_package(Doxygen)
if(DOXYGEN_FOUND)
	add_custom_target(doc
		COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		COMMENT "Generating API documentation with Doxygen"
		VERBATIM
	)
else()
	message(STATUS "Doxygen not found - 'doc' target will not be available.")
endif()
