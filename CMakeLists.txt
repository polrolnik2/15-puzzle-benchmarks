cmake_minimum_required(VERSION 3.14)
project(ok-projekt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable the vendor fixed-size allocator (can be turned off for debugging)
option(USE_FSA_MEMORY "Enable fixed-size allocator in A* vendor (stlastar)" ON)


add_subdirectory(15-puzzle-state)
add_subdirectory(A-star-model)
add_subdirectory(BFS-solver)
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
