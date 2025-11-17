cmake_minimum_required(VERSION 3.14)
include(FetchContent)

if (NOT TARGET gtest AND NOT TARGET gtest_main)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
  )
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(test_state test_state.cpp)
target_link_libraries(test_state PRIVATE astarmodel gtest_main)
target_include_directories(test_state PRIVATE ${CMAKE_SOURCE_DIR}/A-star-model)

add_test(NAME state_tests COMMAND test_state)

add_executable(test_available_moves test_get_available_moves.cpp)
target_link_libraries(test_available_moves PRIVATE astarmodel gtest_main)
target_include_directories(test_available_moves PRIVATE ${CMAKE_SOURCE_DIR}/A-star-model)
add_test(NAME available_moves_tests COMMAND test_available_moves)
