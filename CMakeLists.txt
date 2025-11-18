cmake_minimum_required(VERSION 3.14)
project(ok-projekt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(A-star-model)

enable_testing()
add_subdirectory(tests)
