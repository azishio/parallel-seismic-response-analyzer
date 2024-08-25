cmake_minimum_required(VERSION 3.29)
project(untitled)

set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

add_executable(my_pg main.cpp)

target_link_libraries(my_pg ${MPI_LIBRARIES})
