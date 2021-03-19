cmake_minimum_required(VERSION 3.0.0)
project(knn-scan)

add_executable(knn-scan main.cpp)

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native -O3")

add_subdirectory(test)
