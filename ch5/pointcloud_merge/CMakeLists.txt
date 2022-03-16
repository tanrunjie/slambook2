cmake_minimum_required(VERSION 3.0)

project(join)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)


include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(join join.cpp)


target_link_libraries(join ${OpenCV_LIBS} ${Pangolin_LIBRARIES} Sophus::Sophus)