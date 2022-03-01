cmake_minimum_required(VERSION 2.8)

project(visualize)

find_package(Pangolin)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(visualize visualize.cpp) # 编译执行
target_link_libraries(visualize ${Pangolin_LIBRARIES})
