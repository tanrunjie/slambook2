cmake_minimum_required(VERSION 2.8)

project(libHelloSLAM)

set(CMAKE_BUILD_TYPE "Debug")

include_directories(/usr/local/include/libHelloSLAM)

add_executable(useHello useHello.cpp)

target_link_libraries(useHello /usr/local/lib/libhello_shared.so)