cmake_minimum_required(VERSION 3.0)
project(useSophus)

find_package(Sophus REQUIRED)


add_executable(useSophus useSophus.cpp)

target_link_libraries(useSophus Sophus::Sophus)

