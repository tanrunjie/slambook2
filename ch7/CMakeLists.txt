cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SEE")
set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
find_package(G2O REQUIRED)
find_package(FMT REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
)

add_executable(feature_mapping feature_mapping.cpp)
target_link_libraries(feature_mapping ${OpenCV_LIBS})

add_executable(pose2d2d pose2d2d.cpp)
target_link_libraries(pose2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose3d2d pose3d2d.cpp)
target_link_libraries(pose3d2d ${OpenCV_LIBS})

add_executable(pose3d3d pose3d3d.cpp)
target_link_libraries(pose3d3d ${OpenCV_LIBS} g2o_core g2o_stuff fmt::fmt)