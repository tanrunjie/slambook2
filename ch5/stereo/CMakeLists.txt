project(stereo)

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)

add_executable(stereo stereo.cpp)
target_link_libraries(stereo ${OpenCV_LIBS} ${Pangolin_LIBRARIES})