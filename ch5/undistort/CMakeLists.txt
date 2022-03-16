project(undistort)

find_package(OpenCV REQUIRED)

add_executable(undistort undistortImage.cpp)
target_link_libraries(undistort ${OpenCV_LIBS})