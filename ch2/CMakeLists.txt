cmake_minimum_required(VERSION 2.8)

project(HelloSLAM)


set(CMAKE_BUILD_TYPE "Debug")

add_executable(helloSLAM helloSLAM.cpp) # 编译执行

add_library(hello libHelloSLAM.cpp)  # 编译静态库

add_library(hello_shared SHARED libHelloSLAM.cpp)  # 编译动态库

# 将库hello和共享库hello_shared安装到/usr/local/lib下
INSTALL(TARGETS hello hello_shared
LIBRARY DESTINATION lib
ARCHIVE DESTINATION lib)

INSTALL(FILES libHelloSLAM.h DESTINATION include/libHelloSLAM)


add_executable(useHello useHello.cpp)
target_link_libraries(useHello hello_shared) 