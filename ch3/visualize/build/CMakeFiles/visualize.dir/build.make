# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tan/reading/slambook2/ch3/visualize

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tan/reading/slambook2/ch3/visualize/build

# Include any dependencies generated for this target.
include CMakeFiles/visualize.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/visualize.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/visualize.dir/flags.make

CMakeFiles/visualize.dir/visualize.cpp.o: CMakeFiles/visualize.dir/flags.make
CMakeFiles/visualize.dir/visualize.cpp.o: ../visualize.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tan/reading/slambook2/ch3/visualize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/visualize.dir/visualize.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/visualize.dir/visualize.cpp.o -c /home/tan/reading/slambook2/ch3/visualize/visualize.cpp

CMakeFiles/visualize.dir/visualize.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/visualize.dir/visualize.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tan/reading/slambook2/ch3/visualize/visualize.cpp > CMakeFiles/visualize.dir/visualize.cpp.i

CMakeFiles/visualize.dir/visualize.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/visualize.dir/visualize.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tan/reading/slambook2/ch3/visualize/visualize.cpp -o CMakeFiles/visualize.dir/visualize.cpp.s

# Object files for target visualize
visualize_OBJECTS = \
"CMakeFiles/visualize.dir/visualize.cpp.o"

# External object files for target visualize
visualize_EXTERNAL_OBJECTS =

visualize: CMakeFiles/visualize.dir/visualize.cpp.o
visualize: CMakeFiles/visualize.dir/build.make
visualize: /usr/local/lib/libpango_glgeometry.so
visualize: /usr/local/lib/libpango_plot.so
visualize: /usr/local/lib/libpango_python.so
visualize: /usr/local/lib/libpango_scene.so
visualize: /usr/local/lib/libpango_tools.so
visualize: /usr/local/lib/libpango_video.so
visualize: /usr/local/lib/libpango_geometry.so
visualize: /usr/local/lib/libtinyobj.so
visualize: /usr/local/lib/libpango_display.so
visualize: /usr/local/lib/libpango_vars.so
visualize: /usr/local/lib/libpango_windowing.so
visualize: /usr/local/lib/libpango_opengl.so
visualize: /usr/lib/x86_64-linux-gnu/libGLEW.so
visualize: /usr/lib/x86_64-linux-gnu/libOpenGL.so
visualize: /usr/lib/x86_64-linux-gnu/libGLX.so
visualize: /usr/lib/x86_64-linux-gnu/libGLU.so
visualize: /usr/local/lib/libpango_image.so
visualize: /usr/local/lib/libpango_packetstream.so
visualize: /usr/local/lib/libpango_core.so
visualize: CMakeFiles/visualize.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tan/reading/slambook2/ch3/visualize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable visualize"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/visualize.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/visualize.dir/build: visualize

.PHONY : CMakeFiles/visualize.dir/build

CMakeFiles/visualize.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/visualize.dir/cmake_clean.cmake
.PHONY : CMakeFiles/visualize.dir/clean

CMakeFiles/visualize.dir/depend:
	cd /home/tan/reading/slambook2/ch3/visualize/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tan/reading/slambook2/ch3/visualize /home/tan/reading/slambook2/ch3/visualize /home/tan/reading/slambook2/ch3/visualize/build /home/tan/reading/slambook2/ch3/visualize/build /home/tan/reading/slambook2/ch3/visualize/build/CMakeFiles/visualize.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/visualize.dir/depend

