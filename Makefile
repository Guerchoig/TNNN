# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.10/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.10/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sasha/source/tnnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sasha/source/tnnn

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "No interactive CMake dialog available..."
	/usr/local/lib/python3.10/dist-packages/cmake/data/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Running CMake to regenerate build system..."
	/usr/local/lib/python3.10/dist-packages/cmake/data/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/sasha/source/tnnn/CMakeFiles /home/sasha/source/tnnn//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/sasha/source/tnnn/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named test

# Build rule for target.
test: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 test
.PHONY : test

# fast build rule for target.
test/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/build
.PHONY : test/fast

#=============================================================================
# Target rules for targets named TNN

# Build rule for target.
TNN: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 TNN
.PHONY : TNN

# fast build rule for target.
TNN/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/TNN.dir/build.make CMakeFiles/TNN.dir/build
.PHONY : TNN/fast

#=============================================================================
# Target rules for targets named atomic_queue

# Build rule for target.
atomic_queue: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 atomic_queue
.PHONY : atomic_queue

# fast build rule for target.
atomic_queue/fast:
	$(MAKE) $(MAKESILENT) -f atomic_queue/include/CMakeFiles/atomic_queue.dir/build.make atomic_queue/include/CMakeFiles/atomic_queue.dir/build
.PHONY : atomic_queue/fast

src/brain.o: src/brain.cpp.o
.PHONY : src/brain.o

# target to build an object file
src/brain.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/TNN.dir/build.make CMakeFiles/TNN.dir/src/brain.cpp.o
.PHONY : src/brain.cpp.o

src/brain.i: src/brain.cpp.i
.PHONY : src/brain.i

# target to preprocess a source file
src/brain.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/TNN.dir/build.make CMakeFiles/TNN.dir/src/brain.cpp.i
.PHONY : src/brain.cpp.i

src/brain.s: src/brain.cpp.s
.PHONY : src/brain.s

# target to generate assembly for a file
src/brain.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/TNN.dir/build.make CMakeFiles/TNN.dir/src/brain.cpp.s
.PHONY : src/brain.cpp.s

src/input_output.o: src/input_output.cpp.o
.PHONY : src/input_output.o

# target to build an object file
src/input_output.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/TNN.dir/build.make CMakeFiles/TNN.dir/src/input_output.cpp.o
.PHONY : src/input_output.cpp.o

src/input_output.i: src/input_output.cpp.i
.PHONY : src/input_output.i

# target to preprocess a source file
src/input_output.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/TNN.dir/build.make CMakeFiles/TNN.dir/src/input_output.cpp.i
.PHONY : src/input_output.cpp.i

src/input_output.s: src/input_output.cpp.s
.PHONY : src/input_output.s

# target to generate assembly for a file
src/input_output.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/TNN.dir/build.make CMakeFiles/TNN.dir/src/input_output.cpp.s
.PHONY : src/input_output.cpp.s

src/mnist_set.o: src/mnist_set.cpp.o
.PHONY : src/mnist_set.o

# target to build an object file
src/mnist_set.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/TNN.dir/build.make CMakeFiles/TNN.dir/src/mnist_set.cpp.o
.PHONY : src/mnist_set.cpp.o

src/mnist_set.i: src/mnist_set.cpp.i
.PHONY : src/mnist_set.i

# target to preprocess a source file
src/mnist_set.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/TNN.dir/build.make CMakeFiles/TNN.dir/src/mnist_set.cpp.i
.PHONY : src/mnist_set.cpp.i

src/mnist_set.s: src/mnist_set.cpp.s
.PHONY : src/mnist_set.s

# target to generate assembly for a file
src/mnist_set.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/TNN.dir/build.make CMakeFiles/TNN.dir/src/mnist_set.cpp.s
.PHONY : src/mnist_set.cpp.s

src/test.o: src/test.cpp.o
.PHONY : src/test.o

# target to build an object file
src/test.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/src/test.cpp.o
.PHONY : src/test.cpp.o

src/test.i: src/test.cpp.i
.PHONY : src/test.i

# target to preprocess a source file
src/test.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/src/test.cpp.i
.PHONY : src/test.cpp.i

src/test.s: src/test.cpp.s
.PHONY : src/test.s

# target to generate assembly for a file
src/test.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/src/test.cpp.s
.PHONY : src/test.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... TNN"
	@echo "... atomic_queue"
	@echo "... test"
	@echo "... src/brain.o"
	@echo "... src/brain.i"
	@echo "... src/brain.s"
	@echo "... src/input_output.o"
	@echo "... src/input_output.i"
	@echo "... src/input_output.s"
	@echo "... src/mnist_set.o"
	@echo "... src/mnist_set.i"
	@echo "... src/mnist_set.s"
	@echo "... src/test.o"
	@echo "... src/test.i"
	@echo "... src/test.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

