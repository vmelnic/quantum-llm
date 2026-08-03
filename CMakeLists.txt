cmake_minimum_required(VERSION 3.25)

project(expert_runtime
    VERSION 0.1.0
    DESCRIPTION "Expert Pack compiler and Windows/CUDA runtime"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EXPERT_BUILD_TESTS "Build contract and unit tests" ON)
option(EXPERT_ENABLE_CUDA "Enable the CUDA backend when available" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ExpertWarnings)

add_subdirectory(core)

# P1+ components own their build files. Keeping these conditional lets P0 build
# independently while preserving one root entry point for compiler and runtime.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/compiler/CMakeLists.txt")
    add_subdirectory(compiler)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/runtime/CMakeLists.txt")
    add_subdirectory(runtime)
endif()

if(EXPERT_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
