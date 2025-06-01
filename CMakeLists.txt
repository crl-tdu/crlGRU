cmake_minimum_required(VERSION 3.18)
project(crlGRU VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set RPATH for installed library
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Include custom cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindLibTorch)

# Build options
option(CRLGRU_DISABLE_TORCH "Disable LibTorch support" OFF)

# Find required packages
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
else()
    message(WARNING "OpenMP not found - building without OpenMP support")
endif()

# LibTorch detection with guidance
find_libtorch_for_project(CRLGRU)

if(LIBTORCH_FOUND)
    message(STATUS "Building crlGRU with LibTorch support")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIBTORCH_CXX_FLAGS}")
else()
    message(STATUS "Building crlGRU without LibTorch support")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Core library
file(GLOB_RECURSE CORE_SOURCES 
    "src/core/*.cpp"
    "src/som/*.cpp"
    "src/imitation/*.cpp"
)

add_library(crlGRU SHARED ${CORE_SOURCES})

# Link libraries conditionally
if(LIBTORCH_FOUND)
    target_link_libraries(crlGRU ${LIBTORCH_LIBRARIES})
    target_compile_definitions(crlGRU PUBLIC CRLGRU_HAS_TORCH)
    
    # Set RPATH for runtime library loading
    set_target_properties(crlGRU PROPERTIES
        INSTALL_RPATH "$ENV{HOME}/local/lib;$ENV{HOME}/local/libtorch/lib"
        BUILD_RPATH "$ENV{HOME}/local/lib;$ENV{HOME}/local/libtorch/lib"
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(crlGRU OpenMP::OpenMP_CXX)
endif()

# Tests (includes examples)
option(BUILD_TESTS "Build test programs and examples" ON)
if(BUILD_TESTS)
    # Enable CTest for running tests
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS crlGRU DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
