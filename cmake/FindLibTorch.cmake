# FindLibTorch.cmake - Simplified LibTorch detection for crlGRU
# 
# This module finds LibTorch installation using standard CMake mechanisms
#
# Variables set by this module:
#   LIBTORCH_FOUND          - True if LibTorch is found
#   LIBTORCH_LIBRARIES      - LibTorch libraries
#   TORCH_VERSION           - PyTorch version

cmake_minimum_required(VERSION 3.16)

# Standard LibTorch paths (in order of preference)
set(LIBTORCH_SEARCH_PATHS
    "$ENV{HOME}/local/libtorch"
    "/opt/homebrew/opt/pytorch"
    "/usr/local/opt/pytorch"
)

# Function to display LibTorch installation instructions
function(display_libtorch_installation_instructions)
    message(STATUS "")
    message(STATUS "LibTorch NOT FOUND - Install with:")
    message(STATUS "  brew install pytorch  # macOS")
    message(STATUS "  Or download from: https://pytorch.org/")
    message(STATUS "  Or set: -DCRLGRU_DISABLE_TORCH=ON")
    message(STATUS "")
endfunction()

# Function to add LibTorch paths to CMAKE_PREFIX_PATH
function(setup_libtorch_paths)
    foreach(path ${LIBTORCH_SEARCH_PATHS})
        if(EXISTS "${path}")
            list(APPEND CMAKE_PREFIX_PATH "${path}")
            message(STATUS "Added LibTorch path: ${path}")
        endif()
    endforeach()
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
endfunction()

# Simplified LibTorch detection
function(find_libtorch_with_guidance PROJECT_DISABLE_OPTION)
    # Skip if explicitly disabled
    if(${PROJECT_DISABLE_OPTION})
        message(STATUS "LibTorch support disabled")
        set(LIBTORCH_FOUND FALSE PARENT_SCOPE)
        return()
    endif()
    
    # Setup search paths
    setup_libtorch_paths()
    
    # Try to find LibTorch
    find_package(Torch QUIET)
    
    if(Torch_FOUND)
        message(STATUS "✓ LibTorch found")
        message(STATUS "  Version: ${TORCH_VERSION}")
        
        # Set variables
        set(LIBTORCH_FOUND TRUE PARENT_SCOPE)
        set(LIBTORCH_LIBRARIES ${TORCH_LIBRARIES} PARENT_SCOPE)
        set(TORCH_VERSION ${TORCH_VERSION} PARENT_SCOPE)
        
    else()
        display_libtorch_installation_instructions()
        set(LIBTORCH_FOUND FALSE PARENT_SCOPE)
    endif()
endfunction()

# Export the main function
macro(find_libtorch_for_project PROJECT_PREFIX)
    find_libtorch_with_guidance(${PROJECT_PREFIX}_DISABLE_TORCH)
endmacro()
