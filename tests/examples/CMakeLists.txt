cmake_minimum_required(VERSION 3.18)

# Find required packages
find_package(Torch REQUIRED)

# Swarm coordination example
add_executable(swarm_coordination_example swarm_coordination_example.cpp)
target_link_libraries(swarm_coordination_example crlGRU ${TORCH_LIBRARIES})
set_target_properties(swarm_coordination_example PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Simple prediction example
add_executable(simple_prediction_example 
    simple_prediction_example.cpp
)
target_link_libraries(simple_prediction_example crlGRU ${TORCH_LIBRARIES})
set_target_properties(simple_prediction_example PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Installation
install(TARGETS swarm_coordination_example simple_prediction_example 
        DESTINATION bin)
