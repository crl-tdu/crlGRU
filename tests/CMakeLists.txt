cmake_minimum_required(VERSION 3.18)

# Find required packages
find_package(Torch REQUIRED)

# Test executables
add_executable(test_fep_gru_cell test_fep_gru_cell.cpp)
target_link_libraries(test_fep_gru_cell crlGRU ${TORCH_LIBRARIES})

add_executable(test_fep_gru_network test_fep_gru_network.cpp)
target_link_libraries(test_fep_gru_network crlGRU ${TORCH_LIBRARIES})

add_executable(test_spsa_optimizer test_spsa_optimizer.cpp)
target_link_libraries(test_spsa_optimizer crlGRU ${TORCH_LIBRARIES})

add_executable(test_utils test_utils.cpp)
target_link_libraries(test_utils crlGRU ${TORCH_LIBRARIES})

# Examples (moved from examples directory)
add_subdirectory(examples)

# Enable testing
enable_testing()

# Add tests
add_test(NAME FEPGRUCell COMMAND test_fep_gru_cell)
add_test(NAME FEPGRUNetwork COMMAND test_fep_gru_network)
add_test(NAME SPSAOptimizer COMMAND test_spsa_optimizer)
add_test(NAME Utils COMMAND test_utils)

