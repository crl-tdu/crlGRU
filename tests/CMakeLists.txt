# Tests CMakeLists.txt
# Unified test suite for crlGRU library

# Create unified test executable
add_executable(test_crlgru test_crlgru.cpp)

# Link required libraries
target_link_libraries(test_crlgru crlGRU)

if(LIBTORCH_FOUND)
    target_link_libraries(test_crlgru ${LIBTORCH_LIBRARIES})
    target_compile_definitions(test_crlgru PUBLIC CRLGRU_HAS_TORCH)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_crlgru OpenMP::OpenMP_CXX)
endif()

# Create embodied cell test executable
add_executable(test_embodied_cell test_embodied_cell.cpp)
target_link_libraries(test_embodied_cell crlGRU)

if(LIBTORCH_FOUND)
    target_link_libraries(test_embodied_cell ${LIBTORCH_LIBRARIES})
    target_compile_definitions(test_embodied_cell PUBLIC CRLGRU_HAS_TORCH)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_embodied_cell OpenMP::OpenMP_CXX)
endif()

# Add tests to CTest
add_test(NAME crlgru_unified_test COMMAND test_crlgru)
add_test(NAME crlgru_embodied_test COMMAND test_embodied_cell)

# Set test properties
set_tests_properties(crlgru_unified_test PROPERTIES
    TIMEOUT 300
    PASS_REGULAR_EXPRESSION "All tests passed"
)

set_tests_properties(crlgru_embodied_test PROPERTIES
    TIMEOUT 300
    PASS_REGULAR_EXPRESSION "Embodied Tests Complete"
)

# Copy test to build directory with proper RPATH
if(LIBTORCH_FOUND)
    set_target_properties(test_crlgru PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    set_target_properties(test_embodied_cell PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Install test executables (optional)
install(TARGETS test_crlgru test_embodied_cell DESTINATION bin)
