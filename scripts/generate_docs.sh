#!/bin/bash
# Documentation generation script for crlGRU
# Generates comprehensive documentation using Doxygen

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$PROJECT_ROOT/docs"
DOXYGEN_OUTPUT="$DOCS_DIR/doxygen"

echo "=== crlGRU Documentation Generator ==="
echo "Project root: $PROJECT_ROOT"

# Check if Doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo "Error: Doxygen not found. Please install Doxygen."
    echo "  macOS: brew install doxygen"
    echo "  Ubuntu: sudo apt-get install doxygen"
    exit 1
fi

# Check if Graphviz is installed (for diagrams)
if ! command -v dot &> /dev/null; then
    echo "Warning: Graphviz not found. Class diagrams will not be generated."
    echo "  macOS: brew install graphviz"
    echo "  Ubuntu: sudo apt-get install graphviz"
fi

# Create docs directory if it doesn't exist
mkdir -p "$DOCS_DIR"
mkdir -p "$DOXYGEN_OUTPUT"

# Navigate to project root
cd "$PROJECT_ROOT"

echo "Generating documentation..."

# Run Doxygen
if doxygen Doxyfile; then
    echo "✓ Doxygen documentation generated successfully"
    echo "  HTML output: $DOXYGEN_OUTPUT/html/index.html"
    echo "  XML output: $DOXYGEN_OUTPUT/xml/"
else
    echo "✗ Doxygen documentation generation failed"
    exit 1
fi

# Generate additional documentation files
echo ""
echo "Generating additional documentation..."

# Create API overview
cat > "$DOCS_DIR/API_OVERVIEW.md" << 'EOF'
# crlGRU API Overview

## Core Components

### FEPGRUCell
- **Purpose**: Free Energy Principle-based GRU cell
- **Key Features**: Predictive coding, meta-evaluation, SOM extraction
- **Usage**: `#include "crlgru/core/fep_gru_cell.hpp"`

### FEPGRUNetwork  
- **Purpose**: Multi-layer FEP-GRU network
- **Key Features**: Agent management, parameter sharing, collective energy
- **Usage**: `#include "crlgru/core/fep_gru_network.hpp"`

### EmbodiedFEPGRUCell
- **Purpose**: Embodied AI with physical constraints
- **Key Features**: Physical simulation, sensor modeling, polar coordinates
- **Usage**: `#include "crlgru/core/embodied_fep_gru_cell.hpp"`

### PolarSpatialAttention
- **Purpose**: Polar coordinate spatial attention mechanism
- **Key Features**: Distance rings, angle sectors, adaptive resolution
- **Usage**: `#include "crlgru/core/polar_spatial_attention.hpp"`

## Utilities

### Math Utilities
- Spatial transformations
- Tensor operations
- Numerical stability functions

### Optimizers
- SPSA (Simultaneous Perturbation Stochastic Approximation)
- Meta-evaluation optimization
- Adaptive weight adjustment

## Configuration
- All configuration structures in `crlgru/utils/config_types.hpp`
- Default configurations provided
- Runtime parameter validation

## Examples
See `tests/` directory for comprehensive usage examples.
EOF

# Create build instructions
cat > "$DOCS_DIR/BUILD_INSTRUCTIONS.md" << 'EOF'
# Build Instructions

## Prerequisites
- CMake 3.18+
- C++17 compatible compiler
- LibTorch (PyTorch C++ API)
- OpenMP (optional)

## Quick Start
```bash
# Clone repository
git clone <repository-url>
cd crlGRU

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build .

# Test
ctest --verbose
```

## CMake Options
- `CRLGRU_BUILD_SHARED=ON/OFF`: Build shared library
- `CRLGRU_BUILD_STATIC=ON/OFF`: Build static library  
- `CRLGRU_BUILD_TESTS=ON/OFF`: Build tests
- `CRLGRU_INSTALL=ON/OFF`: Enable installation
- `CRLGRU_DISABLE_TORCH=ON/OFF`: Disable LibTorch support

## As Submodule
```cmake
add_subdirectory(external/crlGRU)
target_link_libraries(your_target crlGRU::crlGRU)
```

## Installation
```bash
cmake --build . --target install
```
EOF

# Create troubleshooting guide
cat > "$DOCS_DIR/TROUBLESHOOTING.md" << 'EOF'
# Troubleshooting Guide

## Common Issues

### LibTorch Not Found
```
Solution: Install LibTorch or set CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH=/path/to/libtorch:$CMAKE_PREFIX_PATH
```

### Link Errors
```
Solution: Check RPATH settings
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

### Compilation Errors
```
Solution: Ensure C++17 support
export CXX=g++-9  # or appropriate compiler
```

### Test Failures
```
Solution: Run with verbose output
ctest --verbose --output-on-failure
```

## Debug Builds
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCRLGRU_BUILD_TESTS=ON
```

## Memory Issues
- Use AddressSanitizer: `-DCMAKE_CXX_FLAGS=-fsanitize=address`
- Use Valgrind: `valgrind --tool=memcheck ./test_program`

## Performance Profiling
- Use gperftools: Link with `-lprofiler`
- Use Intel VTune or perf for detailed analysis
EOF

echo "✓ Additional documentation generated"
echo ""

# Check if we can open the documentation
if command -v open &> /dev/null; then
    echo "Opening documentation..."
    open "$DOXYGEN_OUTPUT/html/index.html"
elif command -v xdg-open &> /dev/null; then
    echo "Opening documentation..."
    xdg-open "$DOXYGEN_OUTPUT/html/index.html"
else
    echo "Documentation ready at: file://$DOXYGEN_OUTPUT/html/index.html"
fi

echo ""
echo "=== Documentation Generation Complete ==="
echo "Files generated:"
echo "  - Doxygen HTML: $DOXYGEN_OUTPUT/html/"
echo "  - API Overview: $DOCS_DIR/API_OVERVIEW.md"
echo "  - Build Instructions: $DOCS_DIR/BUILD_INSTRUCTIONS.md"
echo "  - Troubleshooting: $DOCS_DIR/TROUBLESHOOTING.md"
echo ""
echo "To regenerate documentation, run: $0"