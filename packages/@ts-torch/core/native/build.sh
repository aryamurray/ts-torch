#!/bin/bash
# Build script for ts-torch native library (Unix-like systems)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
LIBTORCH_PATH=""
INSTALL_PREFIX=""
CLEAN=false
BUILD_EXAMPLES=false
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --libtorch)
            LIBTORCH_PATH="$2"
            shift 2
            ;;
        --prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --examples)
            BUILD_EXAMPLES=true
            shift
            ;;
        --jobs|-j)
            JOBS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug           Build in Debug mode"
            echo "  --release         Build in Release mode (default)"
            echo "  --libtorch PATH   Path to LibTorch installation (required)"
            echo "  --prefix PATH     Installation prefix"
            echo "  --clean           Clean build directory before building"
            echo "  --examples        Build examples after library"
            echo "  --jobs N, -j N    Number of parallel jobs (default: auto)"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check for LibTorch path
if [ -z "$LIBTORCH_PATH" ]; then
    echo -e "${RED}Error: LibTorch path is required${NC}"
    echo "Please specify it with --libtorch /path/to/libtorch"
    echo "Or set LIBTORCH_PATH environment variable"
    exit 1
fi

if [ ! -d "$LIBTORCH_PATH" ]; then
    echo -e "${RED}Error: LibTorch directory not found: $LIBTORCH_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}=== Building ts-torch Native Library ===${NC}"
echo "Build type: $BUILD_TYPE"
echo "LibTorch path: $LIBTORCH_PATH"
echo "Parallel jobs: $JOBS"

# Clean if requested
if [ "$CLEAN" = true ] && [ -d "build" ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Configure CMake
echo -e "${YELLOW}Configuring CMake...${NC}"
CMAKE_ARGS=(
    -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if [ -n "$INSTALL_PREFIX" ]; then
    CMAKE_ARGS+=(-DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX")
fi

cmake "${CMAKE_ARGS[@]}" ..

# Build
echo -e "${YELLOW}Building...${NC}"
cmake --build . -j "$JOBS"

echo -e "${GREEN}Build successful!${NC}"
echo ""
echo "Library location: $(pwd)/libts_torch.*"

# Build examples if requested
if [ "$BUILD_EXAMPLES" = true ]; then
    echo ""
    echo -e "${YELLOW}Building examples...${NC}"

    if [ ! -d "../examples" ]; then
        echo -e "${RED}Examples directory not found${NC}"
        exit 1
    fi

    mkdir -p examples_build
    cd examples_build

    # Examples need to find the installed library
    # For development, we'll install to a local prefix
    LOCAL_PREFIX="$(pwd)/../install"
    cd ..

    # Install library to local prefix
    echo -e "${YELLOW}Installing library to local prefix for examples...${NC}"
    cmake --install . --prefix "$LOCAL_PREFIX"

    # Build examples
    cd examples_build
    cmake -DCMAKE_PREFIX_PATH="$LOCAL_PREFIX;$LIBTORCH_PATH" ../../examples
    cmake --build . -j "$JOBS"

    echo -e "${GREEN}Examples built successfully!${NC}"
    echo "Example binaries: $(pwd)/"
    echo ""
    echo "Run example with:"
    echo "  ./simple_example"
fi

echo ""
echo -e "${GREEN}All done!${NC}"
