# @ts-torch-platform/win32-x64

Native binaries for ts-torch on Windows x64 platform.

## Overview

This package contains the pre-compiled native binaries for ts-torch on Windows x64 systems. It is automatically installed as an optional dependency when you install `@ts-torch/core` on a Windows x64 machine.

## Platform Requirements

- **Operating System**: Windows (win32)
- **Architecture**: x64 (64-bit)
- **Node.js**: >= 18

## Contents

This package includes:

- `ts_torch.node` - Node.js native addon with PyTorch bindings
- Supporting DLL files (if required)

## Installation

This package is installed automatically as an optional dependency:

```bash
bun add @ts-torch/core
```

The package manager will automatically install this package only on Windows x64 systems.

## Manual Installation

If needed, you can install this package directly:

```bash
bun add @ts-torch-platform/win32-x64
```

## Building from Source

To build the native binaries from source:

### Prerequisites

- Visual Studio 2019 or later with C++ build tools
- CMake 3.15 or later
- PyTorch C++ libraries (libtorch) for Windows
- Node.js development headers

### Build Steps

1. Install build dependencies:

   ```bash
   # Install CMake
   winget install Kitware.CMake

   # Download libtorch from pytorch.org
   # Extract to a known location
   ```

2. Configure and build:

   ```bash
   cd native
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/libtorch
   cmake --build . --config Release
   ```

3. Copy binaries:
   ```bash
   cp build/Release/ts_torch.node ../packages/@ts-torch-platform/win32-x64/lib/
   ```

## Troubleshooting

### Missing DLL errors

If you get errors about missing DLLs, ensure that:

- Visual C++ Redistributable is installed
- PyTorch DLLs are in your PATH or next to the binary

### Version mismatch

Make sure the binary version matches your `@ts-torch/core` version.

## License

MIT
