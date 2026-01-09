# Native Library Setup Guide

This guide explains how to build and connect the native libtorch bindings.

## Overview

The architecture has three layers:

```
TypeScript (Bun FFI) → C Shim (ts_torch.dll) → LibTorch (C++)
```

## Step 1: Download LibTorch

Download the pre-built LibTorch for your platform from [pytorch.org](https://pytorch.org/get-started/locally/).

### Windows (CPU)
```powershell
# Download and extract to C:\libtorch
curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.5.1%2Bcpu.zip
Expand-Archive libtorch.zip -DestinationPath C:\
```

### Windows (CUDA 12.4)
```powershell
curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cu124/libtorch-win-shared-with-deps-2.5.1%2Bcu124.zip
Expand-Archive libtorch.zip -DestinationPath C:\
```

### macOS (CPU)
```bash
curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.1.zip
unzip libtorch.zip -d /usr/local/lib/
```

### Linux (CPU)
```bash
curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
unzip libtorch.zip -d /opt/
```

## Step 2: Set Environment Variable

```powershell
# Windows (PowerShell)
$env:LIBTORCH = "C:\libtorch"
$env:PATH = "$env:LIBTORCH\lib;$env:PATH"

# Or permanently via System Properties > Environment Variables
```

```bash
# macOS/Linux
export LIBTORCH=/usr/local/lib/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

## Step 3: Build the C Shim

The C shim source is in `packages/@ts-torch/core/native/`.

### Prerequisites

- **Windows**: Visual Studio 2019/2022 with C++ workload, CMake 3.18+
- **macOS**: Xcode Command Line Tools, CMake 3.18+
- **Linux**: GCC 9+, CMake 3.18+

### Build Commands

```bash
cd packages/@ts-torch/core/native

# Configure
cmake -B build -DCMAKE_PREFIX_PATH=$LIBTORCH

# Build
cmake --build build --config Release

# The output will be:
# - Windows: build/Release/ts_torch.dll
# - macOS:   build/libts_torch.dylib
# - Linux:   build/libts_torch.so
```

### Windows with Visual Studio

```powershell
cd packages/@ts-torch/core/native

cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH=C:\libtorch
cmake --build build --config Release
```

## Step 4: Install the Native Library

Copy the built library to the platform package:

```bash
# Windows
cp packages/@ts-torch/core/native/build/Release/ts_torch.dll packages/@ts-torch-platform/win32-x64/lib/

# macOS
cp packages/@ts-torch/core/native/build/libts_torch.dylib packages/@ts-torch-platform/darwin-arm64/lib/

# Linux
cp packages/@ts-torch/core/native/build/libts_torch.so packages/@ts-torch-platform/linux-x64/lib/
```

Or set the environment variable to point directly to it:

```bash
export TS_TORCH_LIB=/path/to/ts_torch.dll
```

## Step 5: Verify Installation

```typescript
// test-native.ts
import { torch } from "@ts-torch/core";

console.log("LibTorch version:", torch.version());
console.log("CUDA available:", torch.cuda.isAvailable());

// Test tensor creation
torch.run(() => {
  const t = torch.zeros([2, 3]);
  console.log("Created tensor with shape:", t.shape);
});
```

```bash
bun run test-native.ts
```

## Troubleshooting

### "Cannot find native library"

1. Check `TS_TORCH_LIB` environment variable points to the `.dll`/`.so`/`.dylib`
2. Or ensure the platform package has the library in its `lib/` folder
3. Run `bun install` to re-link workspace packages

### "DLL not found" / Symbol errors

1. Ensure LibTorch DLLs are in PATH:
   ```powershell
   $env:PATH = "C:\libtorch\lib;$env:PATH"
   ```
2. On Windows, you may need the Visual C++ Redistributable

### CMake can't find LibTorch

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch
```

## Project Structure After Setup

```
ts-tools/
├── packages/@ts-torch/core/
│   ├── native/
│   │   ├── include/ts_torch.h      # C API header
│   │   ├── src/ts_torch.cpp        # Implementation
│   │   ├── CMakeLists.txt          # Build config
│   │   └── build/                  # Build output
│   │       └── Release/
│   │           └── ts_torch.dll    # Built library
│   └── src/
│       └── ffi/
│           ├── symbols.ts          # FFI definitions
│           └── loader.ts           # Library loader
│
├── packages/@ts-torch-platform/
│   └── win32-x64/
│       └── lib/
│           └── ts_torch.dll        # Distributed library
│
└── C:\libtorch/                    # Downloaded LibTorch
    ├── include/
    ├── lib/
    │   ├── torch.dll
    │   ├── c10.dll
    │   └── ...
    └── share/
```

## Next Steps

Once the native library is built and loadable:

1. The `torch.*` functions will call real LibTorch operations
2. Tensor creation (`torch.zeros`, `torch.randn`, etc.) will allocate GPU/CPU memory
3. The `torch.run()` scoping will properly free native tensors
4. CUDA operations will work if you have a compatible GPU and CUDA LibTorch build
