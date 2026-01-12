# @ts-torch-platform/linux-x64

Native binaries for ts-torch on Linux x64.

## Installation

This package is automatically installed as a dependency of `@ts-torch/core` on compatible platforms.

```bash
bun add @ts-torch-platform/linux-x64
```

## Contents

- `lib/libts_torch.so` - Native library compiled for Linux x64

## Requirements

- Linux (glibc 2.17+)
- x86_64 architecture

## Building from Source

If you need to build the native library yourself:

```bash
# From the ts-torch repository root
bun run setup
bun run build:native
```

This requires:
- CMake 3.18+
- GCC 9+ or Clang 10+
- LibTorch 2.5.1+ for Linux x64
