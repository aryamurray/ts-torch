# @ts-torch-platform/darwin-arm64

Native binaries for ts-torch on macOS ARM64 (Apple Silicon).

## Installation

This package is automatically installed as a dependency of `@ts-torch/core` on compatible platforms.

```bash
bun add @ts-torch-platform/darwin-arm64
```

## Contents

- `lib/libts_torch.dylib` - Native library compiled for macOS ARM64

## Requirements

- macOS 11.0 or later
- Apple Silicon (M1/M2/M3) Mac

## Building from Source

If you need to build the native library yourself:

```bash
# From the ts-torch repository root
bun run setup
bun run build:native
```

This requires:
- CMake 3.18+
- Xcode Command Line Tools
- LibTorch 2.5.1+ for macOS ARM64
