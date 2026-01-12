# @ts-torch-platform/darwin-x64

Native binaries for ts-torch on macOS x64 (Intel).

## Installation

This package is automatically installed as a dependency of `@ts-torch/core` on compatible platforms.

```bash
bun add @ts-torch-platform/darwin-x64
```

## Contents

- `lib/libts_torch.dylib` - Native library compiled for macOS x64

## Requirements

- macOS 10.15 or later
- Intel-based Mac

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
- LibTorch 2.5.1+ for macOS x64
