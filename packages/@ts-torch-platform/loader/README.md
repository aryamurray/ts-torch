# @ts-torch-platform/loader

Platform detection and native binary loading for ts-torch.

## Overview

This package handles platform detection and loading of platform-specific native binaries for ts-torch. It automatically detects the current operating system and architecture, then loads the appropriate native library.

## Features

- **Platform Detection**: Automatically detects OS and CPU architecture
- **Binary Loading**: Loads the correct native library for the current platform
- **Graceful Fallback**: Provides helpful error messages when binaries are missing
- **Optional Dependencies**: Uses optional dependencies for platform packages

## Supported Platforms

- Windows x64 (`win32-x64`)
- Windows ARM64 (`win32-arm64`)
- macOS x64 (`darwin-x64`)
- macOS ARM64 (Apple Silicon) (`darwin-arm64`)
- Linux x64 (`linux-x64`)
- Linux ARM64 (`linux-arm64`)

## Installation

This package is typically installed as a dependency of `@ts-torch/core` and doesn't need to be installed directly.

```bash
bun add @ts-torch-platform/loader
```

## Usage

```typescript
import {
  getPlatformInfo,
  loadNativeBinary,
  isNativeAvailable,
  getPlatformIdentifier,
} from "@ts-torch-platform/loader";

// Get platform information
const info = getPlatformInfo();
console.log("Platform:", info.platform);
console.log("Architecture:", info.arch);
console.log("Package name:", info.packageName);

// Check if native binaries are available
if (isNativeAvailable()) {
  console.log("Native binaries found");
  const binaryPath = loadNativeBinary();
  console.log("Binary path:", binaryPath);
} else {
  console.log("Native binaries not available, using fallback");
}

// Get platform identifier
const identifier = getPlatformIdentifier();
console.log("Platform ID:", identifier); // e.g., "win32-x64"
```

## Error Handling

When native binaries are not available, the package provides helpful error messages:

```typescript
import { loadNativeBinaryOrThrow, getMissingBinaryInfo } from "@ts-torch-platform/loader";

try {
  const binaryPath = loadNativeBinaryOrThrow();
  // Use binary
} catch (error) {
  console.error(error.message);
  // Includes installation instructions
}
```

## Platform Packages

Each platform has its own package containing native binaries:

- `@ts-torch-platform/win32-x64` - Windows x64
- `@ts-torch-platform/win32-arm64` - Windows ARM64
- `@ts-torch-platform/darwin-x64` - macOS Intel
- `@ts-torch-platform/darwin-arm64` - macOS Apple Silicon
- `@ts-torch-platform/linux-x64` - Linux x64
- `@ts-torch-platform/linux-arm64` - Linux ARM64

These packages are listed as optional dependencies and are automatically installed for the matching platform.
