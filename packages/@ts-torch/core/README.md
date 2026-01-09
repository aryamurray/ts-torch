# @ts-torch/core

Core tensor operations & FFI bindings for ts-torch.

## Overview

This package provides the foundational tensor operations and Foreign Function Interface (FFI) bindings for ts-torch, enabling high-performance tensor computations backed by native implementations.

## Features

- **Tensor Operations**: Core tensor class and operations (creation, manipulation, etc.)
- **FFI Bindings**: Low-level bindings to native tensor operations and PyTorch C++ API
- **Type Definitions**: Comprehensive TypeScript type definitions for tensor operations
- **Memory Management**: Efficient memory pooling and tracking for tensor allocations

## Installation

```bash
bun add @ts-torch/core
```

## Usage

```typescript
import { tensor, zeros, ones } from "@ts-torch/core";

// Create a tensor from data
const t = tensor([
  [1, 2, 3],
  [4, 5, 6],
]);
console.log(t.shape); // [2, 3]

// Create zero/one tensors
const z = zeros([3, 3]);
const o = ones([2, 4]);
```

## Development

```bash
# Build the package
bun run build

# Watch mode
bun run dev

# Type checking
bun run check-types
```
