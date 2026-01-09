# ts-torch

A PyTorch-like deep learning framework for TypeScript/JavaScript with native performance.

## Overview

ts-torch brings the power of PyTorch to the TypeScript ecosystem with a familiar API, strong typing, and native performance through FFI bindings. Build, train, and deploy neural networks using TypeScript with compile-time shape checking and type safety.

## Packages

### Core Packages

#### [@ts-torch/core](./core)
Core tensor operations and FFI bindings to native PyTorch libraries.

**Features:**
- Tensor creation and manipulation
- FFI bindings to native operations
- Memory management and pooling
- Advanced TypeScript type system for compile-time shape checking

#### [@ts-torch/nn](./nn)
Neural network modules and layers.

**Features:**
- Module base class and Sequential container
- Linear (fully connected) layers
- Activation functions (ReLU, Sigmoid, Tanh, Softmax)
- Convolutional layers (Conv2d)
- Pooling layers (MaxPool2d, AvgPool2d)
- Normalization (BatchNorm2d, LayerNorm)
- Regularization (Dropout)

#### [@ts-torch/optim](./optim)
Optimization algorithms for neural network training.

**Features:**
- SGD (with momentum and Nesterov)
- Adam
- AdamW (decoupled weight decay)
- RMSprop
- Extensible optimizer base class

#### [@ts-torch/datasets](./datasets)
Dataset loaders and utilities.

**Features:**
- Base dataset classes and interfaces
- DataLoader with batching and shuffling
- Data transformations (normalization, augmentation)
- Vision datasets (MNIST, CIFAR-10/100, ImageFolder)
- Text datasets
- Train/test splitting utilities

### Platform Packages

#### [@ts-torch-platform/loader](../@ts-torch-platform/loader)
Platform detection and binary loading.

Automatically detects the current platform and loads the appropriate native binaries.

#### [@ts-torch-platform/win32-x64](../@ts-torch-platform/win32-x64)
Native binaries for Windows x64.

Pre-compiled native binaries for Windows x64 systems.

## Installation

```bash
# Install all core packages
bun add @ts-torch/core @ts-torch/nn @ts-torch/optim @ts-torch/datasets

# Platform packages are installed automatically as optional dependencies
```

## Quick Start

```typescript
import { tensor, zeros, ones } from '@ts-torch/core';
import { Sequential, Linear, ReLU, Dropout } from '@ts-torch/nn';
import { Adam } from '@ts-torch/optim';
import { DataLoader, TensorDataset } from '@ts-torch/datasets';

// Create tensors
const x = tensor([[1, 2, 3], [4, 5, 6]]);
const z = zeros([3, 3]);

// Build a neural network
const model = new Sequential(
  new Linear(784, 256),
  new ReLU(),
  new Dropout(0.5),
  new Linear(256, 10)
);

// Create an optimizer
const optimizer = new Adam(model.parameters(), { lr: 0.001 });

// Create a dataset and data loader
const dataset = new TensorDataset([trainData, trainLabels]);
const loader = new DataLoader(dataset, {
  batchSize: 32,
  shuffle: true
});

// Training loop
model.train();
for (const [inputs, targets] of loader.iter()) {
  optimizer.zeroGrad();
  const outputs = model.forward(inputs);
  // const loss = criterion(outputs, targets);
  // loss.backward();
  optimizer.step();
}

// Evaluation
model.eval();
const testOutput = model.forward(testInput);
```

## Advanced Type System

ts-torch includes an advanced TypeScript type system for compile-time shape and dtype checking:

```typescript
import type { TensorType, MatMulShape } from '@ts-torch/core/types';

// Define tensor types
type Matrix = TensorType<[100, 50], "float32">;
type Vector = TensorType<[50, 1], "float32">;

// Compute result shape at compile time
type Result = MatMulShape<[100, 50], [50, 1]>; // [100, 1]

// This would be a compile error:
type Invalid = MatMulShape<[100, 50], [60, 1]>; // never (incompatible)
```

## Architecture

```
ts-torch
├── @ts-torch/core           # Core tensor operations
│   ├── src/ffi/            # FFI bindings
│   ├── src/tensor/         # Tensor class
│   ├── src/types/          # Type definitions
│   └── src/memory/         # Memory management
├── @ts-torch/nn             # Neural networks
│   ├── src/module.ts       # Base module
│   └── src/modules/        # Layer implementations
├── @ts-torch/optim          # Optimizers
│   ├── src/optimizer.ts    # Base optimizer
│   ├── src/sgd.ts          # SGD
│   ├── src/adam.ts         # Adam
│   └── src/adamw.ts        # AdamW
├── @ts-torch/datasets       # Datasets
│   ├── src/dataset.ts      # Base classes
│   ├── src/dataloader.ts   # DataLoader
│   ├── src/transforms.ts   # Transformations
│   ├── src/vision/         # Vision datasets
│   └── src/text/           # Text datasets
└── @ts-torch-platform/      # Platform binaries
    ├── loader/             # Binary loader
    └── win32-x64/          # Windows x64 binaries
```

## Development

### Building All Packages

```bash
# Build all packages
bun run build

# Watch mode for development
bun run dev

# Type checking
bun run check-types

# Linting
bun run lint
```

### Building Individual Packages

```bash
cd packages/@ts-torch/core
bun run build
```

## Requirements

- Node.js >= 18
- Bun (package manager)
- TypeScript 5.9.2

For native binary compilation:
- CMake >= 3.15
- C++ compiler (MSVC on Windows, GCC/Clang on Linux/macOS)
- PyTorch C++ libraries (libtorch)

## Roadmap

- [ ] Complete tensor operations implementation
- [ ] Native FFI bindings to PyTorch
- [ ] GPU support (CUDA/Metal)
- [ ] Autograd and automatic differentiation
- [ ] Pre-trained model zoo
- [ ] ONNX export support
- [ ] Distributed training
- [ ] More optimizers and schedulers
- [ ] Additional datasets

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

MIT

## Acknowledgments

ts-torch is inspired by PyTorch and aims to bring its elegant API to the TypeScript ecosystem.
