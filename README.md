# ts-torch

A PyTorch-like deep learning library for TypeScript. Build, train, and evaluate neural networks with first-class TypeScript support.

## Packages

- **`@ts-torch/core`** - Tensor operations and FFI bindings for native compute
- **`@ts-torch/nn`** - Neural network modules (layers, activations, loss functions)
- **`@ts-torch/datasets`** - Dataset utilities and data loaders
- **`@ts-torch/optim`** - Optimization algorithms (SGD, Adam, etc.)
- **`@ts-torch-platform/loader`** - Native platform bindings loader

## Getting Started

### Installation

```bash
bun install
```

### Development

Watch mode for all packages:
```bash
bun run dev
```

Build all packages:
```bash
bun run build
```

Run type checking:
```bash
bun run check-types
```

Lint code:
```bash
bun run lint:check
```

Run tests:
```bash
bun run test
```

### Project Structure

```
packages/
├── @ts-torch/
│   ├── core/         # Tensor operations & FFI
│   ├── nn/           # Neural network modules
│   ├── datasets/     # Dataset utilities
│   └── optim/        # Optimizers
└── @ts-torch-platform/
    └── loader/       # Native bindings
examples/
└── mnist.ts          # MNIST training example
```

## Tools

- **TypeScript** 5.9.2 - Static type checking
- **Turborepo** - Monorepo task orchestration and caching
- **Bun** - Package manager and runtime
- **Vitest** - Testing framework

## CI/CD

GitHub Actions CI runs on every push and PR:
- ✓ Cross-platform builds (Linux, macOS, Windows)
- ✓ Type checking, linting, and tests
- ✓ Turborepo remote caching via Vercel

## License

MIT
