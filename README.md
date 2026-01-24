# ts-torch

A PyTorch-like deep learning library for TypeScript. Build, train, and evaluate neural networks with first-class TypeScript support.

## Packages

- **`@ts-torch/core`** - Tensor operations and FFI bindings for native compute
- **`@ts-torch/nn`** - Neural network modules (layers, activations, loss functions)
- **`@ts-torch/datasets`** - Dataset utilities and data loaders
- **`@ts-torch/optim`** - Optimization algorithms (SGD, Adam, etc.)
- **`@ts-torch/train`** - Declarative training API (Trainer, schedulers, metrics)
- **`@ts-torch/rl`** - Reinforcement learning (PPO, A2C, DQN, SAC)
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

## For Package Consumers

When you install ts-torch packages via npm/bun, native binaries are automatically included for your platform:

```bash
bun add @ts-torch/core @ts-torch/nn
```

The platform-specific binaries (`@ts-torch-platform/*`) are installed as optional dependencies and loaded automatically.

### CUDA Support (GPU Acceleration)

For GPU support, install the CUDA package for your platform:

```bash
bun add @ts-torch-cuda/linux-x64-cu124   # Linux with CUDA 12.4
bun add @ts-torch-cuda/win32-x64-cu124   # Windows with CUDA 12.4
```

The postinstall script automatically downloads the required LibTorch CUDA binaries.

---

## For Library Contributors

If you're developing ts-torch itself (git clone), you need to set up the development environment:

### Quick Setup

```bash
# Clone and install dependencies
git clone https://github.com/aryamurray/ts-torch.git
cd ts-torch
bun install

# Download LibTorch and build native bindings
bun run setup

# Run tests to verify
bun test
```

### CUDA Development Setup

```bash
bun run setup:cuda
```

### Environment Variables (Advanced)

| Variable | Purpose |
|----------|---------|
| `LIBTORCH` | Override CPU LibTorch path |
| `LIBTORCH_CUDA` | Override CUDA LibTorch path |
| `TS_TORCH_LIB` | Override native library path |
| `TS_TORCH_DEBUG` | Enable debug logging (`1` to enable) |

### Troubleshooting

**"Native library not found"**
- Re-run `bun run setup` to rebuild native bindings
- Check that `libtorch/lib` directory exists

**Windows DLL dependency errors**
- Install [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

**CUDA issues**
- Use `bun run setup:cuda` for GPU support
- Verify CUDA toolkit version matches LibTorch

### Project Structure

```
packages/
├── @ts-torch/
│   ├── core/         # Tensor operations & FFI
│   ├── nn/           # Neural network modules
│   ├── datasets/     # Dataset utilities
│   ├── optim/        # Optimizers
│   ├── train/        # Declarative training API
│   └── rl/           # Reinforcement learning
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
