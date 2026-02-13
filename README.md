# ts-torch

A PyTorch-like deep learning library for TypeScript. Build, train, and evaluate neural networks with first-class TypeScript support.

## Quick Example

```typescript
import { device } from '@ts-torch/core'
import { nn } from '@ts-torch/nn'
import { Data, MNIST } from '@ts-torch/datasets'
import { Trainer, Adam, loss, logger } from '@ts-torch/train'

// Load data
const mnist = new MNIST('./data/mnist', true)
await mnist.load()
const trainLoader = Data.pipeline(mnist).shuffle().batch(64)

// Define model
const model = nn.sequence(
  nn.input(784),
  nn.fc(128).relu(),
  nn.fc(64).relu(),
  nn.fc(10)
).init(device.cpu())

// Train
const trainer = new Trainer({
  model,
  data: trainLoader,
  epochs: 3,
  optimizer: Adam({ lr: 1e-3 }),
  loss: loss.crossEntropy(),
  metrics: ['loss', 'accuracy'],
  callbacks: [logger.console()],
})

const history = await trainer.fit()
```

## Packages

| Package | Description |
|---------|-------------|
| `@ts-torch/core` | Tensor operations and FFI bindings for native compute |
| `@ts-torch/nn` | Neural network modules (layers, activations) |
| `@ts-torch/datasets` | Dataset loaders and data pipelines |
| `@ts-torch/optim` | Optimization algorithms (SGD, Adam, AdamW, RMSprop) |
| `@ts-torch/train` | Declarative training API (Trainer, callbacks, metrics, schedulers) |
| `@ts-torch/rl` | Reinforcement learning (PPO, A2C, DQN, SAC) |
| `@ts-torch-platform/loader` | Native platform bindings loader |

## API Overview

### Model Definition

Models are built declaratively with `nn.input()`, `nn.fc()`, and `nn.sequence()`. No memory is allocated until `.init(device)`:

```typescript
const model = nn.sequence(
  nn.input(784),             // explicit input shape
  nn.fc(128).relu(),         // Linear(784, 128) + ReLU
  nn.fc(64).relu(),          // Linear(128, 64) + ReLU
  nn.fc(10)                  // Linear(64, 10)
).init(device.cpu())
```

### Data Pipelines

Datasets produce `{ input, target }` batches. Pipelines are lazy — no work until iteration:

```typescript
const mnist = new MNIST('./data/mnist', true)
await mnist.load()

const loader = Data.pipeline(mnist)
  .shuffle()
  .batch(64)

// Use directly with Trainer — no .map() needed
```

### Training

All configuration goes to the constructor. `.fit()` is zero-arg:

```typescript
const trainer = new Trainer({
  model,
  data: trainLoader,
  epochs: 10,
  optimizer: Adam({ lr: 1e-3 }),
  loss: loss.crossEntropy(),
  metrics: ['loss', 'accuracy'],
  validation: testLoader,
  callbacks: [
    logger.console(),
    earlyStop({ patience: 5, monitor: 'loss' }),
  ],
})

const history = await trainer.fit()
```

### Loss Functions

Type-safe, serializable loss configuration:

```typescript
loss.crossEntropy()                      // classification
loss.crossEntropy({ labelSmoothing: 0.1 }) // with label smoothing
loss.mse()                               // regression
loss.nll()                               // negative log-likelihood
loss.custom('myLoss', (pred, target) => ...) // custom
```

### Callbacks

Composable callbacks replace boilerplate logging and control flow:

```typescript
import { logger, earlyStop, checkpoint } from '@ts-torch/train'

callbacks: [
  logger.console(),                           // auto-formatted epoch logging
  earlyStop({ patience: 5, monitor: 'loss' }), // stop when loss plateaus
  checkpoint({ every: 10 }),                   // save model periodically
]
```

Or use the `onEpochEnd` shorthand for one-off logging:

```typescript
new Trainer({
  // ...
  onEpochEnd: (ctx) => console.log(`Epoch ${ctx.epoch}: ${ctx.metrics.loss}`),
})
```

### Evaluation

```typescript
// Use configured validation data and metrics
const metrics = await trainer.evaluate()

// Override data
const metrics = await trainer.evaluate(testLoader)

// Override both
const metrics = await trainer.evaluate(testLoader, { metrics: ['loss'] })
```

### GPU Support

Swap `device.cpu()` for `device.cuda(0)`. The Trainer auto-transfers batches to GPU:

```typescript
const model = nn.sequence(
  nn.input(784),
  nn.fc(128).relu(),
  nn.fc(10)
).init(device.cuda(0))

// Data pipelines stay on CPU — Trainer handles transfer
const trainer = new Trainer({ model, data: trainLoader, ... })
```

## Getting Started

### Installation

```bash
bun install
```

### Development

```bash
bun run dev          # Watch mode
bun run build        # Build all packages
bun run check-types  # Type checking
bun run lint:check   # Lint
bun run test         # Run tests
```

## For Package Consumers

```bash
bun add @ts-torch/core @ts-torch/nn @ts-torch/train @ts-torch/datasets
```

The platform-specific binaries (`@ts-torch-platform/*`) are installed as optional dependencies and loaded automatically.

### CUDA Support

```bash
bun add @ts-torch-cuda/linux-x64-cu124   # Linux with CUDA 12.4
bun add @ts-torch-cuda/win32-x64-cu124   # Windows with CUDA 12.4
```

---

## For Library Contributors

### Quick Setup

```bash
git clone https://github.com/aryamurray/ts-torch.git
cd ts-torch
bun install
bun run setup        # Download LibTorch + build native bindings
bun test             # Verify
```

### CUDA Development Setup

```bash
bun run setup:cuda
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `LIBTORCH` | Override CPU LibTorch path |
| `LIBTORCH_CUDA` | Override CUDA LibTorch path |
| `TS_TORCH_LIB` | Override native library path |
| `TS_TORCH_DEBUG` | Enable debug logging (`1` to enable) |

### Troubleshooting

**"Native library not found"** — Re-run `bun run setup` to rebuild native bindings.

**Windows DLL dependency errors** — Install [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

**CUDA issues** — Use `bun run setup:cuda` and verify CUDA toolkit version matches LibTorch.

### Project Structure

```
packages/
├── @ts-torch/
│   ├── core/         # Tensor operations & FFI
│   ├── nn/           # Neural network modules
│   ├── datasets/     # Dataset loaders & pipelines
│   ├── optim/        # Optimizers & LR schedulers
│   ├── train/        # Declarative Trainer, callbacks, metrics
│   └── rl/           # Reinforcement learning
└── @ts-torch-platform/
    └── loader/       # Native bindings
examples/
├── mnist-cpu.ts      # MNIST training (CPU)
└── mnist-cuda.ts     # MNIST training (GPU)
```

## Tools

- **TypeScript** 5.9.2 — Static type checking
- **Turborepo** — Monorepo task orchestration and caching
- **Bun** — Package manager and runtime
- **Vitest** — Testing framework

## CI/CD

GitHub Actions CI runs on every push and PR:
- Cross-platform builds (Linux, macOS, Windows)
- Type checking, linting, and tests
- Turborepo remote caching via Vercel

## License

MIT
