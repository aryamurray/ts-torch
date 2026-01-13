# ts-torch Architecture

A PyTorch-like deep learning library for TypeScript with compile-time type safety and native performance through FFI bindings to LibTorch.

## Project Structure

```
packages/
├── @ts-torch/core/          # Core tensor operations & FFI bindings
├── @ts-torch/nn/            # Neural network modules and layers
├── @ts-torch/optim/         # Optimizers and loss functions
├── @ts-torch/datasets/      # Dataset loaders and utilities
├── @ts-torch/train/         # Declarative training API
├── @ts-torch-platform/      # Platform-specific native bindings
│   ├── loader/
│   ├── darwin-arm64/
│   ├── darwin-x64/
│   ├── linux-x64/
│   └── win32-x64/
├── @ts-torch-cuda/          # CUDA-specific implementations
└── shared-test-utils/       # Shared testing utilities
```

**Stack:** TypeScript 5.9, Turborepo, Bun, Vitest, Vite, Koffi (FFI)

---

## Core Package (`@ts-torch/core`)

Foundation layer providing tensors, memory management, and native bindings.

### Tensor Type System

Tensors carry shape, dtype, and device information at compile time:

```typescript
type TensorType<
  S extends Shape,
  D extends DTypeName,
  Dev extends DeviceType
>

// Usage - types are inferred
const x = zeros([2, 3])           // Tensor<[2, 3], 'float32', 'cpu'>
const y = cuda.randn([3, 4])      // Tensor<[3, 4], 'float32', 'cuda'>
const z = x.matmul(y)             // Tensor<[2, 4], 'float32', 'cpu'> - compile error! device mismatch
```

### Shape Operations

Compile-time shape inference for all operations:

```typescript
type MatMulShape<[2, 3], [3, 4]> = [2, 4]
type BroadcastShape<[1, 3, 4], [2, 1, 4]> = [2, 3, 4]
type TransposeShape<[2, 3, 4], 0, 2> = [4, 3, 2]
type ReduceShape<[2, 3, 4], 1, false> = [2, 4]
```

### Device Context

Device-bound tensor creation:

```typescript
import { device } from '@ts-torch/core'

const cuda = device.cuda(0)
const x = cuda.zeros([784, 128])   // Created directly on GPU
const y = cuda.randn([128, 10])
```

### Scoped Memory Management

Automatic cleanup through lexical scoping:

```typescript
import { run } from '@ts-torch/core'

const result = run(() => {
  const a = zeros([100, 100])   // Registered with scope
  const b = zeros([100, 100])   // Registered with scope
  const c = a.add(b)
  return c.escape()             // Keep c alive
  // a and b freed automatically
})
```

Implementation uses a stack-based scope tracker with native `ts_scope_begin()` and `ts_scope_end()` FFI calls.

### DType System

Branded types for compile-time dtype safety:

```typescript
type DTypeName = 'float32' | 'float64' | 'int32' | 'int64' | 'float16' | 'bfloat16' | 'bool'

// Runtime constants
DType.float32  // { name: 'float32', value: 0, bytes: 4 }
DType.int64    // { name: 'int64', value: 3, bytes: 8 }
```

### FFI Bindings

Native operations via Koffi:

```typescript
// loader.ts loads platform-specific binary
const lib = loadNativeLibrary()

// Tensor operations call native functions
x.matmul(y)  → lib.ts_tensor_matmul(x.handle, y.handle, error)
x.add(y)     → lib.ts_tensor_add(x.handle, y.handle, error)
```

---

## NN Package (`@ts-torch/nn`)

Neural network building blocks with type-safe shape inference.

### Module Base Class

All layers extend `Module<In, Out, D, Dev>`:

```typescript
class Linear<
  InFeatures extends number,
  OutFeatures extends number,
  D extends DType = DType<'float32'>,
  Dev extends DeviceType = 'cpu'
> extends Module<[number, InFeatures], [number, OutFeatures], D, Dev> {

  weight: Parameter<[OutFeatures, InFeatures], D, Dev>
  bias: Parameter<[OutFeatures], D, Dev>

  forward(input: Tensor<[number, InFeatures], D, Dev>) {
    return input.matmul(this.weight.t()).add(this.bias)
  }
}
```

### Module Composition

Chain modules with `.pipe()`:

```typescript
const model = new Linear(784, 128)
  .pipe(new ReLU())
  .pipe(new Linear(128, 64))
  .pipe(new ReLU())
  .pipe(new Linear(64, 10))

// Output type automatically inferred through the chain
```

### Fluent Builder API

Separate definition from initialization:

```typescript
// Define (no memory allocated)
const config = nn.sequence(784,
  nn.fc(128).relu().dropout(0.2),
  nn.fc(64).gelu(),
  nn.fc(10)
)

// Initialize on device
const model = config.init(device.cuda(0))
```

### Available Modules

- **Linear** - Fully connected layer
- **ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU** - Activations
- **Sequential** - Container for chaining
- **Conv2d, BatchNorm2d, Dropout** - Common layers
- **F** namespace - Functional stateless operations

---

## Optim Package (`@ts-torch/optim`)

Optimization algorithms and loss functions.

### Optimizer Base

```typescript
abstract class Optimizer {
  protected parameterGroups: ParameterGroup[]

  abstract step(): void

  zeroGrad(): void {
    for (const group of this.parameterGroups) {
      for (const param of group.params) {
        param.grad?.zero_()
      }
    }
  }
}
```

### Adam Implementation

```typescript
class Adam extends Optimizer {
  step() {
    for (const param of this.parameters) {
      const state = this.state.get(param)
      state.step++

      // Exponential moving averages
      state.exp_avg = β1 * state.exp_avg + (1 - β1) * grad
      state.exp_avg_sq = β2 * state.exp_avg_sq + (1 - β2) * grad²

      // Bias correction + update
      param -= lr * exp_avg / (sqrt(exp_avg_sq) + ε)
    }
  }
}
```

### Loss Functions

```typescript
crossEntropyLoss(predictions, targets)  // Classification
mseLoss(predictions, targets)           // Regression
nllLoss(log_probs, targets)             // Negative log likelihood
```

---

## Datasets Package (`@ts-torch/datasets`)

Data loading and transformation pipelines.

### DataLoader

```typescript
class DataLoader<T> {
  constructor(
    dataset: Dataset<T>,
    batchSize: number,
    shuffle: boolean
  )

  async *[Symbol.asyncIterator]() {
    // Yields batches lazily
  }
}
```

### Pipeline API

Lazy, declarative data pipelines:

```typescript
const pipeline = Data.pipeline(dataset)
  .shuffle()           // Shuffled indices
  .batch(64)           // Group into batches
  .to('cuda')          // Transfer to device
  .map(transform)      // Transform batch

for await (const batch of pipeline) {
  // Work happens during iteration
}
```

### Built-in Datasets

- **MNIST** - Handwritten digits
- **CIFAR10/CIFAR100** - Image classification
- **ImageFolder** - Custom image datasets
- **TextClassification** - Text datasets

---

## Train Package (`@ts-torch/train`)

Declarative training API.

### Trainer

```typescript
const trainer = new Trainer(model)

await trainer.fit(trainLoader, {
  epochs: 10,
  optimizer: Adam({ lr: 1e-3 }),
  loss: 'crossEntropy',
  metrics: { loss: true, accuracy: true },
  validateOn: testLoader,
  onEpochEnd: ({ epoch, metrics }) => console.log(metrics)
})
```

### Training Flow

```
For each epoch:
  For each batch:
    optimizer.zeroGrad()
    predictions = model.forward(data)
    loss = lossFunction(predictions, targets)
    loss.backward()
    optimizer.step()
  Compute metrics
  Validation pass (optional)
  onEpochEnd callback
```

---

## Build System

### Turborepo Configuration

```json
{
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**"]
    },
    "@ts-torch/core#build": {
      "dependsOn": ["build:native", "^build"]
    },
    "build:native": {
      "inputs": ["native/src/**", "native/CMakeLists.txt"],
      "outputs": ["native/build/**"],
      "env": ["LIBTORCH", "LIBTORCH_PATH"]
    }
  }
}
```

### Build Order

1. Core native C++ (CMake) → `.dll` / `.so` / `.dylib`
2. Core TypeScript (Vite) → consumes native bindings
3. NN/Optim/Datasets → depends on Core
4. Train → depends on Core, NN, Optim

### Package Exports

```json
{
  "exports": {
    ".": {
      "bun": "./src/index.ts",
      "types": "./dist/index.d.ts",
      "import": "./dist/index.mjs",
      "require": "./dist/index.cjs"
    }
  }
}
```

---

## Complete Example

```typescript
import { device } from '@ts-torch/core'
import { nn } from '@ts-torch/nn'
import { Data, MNIST } from '@ts-torch/datasets'
import { Trainer, Adam } from '@ts-torch/train'

async function main() {
  // Load data
  const mnist = new MNIST('./data/mnist', true)
  await mnist.load()

  // Create pipeline
  const trainPipeline = Data.pipeline(mnist)
    .shuffle()
    .batch(64)
    .map(b => ({ data: b.images, label: b.labelsTensor }))

  // Define model
  const config = nn.sequence(784,
    nn.fc(128).relu(),
    nn.fc(64).relu(),
    nn.fc(10)
  )

  // Initialize on device
  const model = config.init(device.cpu())

  // Train
  const trainer = new Trainer(model)
  await trainer.fit(trainPipeline, {
    epochs: 3,
    optimizer: Adam({ lr: 1e-3 }),
    loss: 'crossEntropy',
    metrics: { accuracy: true }
  })
}
```

---

## Key Design Decisions

1. **TypeScript-First** - Generics provide compile-time safety without runtime overhead
2. **Lazy Evaluation** - Data pipelines use async iterators for memory efficiency
3. **Memory Scoping** - Automatic cleanup inspired by PyTorch's context managers
4. **Native Bindings** - FFI (Koffi) calls C++ implementations directly
5. **Declarative APIs** - Configuration objects over imperative code
6. **Multi-Device** - Unified abstraction for CPU, CUDA, and MPS

---

## Critical Files

| File | Purpose |
|------|---------|
| `core/src/types/shape.ts` | Shape type operations |
| `core/src/types/tensor.ts` | MatMulShape, BroadcastShape, etc. |
| `core/src/tensor/tensor.ts` | Core Tensor class |
| `core/src/memory/scope.ts` | Scoped memory management |
| `core/src/device/context.ts` | DeviceContext |
| `core/src/ffi/loader.ts` | Koffi FFI loader |
| `nn/src/module.ts` | Base Module class |
| `nn/src/modules/container.ts` | Sequential container |
| `optim/src/optimizer.ts` | Base Optimizer |
| `optim/src/adam.ts` | Adam implementation |
| `datasets/src/dataloader.ts` | DataLoader |
| `train/src/trainer.ts` | Declarative Trainer |
