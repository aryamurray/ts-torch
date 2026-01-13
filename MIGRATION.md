# Migration Guide: Imperative to Declarative API

This guide explains how to migrate your existing code from the old imperative API to the new declarative API in ts-torch.

## Table of Contents

1. [Overview](#overview)
2. [Device Context Migration](#device-context-migration)
3. [Model Building Migration](#model-building-migration)
4. [Training Loop Migration](#training-loop-migration)
5. [Data Loading Migration](#data-loading-migration)
6. [Quick Reference Table](#quick-reference-table)

---

## Overview

### Philosophy Shift

The ts-torch library has evolved from an **imperative** programming model to a **declarative** one. This shift brings several benefits:

| Aspect | Imperative (Old) | Declarative (New) |
|--------|------------------|-------------------|
| **Focus** | How to do something | What you want to achieve |
| **Verbosity** | More boilerplate code | Concise, expressive syntax |
| **Device Management** | Implicit, global state | Explicit, scoped to device context |
| **Error Handling** | Manual at each step | Centralized, automatic |
| **Optimization** | Manual | Framework-managed |

### Backward Compatibility

> **Important**: The old imperative APIs still work and are fully backward compatible. You can migrate incrementally at your own pace. There is no requirement to update all your code at once.

The old APIs are not deprecated and will continue to be maintained. However, new features and optimizations will primarily target the declarative API.

---

## Device Context Migration

The most fundamental change is how tensors are created. Instead of using global factory functions, you now create tensors through a device context.

### Old API (Imperative)

```typescript
import { tensor, zeros, ones } from '@ts-torch/core'

const t = tensor([1, 2, 3])
const z = zeros([2, 3])
```

### New API (Declarative)

```typescript
import { device } from '@ts-torch/core'

// Create a CPU device context
const cpu = device.cpu()

// Create tensors through the device context
const t = cpu.tensor([1, 2, 3])
const z = cpu.zeros([2, 3])
```

### CUDA/GPU Support

The declarative API makes GPU usage explicit and straightforward:

```typescript
import { device } from '@ts-torch/core'

// Create a CUDA device context (specify GPU index)
const cuda = device.cuda(0)

// All tensors created through this context live on GPU
const gpuTensor = cuda.randn([2, 3])
const gpuZeros = cuda.zeros([1000, 1000])

// Easy device transfers
const cpuTensor = gpuTensor.to(device.cpu())
```

### Benefits of Device Contexts

1. **Explicit device placement**: No confusion about where tensors live
2. **Scoped operations**: All operations within a context use the same device
3. **Easy multi-GPU**: Create multiple CUDA contexts for different GPUs
4. **Cleaner error messages**: Device mismatches are caught early

---

## Model Building Migration

Neural network construction has been streamlined with a more declarative, configuration-driven approach.

### Old API (Imperative)

```typescript
import { Linear, ReLU, Sequential } from '@ts-torch/nn'

const model = new Sequential(
  new Linear(784, 128),
  new ReLU(),
  new Linear(128, 10)
)
```

### New API (Declarative)

#### Option 1: Using the MLP helper (recommended for standard architectures)

```typescript
import { nn } from '@ts-torch/nn'
import { device } from '@ts-torch/core'

const cuda = device.cuda(0)

const model = nn.mlp({
  device: cuda,
  layers: [784, 128, 10],
  activation: 'relu'
})
```

#### Option 2: Using sequence for custom architectures

```typescript
import { nn } from '@ts-torch/nn'
import { device } from '@ts-torch/core'

const cuda = device.cuda(0)

const model = nn.sequence(cuda, [
  nn.linear(784, 128),
  nn.relu(),
  nn.linear(128, 10)
])
```

### Advanced Model Configuration

The new API supports rich configuration options:

```typescript
const model = nn.mlp({
  device: cuda,
  layers: [784, 256, 128, 10],
  activation: 'relu',
  dropout: 0.2,
  batchNorm: true,
  weightInit: 'xavier'
})
```

### Custom Architectures

For complex architectures, you can still use the functional approach:

```typescript
const encoder = nn.sequence(cuda, [
  nn.conv2d(3, 64, { kernel: 3, padding: 1 }),
  nn.batchNorm2d(64),
  nn.relu(),
  nn.maxPool2d(2)
])

const decoder = nn.sequence(cuda, [
  nn.convTranspose2d(64, 3, { kernel: 3, padding: 1 }),
  nn.sigmoid()
])

const autoencoder = nn.sequence(cuda, [encoder, decoder])
```

---

## Training Loop Migration

The most significant productivity improvement comes from the declarative training API.

### Old API (Imperative)

```typescript
for (let epoch = 0; epoch < 10; epoch++) {
  for (const batch of dataLoader) {
    optimizer.zeroGrad()
    const output = model.forward(batch.data)
    const loss = crossEntropyLoss(output, batch.labels)
    loss.backward()
    optimizer.step()
  }
}
```

### New API (Declarative)

```typescript
import { Trainer, Adam } from '@ts-torch/train'

const trainer = new Trainer(model)

await trainer.fit(trainLoader, {
  epochs: 10,
  optimizer: Adam({ lr: 1e-3 }),
  loss: 'crossEntropy',
  metrics: { accuracy: true }
})
```

### Extended Training Configuration

The declarative API supports many training features out of the box:

```typescript
await trainer.fit(trainLoader, {
  epochs: 100,
  optimizer: Adam({ lr: 1e-3, weightDecay: 1e-4 }),
  loss: 'crossEntropy',
  metrics: {
    accuracy: true,
    f1: true,
    confusion: true
  },
  validation: {
    data: validLoader,
    frequency: 'epoch'
  },
  callbacks: {
    onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}: ${logs.loss}`),
    earlyStop: { patience: 5, monitor: 'val_loss' }
  },
  checkpoints: {
    save: './checkpoints',
    best: true
  },
  scheduler: {
    type: 'cosine',
    warmup: 5
  }
})
```

### When to Use the Imperative Loop

The imperative training loop is still available and useful for:

- Research experiments requiring fine-grained control
- Custom training procedures (GANs, reinforcement learning)
- Debugging and understanding model behavior

---

## Data Loading Migration

Data loading has been simplified with a pipeline-based approach.

### Old API (Imperative)

```typescript
// Manual batching and device transfer
const batches = []
for (let i = 0; i < data.length; i += batchSize) {
  const batch = data.slice(i, i + batchSize)
  batches.push(tensor(batch).to(device))
}
```

### New API (Declarative)

```typescript
import { Data } from '@ts-torch/core'

const loader = Data.pipeline(dataset)
  .shuffle()
  .batch(64)
  .to(cuda)
```

### Full Data Pipeline Example

```typescript
import { Data } from '@ts-torch/core'

// Create a complete data pipeline
const trainLoader = Data.pipeline(trainDataset)
  .shuffle({ seed: 42 })
  .augment({
    randomCrop: [28, 28],
    horizontalFlip: 0.5,
    normalize: { mean: [0.5], std: [0.5] }
  })
  .batch(64)
  .prefetch(2)
  .to(cuda)

// Validation data (no augmentation)
const validLoader = Data.pipeline(validDataset)
  .batch(128)
  .to(cuda)
```

---

## Quick Reference Table

| Old Pattern | New Pattern |
|-------------|-------------|
| `tensor([...])` | `device.cpu().tensor([...])` |
| `zeros([2, 3])` | `device.cpu().zeros([2, 3])` |
| `ones([2, 3])` | `device.cpu().ones([2, 3])` |
| `randn([2, 3])` | `device.cpu().randn([2, 3])` |
| `new Linear(in, out)` | `nn.linear(in, out)` |
| `new ReLU()` | `nn.relu()` |
| `new Sequential(...)` | `nn.sequence(device, [...])` |
| `new Conv2d(...)` | `nn.conv2d(...)` |
| Manual training loop | `trainer.fit(data, options)` |
| Manual batching | `Data.pipeline(data).batch(n)` |
| `tensor.cuda()` | `tensor.to(device.cuda(0))` |
| `model.to('cuda')` | Pass device in constructor |

---

## Migration Checklist

Use this checklist when migrating your codebase:

- [ ] Replace global tensor factory functions with device context methods
- [ ] Update neural network layer instantiation to use `nn.*` functions
- [ ] Replace `new Sequential` with `nn.sequence` or `nn.mlp`
- [ ] Convert manual training loops to `Trainer.fit()` where appropriate
- [ ] Update data loading to use `Data.pipeline()`
- [ ] Ensure all device placements are explicit
- [ ] Run tests to verify behavior is unchanged

---

## Getting Help

If you encounter issues during migration:

1. Check that you are using the latest version of ts-torch
2. Review the API documentation for the new declarative methods
3. Open an issue on GitHub with a minimal reproduction case

Remember: migration is optional. The old APIs remain fully functional and will continue to be supported.
