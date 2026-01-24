# @ts-torch/train

Declarative training API for ts-torch neural networks.

## Overview

This package provides a high-level, configuration-driven approach to training neural networks. Instead of writing imperative training loops, describe what you want and the Trainer handles the rest.

## Features

- **Declarative `Trainer.fit()` API**: Configure training with a simple options object
- **Built-in Optimizers**: Adam, SGD, AdamW, RMSprop factory functions
- **LR Schedulers**: StepLR, CosineAnnealing, ReduceLROnPlateau, LinearWarmup, and more
- **Metrics Tracking**: Loss, accuracy, top-k accuracy, and custom metrics
- **Validation Support**: Validate during training with configurable frequency
- **Extensibility Hooks**: onEpochStart, onEpochEnd, onBatchStart, onBatchEnd, onForward, onBackward

## Installation

```bash
bun add @ts-torch/train
```

## Quick Start

```typescript
import { Trainer, Adam, StepLR } from '@ts-torch/train'
import { Linear, ReLU } from '@ts-torch/nn'

// Build a model
const model = new Linear(784, 128)
  .pipe(new ReLU())
  .pipe(new Linear(128, 10))

// Create trainer
const trainer = new Trainer(model)

// Train with declarative options
const history = await trainer.fit(trainLoader, {
  epochs: 10,
  optimizer: Adam({ lr: 1e-3 }),
  loss: 'crossEntropy',
  metrics: { loss: true, accuracy: true },
  validateOn: testLoader,
  onEpochEnd: ({ epoch, metrics }) => {
    console.log(`Epoch ${epoch}: loss=${metrics.loss.toFixed(4)}`)
  }
})
```

## API Reference

### Trainer

The main class for training neural networks.

```typescript
const trainer = new Trainer(model, {
  optimizer: Adam({ lr: 1e-3 }),  // Optional default optimizer
  loss: 'crossEntropy',           // Optional default loss
})
```

### FitOptions

Configuration options for `trainer.fit()`:

```typescript
interface FitOptions {
  epochs: number                    // Number of training epochs
  optimizer?: OptimizerConfig       // Optimizer factory
  loss?: LossType | LossFn          // Loss function
  metrics?: MetricsConfig           // Metrics to track
  scheduler?: SchedulerConfig       // LR scheduler
  validateOn?: AsyncIterable<any>   // Validation data
  validateEvery?: number            // Validate every N epochs
  accumulate?: number               // Gradient accumulation steps
  clipGradNorm?: number             // Gradient clipping

  // Callbacks
  onEpochStart?: (ctx) => void
  onEpochEnd?: (ctx) => void
  onBatchStart?: (ctx) => void
  onBatchEnd?: (ctx) => void
  onForward?: (ctx) => Tensor | void
  onBackward?: (ctx) => void
}
```

### Optimizers

Factory functions for creating optimizer configurations:

```typescript
import { Adam, SGD, AdamW, RMSprop } from '@ts-torch/train'

// Adam with weight decay
Adam({ lr: 1e-3, weightDecay: 1e-4 })

// SGD with momentum
SGD({ lr: 0.01, momentum: 0.9, nesterov: true })

// AdamW (decoupled weight decay)
AdamW({ lr: 1e-3, weightDecay: 0.01 })

// RMSprop
RMSprop({ lr: 0.01, alpha: 0.99 })
```

### Schedulers

Learning rate scheduler factories:

```typescript
import {
  StepLR,
  MultiStepLR,
  ExponentialLR,
  CosineAnnealingLR,
  CosineAnnealingWarmRestarts,
  ReduceLROnPlateau,
  LinearWarmup,
} from '@ts-torch/train'

// Decay by 0.1 every 30 epochs
StepLR({ stepSize: 30, gamma: 0.1 })

// Decay at specific epochs
MultiStepLR({ milestones: [30, 80], gamma: 0.1 })

// Cosine annealing
CosineAnnealingLR({ tMax: 50, etaMin: 0.001 })

// Reduce on plateau (needs validation metrics)
ReduceLROnPlateau({ mode: 'min', factor: 0.1, patience: 10 })

// Linear warmup (per-batch stepping)
LinearWarmup({ warmupSteps: 1000, stepOn: 'batch' })
```

### Metrics

Track training progress with built-in and custom metrics:

```typescript
// Built-in metrics
metrics: { loss: true, accuracy: true, topK: [1, 5] }

// Custom metrics
metrics: {
  accuracy: true,
  f1Score: (predictions, targets) => computeF1(predictions, targets)
}
```

### Loss Functions

Supported loss types:

```typescript
loss: 'crossEntropy'  // Cross-entropy loss (default)
loss: 'mse'           // Mean squared error
loss: 'nll'           // Negative log-likelihood

// Custom loss function
loss: (predictions, targets) => myCustomLoss(predictions, targets)
```

## Advanced Examples

### Training with Validation and Scheduler

```typescript
const history = await trainer.fit(trainLoader, {
  epochs: 100,
  optimizer: Adam({ lr: 0.1 }),
  scheduler: CosineAnnealingLR({ tMax: 100, etaMin: 1e-6 }),
  loss: 'crossEntropy',
  metrics: { loss: true, accuracy: true },
  validateOn: valLoader,
  validateEvery: 5,
  onEpochEnd: ({ epoch, metrics, valMetrics }) => {
    console.log(`Epoch ${epoch}`)
    console.log(`  Train: loss=${metrics.loss.toFixed(4)}, acc=${metrics.accuracy.toFixed(1)}%`)
    if (valMetrics) {
      console.log(`  Val:   loss=${valMetrics.loss.toFixed(4)}, acc=${valMetrics.accuracy.toFixed(1)}%`)
    }
  }
})
```

### Gradient Accumulation

```typescript
await trainer.fit(trainLoader, {
  epochs: 10,
  optimizer: Adam({ lr: 1e-3 }),
  accumulate: 4,  // Accumulate gradients over 4 batches
})
```

### Custom Forward Hook

```typescript
await trainer.fit(trainLoader, {
  epochs: 10,
  optimizer: Adam({ lr: 1e-3 }),
  onForward: ({ predictions, targets, loss }) => {
    // Add custom regularization
    const l2Reg = computeL2Regularization(model)
    return loss.add(l2Reg.mul(0.01))
  }
})
```

### Evaluation

```typescript
// Evaluate on test data
const metrics = await trainer.evaluate(testLoader, {
  metrics: { loss: true, accuracy: true }
})

console.log(`Test accuracy: ${metrics.accuracy.toFixed(1)}%`)
```

## License

MIT
