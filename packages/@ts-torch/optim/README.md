# @ts-torch/optim

Optimizers for ts-torch neural network training.

## Overview

This package provides various optimization algorithms for training neural networks with ts-torch. It includes popular optimizers like SGD, Adam, AdamW, and RMSprop.

## Features

- **SGD**: Stochastic Gradient Descent with momentum and Nesterov variants
- **Adam**: Adaptive Moment Estimation
- **AdamW**: Adam with decoupled weight decay
- **RMSprop**: Root Mean Square Propagation
- **Optimizer Base Class**: Extensible base for custom optimizers

## Installation

```bash
bun add @ts-torch/optim
```

## Usage

### SGD (Stochastic Gradient Descent)

```typescript
import { SGD } from '@ts-torch/optim'

const optimizer = new SGD(model.parameters(), {
  lr: 0.01,
  momentum: 0.9,
  weightDecay: 1e-4,
  nesterov: true,
})

// Training loop
for (const [inputs, targets] of dataloader) {
  optimizer.zeroGrad()
  const outputs = model.forward(inputs)
  const loss = criterion(outputs, targets)
  loss.backward()
  optimizer.step()
}
```

### Adam

```typescript
import { Adam } from '@ts-torch/optim'

const optimizer = new Adam(model.parameters(), {
  lr: 0.001,
  betas: [0.9, 0.999],
  eps: 1e-8,
  weightDecay: 0,
})
```

### AdamW

```typescript
import { AdamW } from '@ts-torch/optim'

const optimizer = new AdamW(model.parameters(), {
  lr: 0.001,
  betas: [0.9, 0.999],
  eps: 1e-8,
  weightDecay: 0.01, // Decoupled weight decay
})
```

### RMSprop

```typescript
import { RMSprop } from '@ts-torch/optim'

const optimizer = new RMSprop(model.parameters(), {
  lr: 0.01,
  alpha: 0.99,
  eps: 1e-8,
  momentum: 0,
})
```

## Custom Optimizers

Create custom optimizers by extending the `Optimizer` base class:

```typescript
import { Optimizer } from '@ts-torch/optim'

class MyOptimizer extends Optimizer {
  step(): void {
    // Implement your optimization logic
  }
}
```
