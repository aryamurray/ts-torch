# @ts-torch/optim - Usage Guide

Complete implementation of optimizers, loss functions, and learning rate schedulers for ts-torch.

## Installation

```bash
npm install @ts-torch/optim @ts-torch/core
```

## Optimizers

### SGD (Stochastic Gradient Descent)

Classic SGD optimizer with optional momentum, weight decay, and Nesterov acceleration.

```typescript
import { SGD } from '@ts-torch/optim'
import type { Tensor } from '@ts-torch/core'

// Create model parameters
const params: Tensor[] = model.parameters()

// Initialize SGD optimizer
const optimizer = new SGD(params, {
  lr: 0.01, // Learning rate
  momentum: 0.9, // Momentum factor (default: 0)
  dampening: 0, // Dampening for momentum (default: 0)
  weightDecay: 1e-4, // L2 regularization (default: 0)
  nesterov: false, // Enable Nesterov momentum (default: false)
})

// Training loop
for (let epoch = 0; epoch < numEpochs; epoch++) {
  for (const batch of dataLoader) {
    // Forward pass
    const output = model(batch.input)
    const loss = criterion(output, batch.target)

    // Backward pass
    optimizer.zeroGrad()
    loss.backward()

    // Update parameters
    optimizer.step()
  }
}
```

### Adam (Adaptive Moment Estimation)

Adam optimizer with adaptive learning rates and optional AMSGrad variant.

```typescript
import { Adam } from '@ts-torch/optim'

const optimizer = new Adam(model.parameters(), {
  lr: 0.001, // Learning rate
  betas: [0.9, 0.999], // Coefficients for computing running averages
  eps: 1e-8, // Term for numerical stability
  weightDecay: 0, // L2 regularization (default: 0)
  amsgrad: false, // Use AMSGrad variant (default: false)
})

// Training loop
for (let epoch = 0; epoch < numEpochs; epoch++) {
  trainOneEpoch(model, optimizer, trainLoader)
}
```

### AdamW (Adam with Decoupled Weight Decay)

AdamW properly decouples weight decay from the gradient-based optimization.

```typescript
import { AdamW } from '@ts-torch/optim'

const optimizer = new AdamW(model.parameters(), {
  lr: 0.001,
  betas: [0.9, 0.999],
  eps: 1e-8,
  weightDecay: 0.01, // Decoupled weight decay
  amsgrad: false,
})
```

### RMSprop

RMSprop optimizer with momentum support.

```typescript
import { RMSprop } from '@ts-torch/optim'

const optimizer = new RMSprop(model.parameters(), {
  lr: 0.01,
  alpha: 0.99, // Smoothing constant
  eps: 1e-8,
  weightDecay: 0,
  momentum: 0,
  centered: false, // Compute centered RMSprop
})
```

## Loss Functions

### Cross Entropy Loss

For multi-class classification tasks.

```typescript
import { crossEntropyLoss } from '@ts-torch/optim'
import { tensor } from '@ts-torch/core'

// Predictions (logits) - shape [batch_size, num_classes]
const logits = tensor([
  [2.0, 1.0, 0.1],
  [0.5, 2.5, 0.2],
])

// Ground truth class indices - shape [batch_size]
const targets = tensor([0, 1])

// Compute loss (default reduction: 'mean')
const loss = crossEntropyLoss(logits, targets)

// With custom reduction
const lossPerSample = crossEntropyLoss(logits, targets, { reduction: 'none' })
const lossSum = crossEntropyLoss(logits, targets, { reduction: 'sum' })
```

### Mean Squared Error (MSE) Loss

For regression tasks.

```typescript
import { mseLoss } from '@ts-torch/optim'

const predictions = tensor([2.5, 0.0, 2.0, 8.0])
const targets = tensor([3.0, -0.5, 2.0, 7.0])

const loss = mseLoss(predictions, targets)
// Computes: mean((predictions - targets)^2)
```

### Binary Cross Entropy Loss

For binary classification tasks.

```typescript
import { binaryCrossEntropyLoss } from '@ts-torch/optim'

// Predictions after sigmoid (probabilities in [0, 1])
const predictions = tensor([0.8, 0.3, 0.6, 0.9])
const targets = tensor([1.0, 0.0, 1.0, 1.0])

const loss = binaryCrossEntropyLoss(predictions, targets)
```

### L1 Loss (Mean Absolute Error)

Less sensitive to outliers than MSE.

```typescript
import { l1Loss } from '@ts-torch/optim'

const predictions = tensor([2.5, 0.0, 2.0, 8.0])
const targets = tensor([3.0, -0.5, 2.0, 7.0])

const loss = l1Loss(predictions, targets)
// Computes: mean(|predictions - targets|)
```

### Smooth L1 Loss (Huber Loss)

Combines benefits of L1 and L2 loss.

```typescript
import { smoothL1Loss } from '@ts-torch/optim'

const loss = smoothL1Loss(predictions, targets, {
  beta: 1.0,
  reduction: 'mean',
})
```

### KL Divergence Loss

Measures divergence between probability distributions.

```typescript
import { klDivLoss } from '@ts-torch/optim'

// Input should be log probabilities
const logProbs = model.logSoftmax(logits)
const targetProbs = tensor([[0.1, 0.8, 0.1]])

const loss = klDivLoss(logProbs, targetProbs)
```

## Learning Rate Schedulers

### StepLR

Decay learning rate by gamma every stepSize epochs.

```typescript
import { StepLR } from '@ts-torch/optim'

const scheduler = new StepLR(optimizer, 30, 0.1)

for (let epoch = 0; epoch < 100; epoch++) {
  train()
  validate()
  scheduler.step()
}
// LR: 0.1 for epochs [0-29], 0.01 for [30-59], 0.001 for [60-89], etc.
```

### MultiStepLR

Decay learning rate at specific milestones.

```typescript
import { MultiStepLR } from '@ts-torch/optim'

const scheduler = new MultiStepLR(optimizer, [30, 80], 0.1)

// LR decays at epochs 30 and 80
```

### ExponentialLR

Decay learning rate exponentially every epoch.

```typescript
import { ExponentialLR } from '@ts-torch/optim'

const scheduler = new ExponentialLR(optimizer, 0.95)

// LR = initial_lr * 0.95^epoch
```

### CosineAnnealingLR

Cosine annealing schedule for smooth learning rate decay.

```typescript
import { CosineAnnealingLR } from '@ts-torch/optim'

const scheduler = new CosineAnnealingLR(optimizer, 50, 0.001)

// LR follows cosine curve from initial_lr to 0.001 over 50 epochs
```

### CosineAnnealingWarmRestarts

Cosine annealing with periodic restarts (SGDR).

```typescript
import { CosineAnnealingWarmRestarts } from '@ts-torch/optim'

const scheduler = new CosineAnnealingWarmRestarts(optimizer, 10, 2)

// Restarts every T_0 * T_mult epochs with cosine annealing
```

### ReduceLROnPlateau

Reduce learning rate when a metric stops improving.

```typescript
import { ReduceLROnPlateau } from '@ts-torch/optim'

const scheduler = new ReduceLROnPlateau(optimizer, 'min', {
  factor: 0.1, // Reduce LR by factor of 0.1
  patience: 10, // Wait 10 epochs before reducing
  threshold: 0.0001, // Minimum change to qualify as improvement
  cooldown: 0, // Epochs to wait after LR reduction
  minLr: 1e-6, // Minimum learning rate
})

for (let epoch = 0; epoch < 100; epoch++) {
  train()
  const valLoss = validate()

  // Step with metric value
  scheduler.step(valLoss)
}
```

### LinearWarmup

Linearly increase learning rate from 0 to base LR.

```typescript
import { LinearWarmup } from '@ts-torch/optim'

const scheduler = new LinearWarmup(optimizer, 1000)

for (let step = 0; step < totalSteps; step++) {
  train()
  scheduler.step()
}
// LR linearly increases for first 1000 steps
```

## Complete Training Example

```typescript
import { Adam, crossEntropyLoss, CosineAnnealingLR } from '@ts-torch/optim'
import { tensor } from '@ts-torch/core'

// Initialize model and optimizer
const model = createModel()
const optimizer = new Adam(model.parameters(), {
  lr: 0.001,
  weightDecay: 0.01,
})

// Initialize scheduler
const scheduler = new CosineAnnealingLR(optimizer, numEpochs)

// Training loop
for (let epoch = 0; epoch < numEpochs; epoch++) {
  let totalLoss = 0

  // Training phase
  model.train()
  for (const batch of trainLoader) {
    // Forward pass
    const logits = model(batch.input)
    const loss = crossEntropyLoss(logits, batch.targets)

    // Backward pass and optimization
    optimizer.zeroGrad()
    loss.backward()
    optimizer.step()

    totalLoss += loss.item()
  }

  // Validation phase
  model.eval()
  let valLoss = 0
  for (const batch of valLoader) {
    const logits = model(batch.input)
    const loss = crossEntropyLoss(logits, batch.targets)
    valLoss += loss.item()
  }

  // Update learning rate
  scheduler.step()

  console.log(`Epoch ${epoch + 1}/${numEpochs}`)
  console.log(`  Train Loss: ${totalLoss / trainLoader.length}`)
  console.log(`  Val Loss: ${valLoss / valLoader.length}`)
  console.log(`  LR: ${scheduler.getCurrentLr()[0]}`)
}
```

## Advanced Features

### Multiple Parameter Groups

Different learning rates for different parts of the model.

```typescript
const optimizer = new SGD(
  [
    { params: model.features.parameters(), lr: 0.001 },
    { params: model.classifier.parameters(), lr: 0.01 },
  ],
  { lr: 0.01 },
) // Default LR for new groups
```

### Accessing Optimizer State

```typescript
// Get optimizer state (for checkpointing)
const state = optimizer.getState()

// Load optimizer state (from checkpoint)
optimizer.loadState(state)
```

### Dynamic Learning Rate Adjustment

```typescript
// Get current learning rate
const currentLr = optimizer.learningRate

// Set new learning rate
optimizer.learningRate = 0.001
```

## API Reference

### Optimizer Base Class

All optimizers inherit from the base `Optimizer` class:

```typescript
abstract class Optimizer {
  step(): void // Perform optimization step
  zeroGrad(): void // Zero all gradients
  getState(): Map<Tensor, Record> // Get optimizer state
  loadState(state: Map): void // Load optimizer state
  addParamGroup(group: ParamGroup): void // Add parameter group
  getAllParams(): Tensor[] // Get all parameters
  get learningRate(): number // Get learning rate
  set learningRate(lr: number): void // Set learning rate
}
```

### Scheduler Base Class

All schedulers inherit from the base `LRScheduler` class:

```typescript
abstract class LRScheduler {
  step(): void // Update learning rate
  getCurrentLr(): number[] // Get current learning rates
  getLastEpoch(): number // Get last epoch number
  protected abstract getLr(): number[] // Compute new learning rates
}
```

## TypeScript Type Safety

The optim module provides full TypeScript type safety:

```typescript
import type { Tensor, Shape, DType } from '@ts-torch/core';

// Type-safe loss functions
function crossEntropyLoss<
  B extends number,
  C extends number,
  D extends DType<string>
>(
  logits: Tensor<readonly [B, C], D>,
  targets: Tensor<readonly [B], D>
): Tensor<readonly [], D>;

// Compile-time shape checking
const logits: Tensor<readonly [32, 10], typeof float32> = ...;
const targets: Tensor<readonly [32], typeof float32> = ...;
const loss = crossEntropyLoss(logits, targets); // ✓ Type-safe

// This would be a compile error:
const wrongTargets: Tensor<readonly [32, 10], typeof float32> = ...;
// crossEntropyLoss(logits, wrongTargets); // ✗ Compile error
```

## Implementation Details

### SGD Update Rule

```
velocity = momentum * velocity + (1 - dampening) * gradient
if nesterov:
    gradient = gradient + momentum * velocity
else:
    gradient = velocity
param = param - lr * gradient
```

### Adam Update Rule

```
m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
param = param - lr * m_hat / (sqrt(v_hat) + eps)
```

### Weight Decay

- **L2 Regularization (SGD, Adam)**: Added to gradient

  ```
  gradient = gradient + weight_decay * param
  ```

- **Decoupled Weight Decay (AdamW)**: Applied separately
  ```
  param = param - lr * weight_decay * param
  ```

## Performance Considerations

1. **Batch Size**: Larger batches allow more stable gradient estimates
2. **Learning Rate**: Start with recommended defaults, tune based on loss curves
3. **Weight Decay**: Prevents overfitting, typical values: 1e-4 to 1e-2
4. **Momentum**: Accelerates convergence, typical value: 0.9
5. **Gradient Clipping**: Not implemented yet, but recommended for RNNs

## Best Practices

1. **Optimizer Selection**:
   - Use Adam for most tasks (adaptive learning rate)
   - Use SGD with momentum for computer vision (often better generalization)
   - Use AdamW when using weight decay with Adam

2. **Learning Rate**:
   - Start with default values
   - Use learning rate schedulers for better convergence
   - Consider warmup for transformers

3. **Regularization**:
   - Add weight decay to prevent overfitting
   - Typical values: 1e-4 for SGD, 0.01 for AdamW

4. **Debugging**:
   - Monitor loss curves
   - Check gradient norms
   - Visualize learning rate schedule

## License

MIT
