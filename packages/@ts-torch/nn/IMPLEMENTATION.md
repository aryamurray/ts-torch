# ts-torch/nn Implementation Summary

## Overview

This implementation provides a type-safe neural network module system for ts-torch with advanced TypeScript features including:

- Compile-time shape checking
- Type-safe .pipe() chaining
- Advanced weight initialization (Kaiming/Xavier)
- Functional API for stateless operations

## File Structure

```
packages/@ts-torch/nn/src/
├── module.ts                          # Base Module class with generics
├── functional.ts                      # Functional API (stateless ops)
├── index.ts                          # Main exports
├── modules/
│   ├── index.ts                      # Module exports
│   ├── linear.ts                     # Linear (fully connected) layer
│   ├── activation.ts                 # ReLU, Sigmoid, Tanh, Softmax, etc.
│   ├── container.ts                  # Sequential, SequentialBuilder
│   ├── conv.ts                       # Convolutional layers (existing)
│   ├── pooling.ts                    # Pooling layers (existing)
│   ├── normalization.ts              # Normalization layers (existing)
│   └── dropout.ts                    # Dropout (existing)
└── examples/
    └── type-safe-chaining.ts         # Comprehensive examples
```

## Key Features Implemented

### 1. Type-Safe Module Base Class

**File: `module.ts`**

```typescript
abstract class Module<
  InShape extends Shape = Shape,      // Input tensor shape
  OutShape extends Shape = Shape,     // Output tensor shape
  D extends DType<string> = float32   // Data type
>
```

Key methods:

- `abstract forward(input: Tensor<InShape, D>): Tensor<OutShape, D>`
- `__call__(input: Tensor<InShape, D>): Tensor<OutShape, D>` - Callable syntax
- `pipe<NextOut>(next: Module<OutShape, NextOut, D>): PipedModule<...>` - Composable chaining
- `train(mode?: boolean): this` - Set training mode
- `eval(): this` - Set evaluation mode
- `parameters(): Parameter[]` - Get all parameters
- `namedParameters(): Map<string, Parameter>` - Get named parameters
- `to(device: Device): this` - Move to device

### 2. PipedModule for Composition

**File: `module.ts`**

The `PipedModule` class enables type-safe chaining:

```typescript
class PipedModule<
  In extends Shape,
  Mid extends Shape,
  Out extends Shape,
  D extends DType<string>
> extends Module<In, Out, D>
```

This allows compile-time verification that intermediate shapes match:

```typescript
const model = new Linear(784, 128)
  .pipe(new ReLU()) // Verified: 128 matches 128
  .pipe(new Linear(128, 10)) // Verified: 128 matches 128
```

### 3. Linear Layer with Weight Initialization

**File: `modules/linear.ts`**

```typescript
class Linear<
  InFeatures extends number,
  OutFeatures extends number,
  D extends DType<string> = float32
> extends Module<
  readonly [number, InFeatures],
  readonly [number, OutFeatures],
  D
>
```

Features:

- Type-safe shape inference: `[Batch, InFeatures]` → `[Batch, OutFeatures]`
- Multiple initialization strategies:
  - **Kaiming Uniform**: `W ~ U(-√(6/fan_in), √(6/fan_in))` for ReLU networks
  - **Kaiming Normal**: `W ~ N(0, √(2/fan_in))` for ReLU networks
  - **Xavier Uniform**: `W ~ U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))` for tanh/sigmoid
  - **Xavier Normal**: `W ~ N(0, √(2/(fan_in+fan_out)))` for tanh/sigmoid
  - **Zeros**: Zero initialization (for testing)
- Optional bias term
- Parameter registration for optimizer integration

### 4. Activation Functions

**File: `modules/activation.ts`**

All activation functions are shape-preserving (input shape = output shape):

```typescript
class ReLU<S extends Shape, D extends DType<string>> extends Module<S, S, D>
class Sigmoid<S extends Shape, D extends DType<string>> extends Module<S, S, D>
class Tanh<S extends Shape, D extends DType<string>> extends Module<S, S, D>
class Softmax<S extends Shape, D extends DType<string>> extends Module<S, S, D>
class LeakyReLU<S extends Shape, D extends DType<string>> extends Module<S, S, D>
class GELU<S extends Shape, D extends DType<string>> extends Module<S, S, D>
```

This makes them trivially composable with any other layer.

### 5. Sequential Container

**File: `modules/container.ts`**

Two ways to create sequential models:

**A. Direct construction:**

```typescript
const model = new Sequential<readonly [number, 784], readonly [number, 10]>(
  new Linear(784, 128),
  new ReLU(),
  new Linear(128, 10),
)
```

**B. Builder pattern (better type inference):**

```typescript
const model = sequential<readonly [number, 784]>()
  .add(new Linear(784, 128))
  .add(new ReLU())
  .add(new Linear(128, 10))
  .build()
```

The builder tracks shape changes at each step for full type safety.

### 6. Functional API

**File: `functional.ts`**

Stateless operations for custom forward passes:

```typescript
import { F } from '@ts-torch/nn';

// All operations preserve type information
F.relu<S>(x: Tensor<S>): Tensor<S>
F.sigmoid<S>(x: Tensor<S>): Tensor<S>
F.tanh<S>(x: Tensor<S>): Tensor<S>
F.softmax<S>(x: Tensor<S>, dim: number): Tensor<S>
F.leakyRelu<S>(x: Tensor<S>, negativeSlope: number): Tensor<S>
F.gelu<S>(x: Tensor<S>): Tensor<S>
F.dropout<S>(x: Tensor<S>, p: number, training: boolean): Tensor<S>
F.linear<...>(input, weight, bias): Tensor<...>
F.logSoftmax<S>(x: Tensor<S>, dim: number): Tensor<S>
F.normalize<S>(x: Tensor<S>, p: number, dim: number, eps: number): Tensor<S>
F.clamp<S>(x: Tensor<S>, min: number | null, max: number | null): Tensor<S>
```

## Type Safety Guarantees

### Compile-Time Shape Checking

The type system catches shape mismatches at compile time:

```typescript
const layer1 = new Linear(128, 64)
const layer2 = new Linear(256, 10) // Expects 256 inputs

// TypeScript ERROR:
const invalid = layer1.pipe(layer2)
// Error: Type 'readonly [number, 64]' is not assignable to
//        type 'readonly [number, 256]'
```

### Shape Inference Through Pipelines

TypeScript tracks shapes through entire pipelines:

```typescript
const model = new Linear(784, 128)
  .pipe(new ReLU())           // [?, 784] → [?, 128]
  .pipe(new Linear(128, 64))  // [?, 128] → [?, 64]
  .pipe(new ReLU())           // [?, 64] → [?, 64]
  .pipe(new Linear(64, 10));  // [?, 64] → [?, 10]

// Type: PipedModule<readonly [number, 784], readonly [number, 10]>

const input: Tensor<readonly [32, 784]> = ...;
const output = model.forward(input);
// Type: Tensor<readonly [number, 10]>  ← TypeScript knows the shape!
```

### Generic Type Parameters

Modules can be fully generic:

```typescript
function createEncoder<InputDim extends number, LatentDim extends number>(inputDim: InputDim, latentDim: LatentDim) {
  return new Linear(inputDim, 512)
    .pipe(new ReLU())
    .pipe(new Linear(512, 256))
    .pipe(new ReLU())
    .pipe(new Linear(256, latentDim))
}

const encoder = createEncoder(784, 64)
// Type: PipedModule<readonly [number, 784], readonly [number, 64]>
```

## Design Patterns

### 1. Builder Pattern for Complex Models

```typescript
class SequentialBuilder<In, Out, D> {
  add<NextOut>(module: Module<Out, NextOut, D>): SequentialBuilder<In, NextOut, D>
  build(): Sequential<In, Out, D>
}
```

This enables fluent API with full type tracking.

### 2. Decorator Pattern for Composition

The `.pipe()` method implements the decorator pattern, wrapping modules in `PipedModule`:

```typescript
pipe<NextOut>(next: Module<OutShape, NextOut, D>): PipedModule<...>
```

### 3. Template Method Pattern

`Module` defines the structure with abstract `forward()`:

```typescript
abstract class Module {
  abstract forward(input: Tensor): Tensor

  __call__(input: Tensor): Tensor {
    return this.forward(input) // Template method
  }
}
```

### 4. Composite Pattern for Nested Modules

Modules can contain sub-modules with automatic parameter tracking:

```typescript
protected registerModule(name: string, module: Module): void
parameters(): Parameter[]  // Recursively collects from all sub-modules
```

## Parameter Management

### Parameter Registration

```typescript
class Linear extends Module {
  constructor() {
    this.weight = initWeight()
    this.bias = initBias()

    this.registerParameter('weight', this.weight)
    this.registerParameter('bias', this.bias)
  }
}
```

### Parameter Access

```typescript
// Get all parameters (flat array)
const params = model.parameters()

// Get named parameters (hierarchical)
const named = model.namedParameters()
// Map {
//   '0.weight': Parameter,
//   '0.bias': Parameter,
//   '1.weight': Parameter,
//   ...
// }
```

## Training vs Evaluation Mode

Modules track training state, which affects layers like Dropout and BatchNorm:

```typescript
model.train() // Enable training mode
model.eval() // Enable evaluation mode
model.train(false) // Explicitly disable training

// Propagates to all sub-modules automatically
```

## Weight Initialization

### Kaiming (He) Initialization

Designed for ReLU networks ([He et al. 2015](https://arxiv.org/abs/1502.01852)):

```typescript
// Uniform distribution
bound = √(6 / fan_in)
W ~ U(-bound, bound)

// Normal distribution
std = √(2 / fan_in)
W ~ N(0, std²)
```

### Xavier (Glorot) Initialization

Designed for tanh/sigmoid networks ([Glorot & Bengio 2010](http://proceedings.mlr.press/v9/glorot10a.html)):

```typescript
// Uniform distribution
bound = √(6 / (fan_in + fan_out))
W ~ U(-bound, bound)

// Normal distribution
std = √(2 / (fan_in + fan_out))
W ~ N(0, std²)
```

## Usage Examples

### Example 1: Simple Classifier

```typescript
const classifier = new Linear(784, 256)
  .pipe(new ReLU())
  .pipe(new Linear(256, 128))
  .pipe(new ReLU())
  .pipe(new Linear(128, 10))
  .pipe(new Softmax(-1))
```

### Example 2: Autoencoder

```typescript
const encoder = new Linear(784, 512).pipe(new ReLU()).pipe(new Linear(512, 128))

const decoder = new Linear(128, 512).pipe(new ReLU()).pipe(new Linear(512, 784)).pipe(new Sigmoid())

const autoencoder = encoder.pipe(decoder)
```

### Example 3: Custom Module with Functional API

```typescript
class ResidualBlock extends Module<readonly [number, 128], readonly [number, 128]> {
  fc1 = new Linear(128, 128)
  fc2 = new Linear(128, 128)

  forward(x: Tensor<readonly [number, 128]>) {
    let residual = x
    let out = this.fc1.forward(x)
    out = F.relu(out)
    out = this.fc2.forward(out)
    out = F.relu(out.add(residual)) // Skip connection
    return out
  }
}
```

## Future Enhancements

### 1. Device Management

```typescript
module.to('cuda') // Move to CUDA device
module.to('cpu') // Move to CPU
```

### 2. Actual Tensor Operations

Currently uses placeholders. Need to implement:

- Matrix multiplication for Linear layers
- Element-wise operations for activations
- Reduction operations for loss functions

### 3. Gradient Computation

```typescript
loss.backward() // Compute gradients
model.zeroGrad() // Zero all gradients
optimizer.step() // Update parameters
```

### 4. More Layer Types

- Convolutional layers (Conv2d, Conv3d)
- Recurrent layers (RNN, LSTM, GRU)
- Transformer layers (MultiHeadAttention, TransformerEncoder)
- Normalization layers (BatchNorm, LayerNorm, GroupNorm)

## Testing Strategy

### Type-Level Tests

Use TypeScript's type system to verify shape inference:

```typescript
namespace TypeTests {
  const model = new Linear(10, 20).pipe(new ReLU())
  type InputType = Parameters<typeof model.forward>[0]
  type OutputType = ReturnType<typeof model.forward>

  // These types should compile correctly
  const input: InputType = {} as Tensor<readonly [number, 10]>
  const output: OutputType = {} as Tensor<readonly [number, 20]>
}
```

### Runtime Tests

Test actual forward pass behavior once tensor operations are implemented.

## Performance Considerations

### 1. Type Erasure

All generic types are erased at runtime - no performance overhead.

### 2. Parameter Caching

Parameters are stored in Maps for O(1) lookup.

### 3. Module Hierarchy

Sub-module traversal is lazy - parameters are only collected when requested.

## Conclusion

This implementation provides a robust, type-safe foundation for neural network modules in ts-torch. The advanced TypeScript features enable catching errors at compile time that would otherwise only be discovered at runtime, significantly improving developer experience and code reliability.
