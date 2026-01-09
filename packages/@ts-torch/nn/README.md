# @ts-torch/nn

Neural network modules with advanced TypeScript type safety and compile-time shape checking.

## Features

- **Type-Safe Shape Inference**: Catch shape mismatches at compile time
- **Composable .pipe() Chaining**: Build models with fluent, type-checked composition
- **Advanced Weight Initialization**: Kaiming (He) and Xavier (Glorot) initialization
- **Functional API**: Stateless operations for custom forward passes
- **Parameter Management**: Automatic tracking of trainable parameters
- **Training/Eval Modes**: Built-in support for different model behaviors

## Installation

```bash
npm install @ts-torch/nn @ts-torch/core
```

## Quick Start

### Type-Safe .pipe() Chaining

```typescript
import { Linear, ReLU, Softmax } from '@ts-torch/nn';

// Build a classifier with type-safe chaining
const model = new Linear(784, 128)
  .pipe(new ReLU())
  .pipe(new Linear(128, 64))
  .pipe(new ReLU())
  .pipe(new Linear(64, 10))
  .pipe(new Softmax(-1));

// Type: PipedModule<readonly [number, 784], readonly [number, 10]>

// Forward pass with compile-time type checking
const input: Tensor<readonly [32, 784]> = ...;
const output = model.forward(input);
// Type: Tensor<readonly [32, 10]> ✓
```

### Compile-Time Shape Errors

TypeScript catches shape mismatches at compile time:

```typescript
const layer1 = new Linear(128, 64);
const layer2 = new Linear(256, 10); // Expects 256 inputs

// This will fail to compile!
const invalid = layer1.pipe(layer2);
// Error: Type 'readonly [number, 64]' is not assignable to 'readonly [number, 256]'
```

### Sequential Container

```typescript
import { Sequential } from "@ts-torch/nn";

// Using Sequential with explicit types
const model = new Sequential<readonly [number, 784], readonly [number, 10]>(
  new Linear(784, 256),
  new ReLU(),
  new Linear(256, 128),
  new ReLU(),
  new Linear(128, 10),
  new Softmax(-1),
);
```

### Sequential Builder (Best Type Inference)

```typescript
import { sequential } from "@ts-torch/nn";

// Builder pattern with full shape tracking
const model = sequential<readonly [number, 784]>()
  .add(new Linear(784, 256))
  .add(new ReLU())
  .add(new Linear(256, 128))
  .add(new ReLU())
  .add(new Linear(128, 10))
  .add(new Softmax(-1))
  .build();

// Type: Sequential<readonly [number, 784], readonly [number, 10]>
```

## Available Modules

### Linear Layers

```typescript
import { Linear } from "@ts-torch/nn";

// Basic usage
const fc = new Linear(784, 128);

// With options
const fc = new Linear(784, 128, {
  bias: true, // Include bias term (default: true)
  init: "kaiming_uniform", // Weight initialization strategy
  dtype: DType.float32, // Data type
});

// Initialization strategies:
// - 'kaiming_uniform': He uniform for ReLU networks
// - 'kaiming_normal': He normal for ReLU networks
// - 'xavier_uniform': Glorot uniform for tanh/sigmoid
// - 'xavier_normal': Glorot normal for tanh/sigmoid
// - 'zeros': Zero initialization (testing)
```

### Activation Functions

All activation functions preserve input shape and are type-safe:

```typescript
import { ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU } from "@ts-torch/nn";

// ReLU: max(0, x)
const relu = new ReLU();

// Sigmoid: 1 / (1 + e^(-x))
const sigmoid = new Sigmoid();

// Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
const tanh = new Tanh();

// Softmax: normalize to probability distribution
const softmax = new Softmax(-1); // dim = -1 (last dimension)

// Leaky ReLU: max(αx, x)
const leaky = new LeakyReLU(0.01); // negative_slope = 0.01

// GELU: Gaussian Error Linear Unit (for transformers)
const gelu = new GELU();
```

### Functional API

Stateless operations for custom forward passes:

```typescript
import { F } from "@ts-torch/nn";

class CustomModule extends Module {
  forward(x: Tensor) {
    x = F.linear(x, this.weight, this.bias);
    x = F.relu(x);
    x = F.dropout(x, 0.5, this.training);
    x = F.softmax(x, -1);
    return x;
  }
}

// Available functions:
// - F.relu, F.sigmoid, F.tanh, F.gelu, F.leakyRelu
// - F.softmax, F.logSoftmax
// - F.dropout
// - F.linear
// - F.normalize, F.clamp
```

## Advanced Examples

### Autoencoder

```typescript
// Encoder: 784 -> 512 -> 256 -> 128
const encoder = new Linear(784, 512)
  .pipe(new ReLU())
  .pipe(new Linear(512, 256))
  .pipe(new ReLU())
  .pipe(new Linear(256, 128));

// Decoder: 128 -> 256 -> 512 -> 784
const decoder = new Linear(128, 256)
  .pipe(new ReLU())
  .pipe(new Linear(256, 512))
  .pipe(new ReLU())
  .pipe(new Linear(512, 784))
  .pipe(new Sigmoid());

// Compose into full autoencoder
const autoencoder = encoder.pipe(decoder);
// Type: PipedModule<readonly [number, 784], readonly [number, 784]>
```

### Reusable Components

```typescript
// Define building blocks
const hiddenLayer = (inFeatures: number, outFeatures: number) =>
  new Linear(inFeatures, outFeatures).pipe(new ReLU());

const outputLayer = (inFeatures: number, numClasses: number) =>
  new Linear(inFeatures, numClasses).pipe(new Softmax(-1));

// Compose them
const classifier = hiddenLayer(784, 512)
  .pipe(hiddenLayer(512, 256))
  .pipe(hiddenLayer(256, 128))
  .pipe(outputLayer(128, 10));
```

### Parameter Management

```typescript
const model = new Linear(784, 128).pipe(new ReLU()).pipe(new Linear(128, 10));

// Get all parameters
const params = model.parameters();
console.log(`Model has ${params.length} parameter tensors`);

// Get named parameters (for debugging/analysis)
const namedParams = model.namedParameters();
for (const [name, param] of namedParams) {
  console.log(`${name}: shape=${param.data.shape}, requires_grad=${param.requiresGrad}`);
}
// Output:
// 0.weight: shape=[128, 784], requires_grad=true
// 0.bias: shape=[128], requires_grad=true
// 1.weight: shape=[10, 128], requires_grad=true
// 1.bias: shape=[10], requires_grad=true
```

### Training vs Evaluation Mode

```typescript
const model = new Linear(784, 128).pipe(new ReLU()).pipe(new Linear(128, 10));

// Training mode (enables dropout, etc.)
model.train();
const trainOutput = model.forward(input);

// Evaluation mode (disables dropout, etc.)
model.eval();
const evalOutput = model.forward(input);

// Toggle back to training
model.train(true);
```

## Type System

The module system uses advanced TypeScript generics to provide compile-time guarantees:

```typescript
/**
 * Base Module type signature
 */
abstract class Module<
  InShape extends Shape = Shape,      // Input tensor shape
  OutShape extends Shape = Shape,     // Output tensor shape
  D extends DType<string> = float32   // Data type
>

/**
 * Example: Linear layer
 */
class Linear<
  InFeatures extends number,
  OutFeatures extends number,
  D extends DType<string> = float32
> extends Module<
  readonly [number, InFeatures],      // Input: [Batch, InFeatures]
  readonly [number, OutFeatures],     // Output: [Batch, OutFeatures]
  D
>
```

### Shape Constraints

The `.pipe()` method enforces shape compatibility at compile time:

```typescript
pipe<NextOut extends Shape>(
  next: Module<OutShape, NextOut, D>
): PipedModule<InShape, OutShape, NextOut, D>
```

The `next` module's input shape must match `OutShape` (this module's output shape).

## Best Practices

### 1. Use .pipe() for Linear Compositions

```typescript
// Good: Type-safe chaining
const model = new Linear(784, 128).pipe(new ReLU()).pipe(new Linear(128, 10));

// Also good: Sequential builder
const model = sequential<readonly [number, 784]>()
  .add(new Linear(784, 128))
  .add(new ReLU())
  .add(new Linear(128, 10))
  .build();
```

### 2. Specify Exact Dimensions as Const

```typescript
const INPUT_DIM = 784 as const;
const HIDDEN_DIM = 128 as const;
const OUTPUT_DIM = 10 as const;

const model = new Linear(INPUT_DIM, HIDDEN_DIM)
  .pipe(new ReLU())
  .pipe(new Linear(HIDDEN_DIM, OUTPUT_DIM));
```

### 3. Use Type Parameters for Generic Models

```typescript
function createClassifier<InputDim extends number, OutputDim extends number>(
  inputDim: InputDim,
  outputDim: OutputDim,
) {
  return new Linear(inputDim, 256)
    .pipe(new ReLU())
    .pipe(new Linear(256, 128))
    .pipe(new ReLU())
    .pipe(new Linear(128, outputDim));
}

const mnist = createClassifier(784, 10);
const cifar = createClassifier(3072, 10);
```

### 4. Leverage Functional API in Custom Modules

```typescript
import { Module, F } from "@ts-torch/nn";

class CustomLayer extends Module<readonly [number, 128], readonly [number, 64]> {
  weight: Parameter<readonly [64, 128]>;

  forward(x: Tensor<readonly [number, 128]>): Tensor<readonly [number, 64]> {
    let out = F.linear(x, this.weight);
    out = F.relu(out);
    out = F.dropout(out, 0.5, this.training);
    return out;
  }
}
```

## Weight Initialization

Linear layers support multiple initialization strategies:

### Kaiming (He) Initialization

For ReLU networks. Based on [He et al. 2015](https://arxiv.org/abs/1502.01852).

```typescript
// Uniform: W ~ U(-√(6/fan_in), √(6/fan_in))
const layer = new Linear(784, 128, { init: "kaiming_uniform" });

// Normal: W ~ N(0, √(2/fan_in))
const layer = new Linear(784, 128, { init: "kaiming_normal" });
```

### Xavier (Glorot) Initialization

For tanh/sigmoid networks. Based on [Glorot & Bengio 2010](http://proceedings.mlr.press/v9/glorot10a.html).

```typescript
// Uniform: W ~ U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
const layer = new Linear(784, 128, { init: "xavier_uniform" });

// Normal: W ~ N(0, √(2/(fan_in+fan_out)))
const layer = new Linear(784, 128, { init: "xavier_normal" });
```

## License

MIT
