# ts-torch Type System

Advanced TypeScript type system for compile-time tensor shape and dtype checking. This type system enables catching shape mismatches and type errors at compile time rather than runtime, similar to shape-checking in languages like Julia or Rust's ndarray.

## Features

- **Compile-time shape validation**: Catch shape mismatches before running your code
- **Type-safe tensor operations**: MatMul, Broadcast, Reshape, Transpose, etc.
- **Dynamic dimensions**: Support for runtime-determined dimensions with `Dim<Label>`
- **DType system**: Branded types for compile-time dtype safety
- **Zero runtime overhead**: All checks happen at compile time

## File Structure

```
types/
├── dtype.ts      - Data type definitions and mappings
├── shape.ts      - Shape type operations and utilities
├── tensor.ts     - Tensor type operations (MatMul, Broadcast, etc.)
├── index.ts      - Main export file
├── examples.ts   - Comprehensive usage examples
└── README.md     - This file
```

## Quick Start

```typescript
import type { TensorType, MatMulShape, BroadcastShape } from '@ts-torch/core/types'

// Define tensor types with literal shapes
type Matrix = TensorType<[100, 50], 'float32'>
type Vector = TensorType<[50, 1], 'float32'>

// Compute result shape at compile time
type Result = MatMulShape<[100, 50], [50, 1]> // [100, 1] ✓

// This causes a compile error (incompatible dimensions)
type Invalid = MatMulShape<[100, 50], [60, 1]> // never ✗
```

## Core Concepts

### 1. DType System

The dtype system uses branded types for compile-time type safety:

```typescript
import { DType, DTypeName, DTypeElement } from '@ts-torch/core/types'

// Concrete dtype values
const dtype = DType.float32 // DType<"float32">

// Type mappings
type ArrayType = DTypeToTypedArray<'float32'> // Float32Array
type Element = DTypeElement<'float32'> // number
type Element64 = DTypeElement<'int64'> // bigint
type ElementBool = DTypeElement<'bool'> // boolean

// Type promotion
type Promoted = PromoteDType<'float32', 'int32'> // "float32"
```

**Supported dtypes:**

- `float16` - 16-bit floating point (uses Uint16Array backing)
- `float32` - 32-bit floating point
- `float64` - 64-bit floating point
- `int32` - 32-bit signed integer
- `int64` - 64-bit signed integer (BigInt)
- `bool` - Boolean (uses Uint8Array backing)
- `bfloat16` - Brain floating point (uses Uint16Array backing)

### 2. Shape Types

Shapes are represented as readonly tuples of dimension sizes:

```typescript
import type { Shape, Dim, NumElements, Rank } from '@ts-torch/core/types'

// Concrete shapes (all dimensions known at compile time)
type ImageShape = Shape<[3, 224, 224]> // CHW format
type MatrixShape = Shape<[100, 50]>
type ScalarShape = Shape<[]>

// Dynamic dimensions (known only at runtime)
type BatchDim = Dim<'batch'>
type SeqLenDim = Dim<'seq_len'>
type DynamicShape = [BatchDim, SeqLenDim, 768]

// Shape utilities
type Elems = NumElements<[2, 3, 4]> // 24
type NDim = Rank<[2, 3, 4]> // 3
```

### 3. Tensor Types

The core `TensorType` combines shape and dtype:

```typescript
import type { TensorType } from '@ts-torch/core/types'

interface TensorType<S extends Shape, D extends DTypeName> {
  readonly shape: S
  readonly dtype: D
  readonly ndim: Rank<S>
}

// Example usage
type Image = TensorType<[3, 224, 224], 'float32'>
type Embeddings = TensorType<[50000, 768], 'float16'>
```

## Tensor Operations

### MatMul - Matrix Multiplication

Computes the output shape of matrix multiplication with proper dimension checking:

```typescript
import type { MatMulShape } from '@ts-torch/core/types'

// 2D x 2D
type R1 = MatMulShape<[100, 50], [50, 20]> // [100, 20] ✓

// Batched matmul
type R2 = MatMulShape<[8, 100, 50], [50, 20]> // [8, 100, 20] ✓
type R3 = MatMulShape<[8, 100, 50], [8, 50, 20]> // [8, 100, 20] ✓

// Invalid (inner dimensions don't match)
type Invalid = MatMulShape<[100, 50], [60, 20]> // never ✗
```

**Rules:**

- Both tensors must be at least 2D
- Inner dimensions (K) must match: `[..., M, K] x [..., K, N] -> [..., M, N]`
- Batch dimensions must broadcast

### BroadcastShape - Broadcasting

Implements NumPy/PyTorch broadcasting semantics:

```typescript
import type { BroadcastShape } from '@ts-torch/core/types'

type R1 = BroadcastShape<[1, 3, 4], [2, 1, 4]> // [2, 3, 4] ✓
type R2 = BroadcastShape<[5, 1], [3, 4]> // [5, 3, 4] ✓
type R3 = BroadcastShape<[], [3, 4]> // [3, 4] ✓

// Invalid (incompatible dimensions)
type Invalid = BroadcastShape<[3, 4], [5, 6]> // never ✗
```

**Rules:**

- Dimensions are aligned from the right
- Each dimension pair must be equal or one must be 1
- Result dimension is the maximum of the two

### TransposeShape - Transpose

Swaps two dimensions:

```typescript
import type { TransposeShape } from '@ts-torch/core/types'

type R1 = TransposeShape<[2, 3, 4], 0, 2> // [4, 3, 2] ✓
type R2 = TransposeShape<[100, 50], 0, 1> // [50, 100] ✓
```

### PermuteShape - Permute Dimensions

Reorders all dimensions according to a permutation:

```typescript
import type { PermuteShape } from '@ts-torch/core/types'

// Convert NCHW to NHWC
type NCHW = [32, 3, 224, 224]
type NHWC = PermuteShape<NCHW, [0, 2, 3, 1]> // [32, 224, 224, 3] ✓
```

### ReshapeValid - Validated Reshape

Ensures element count is preserved during reshape:

```typescript
import type { ReshapeValid } from '@ts-torch/core/types'

type Valid = ReshapeValid<[2, 3, 4], [6, 4]> // [6, 4] ✓ (24 elements)
type Invalid = ReshapeValid<[2, 3, 4], [5, 5]> // never ✗ (24 ≠ 25)
```

### SqueezeShape - Remove Size-1 Dimensions

```typescript
import type { SqueezeShape } from '@ts-torch/core/types'

type R1 = SqueezeShape<[1, 3, 1, 4], 0> // [3, 1, 4] ✓
type R2 = SqueezeShape<[1, 3, 1, 4], undefined> // [3, 4] ✓ (all ones)

// Cannot squeeze non-1 dimensions
type Invalid = SqueezeShape<[2, 3, 4], 0> // [2, 3, 4] (no change)
```

### UnsqueezeShape - Add Size-1 Dimension

```typescript
import type { UnsqueezeShape } from '@ts-torch/core/types'

type R1 = UnsqueezeShape<[3, 4], 0> // [1, 3, 4] ✓
type R2 = UnsqueezeShape<[3, 4], 1> // [3, 1, 4] ✓
type R3 = UnsqueezeShape<[3, 4], 2> // [3, 4, 1] ✓
```

### ConcatShape - Concatenation

Concatenates tensors along a dimension:

```typescript
import type { ConcatShape } from '@ts-torch/core/types'

type R1 = ConcatShape<[2, 3, 4], [2, 5, 4], 1> // [2, 8, 4] ✓
type R2 = ConcatShape<[10, 20], [30, 20], 0> // [40, 20] ✓

// Invalid (other dimensions don't match)
type Invalid = ConcatShape<[2, 3, 4], [3, 3, 4], 1> // never ✗
```

### ReduceShape - Reduction Operations

Reduces along a dimension:

```typescript
import type { ReduceShape } from '@ts-torch/core/types'

type R1 = ReduceShape<[2, 3, 4], 1, false> // [2, 4] ✓
type R2 = ReduceShape<[2, 3, 4], 1, true> // [2, 1, 4] ✓ (keepdim)
```

### ExpandShape - Expand Size-1 Dimensions

```typescript
import type { ExpandShape } from '@ts-torch/core/types'

type R1 = ExpandShape<[1, 3, 4], 0, 8> // [8, 3, 4] ✓

// Cannot expand non-1 dimensions
type Invalid = ExpandShape<[2, 3, 4], 0, 8> // never ✗
```

### FlattenShape - Flatten Dimensions

Flattens a range of dimensions into a single dimension:

```typescript
import type { FlattenShape } from '@ts-torch/core/types'

type R1 = FlattenShape<[2, 3, 4, 5], 1, 3> // [2, 12, 5] ✓
type R2 = FlattenShape<[2, 3, 4], 0, 3> // [24] ✓
```

## Advanced Usage

### Dynamic Dimensions

Use `Dim<Label>` for dimensions determined at runtime:

```typescript
import type { Dim, TensorType } from '@ts-torch/core/types'

type BatchDim = Dim<'batch'>
type SeqLenDim = Dim<'seq_len'>

// Transformer architecture
type TransformerInput = TensorType<[BatchDim, SeqLenDim, 768], 'float32'>
type AttentionWeights = TensorType<[BatchDim, 12, SeqLenDim, SeqLenDim], 'float32'>
```

### Neural Network Layers

```typescript
// Linear layer
type LinearInput = [Dim<'batch'>, 512]
type LinearWeight = [512, 256]
type LinearOutput = MatMulShape<LinearInput, LinearWeight> // [batch, 256]

// Conv layer (simplified)
type ConvInput = [32, 3, 224, 224] // BCHW
type ConvOutput = [32, 64, 112, 112] // After stride=2

// Multi-head attention
type QKVShape = [Dim<'batch'>, Dim<'seq'>, 768]
type HeadsShape = [Dim<'batch'>, 12, Dim<'seq'>, 64] // 12 heads, 64 dim each
```

### Type Guards and Validation

```typescript
import type { IsBroadcastable, IsMatMulCompatible } from '@ts-torch/core/types'

// Check if shapes can broadcast
type CanBroadcast = IsBroadcastable<[1, 3], [2, 3]> // true

// Check if matmul is valid
type CanMatMul = IsMatMulCompatible<[10, 20], [20, 30]> // true
```

## Type-Level Arithmetic

The type system includes limited type-level arithmetic for shape inference:

```typescript
import type { NumElements } from '@ts-torch/core/types'

type Count1 = NumElements<[2, 3, 4]> // 24
type Count2 = NumElements<[]> // 1 (scalar)
type Count3 = NumElements<[10, 20]> // 200
```

**Limitations:**

- Multiplication is computed for literal numbers up to reasonable sizes
- For very large or computed values, falls back to `number` type
- This is a TypeScript limitation, not a design choice

## Best Practices

### 1. Use Literal Types for Static Shapes

```typescript
// Good: Literal shape types
type Image = TensorType<[3, 224, 224], 'float32'>

// Avoid: Generic number types (loses compile-time checking)
type GenericTensor = TensorType<number[], 'float32'>
```

### 2. Use Dim<Label> for Dynamic Dimensions

```typescript
// Good: Named dynamic dimensions
type BatchDim = Dim<'batch'>
type Input = [BatchDim, 512]

// Avoid: Plain number (loses semantic meaning)
type Input = [number, 512]
```

### 3. Validate Operations at Type Level

```typescript
// Good: Let type system validate
type Result = MatMulShape<S1, S2>
// If Result is never, the operation is invalid

// Also good: Use type guards
type IsValid = IsMatMulCompatible<S1, S2>
```

### 4. Document Complex Type Operations

```typescript
/**
 * Computes transformer attention output shape
 * Input: [batch, seq_len, hidden_dim]
 * Output: [batch, seq_len, hidden_dim]
 */
type AttentionOutput<Batch extends Dim<'batch'>, SeqLen extends Dim<'seq_len'>, HiddenDim extends number> = [
  Batch,
  SeqLen,
  HiddenDim,
]
```

## Limitations

1. **Type-level arithmetic**: Limited to common literal values due to TypeScript constraints
2. **Compile time**: Complex type operations may increase compilation time
3. **Error messages**: Type errors can be verbose; use type aliases to improve readability
4. **Variadic operations**: Operations like concat of arbitrary number of tensors are limited

## Examples

See `examples.ts` for comprehensive examples including:

- Basic tensor operations
- Neural network layer shapes
- Vision transformers
- BERT-style transformers
- Type-safe operation pipelines

## Integration with Runtime Code

This type system is designed to work alongside runtime tensor implementations:

```typescript
import type { TensorType, MatMulShape } from '@ts-torch/core/types'
import { Tensor } from '@ts-torch/core'

// Type-safe wrapper
class TypedTensor<S extends Shape, D extends DTypeName> extends Tensor {
  declare readonly shape: S
  declare readonly dtype: D

  matmul<S2 extends Shape>(other: TypedTensor<S2, D>): TypedTensor<MatMulShape<S, S2>, D> {
    // Runtime implementation
    return super.matmul(other) as any
  }
}
```

## Contributing

When adding new operations:

1. Add type definition in `tensor.ts`
2. Export from `index.ts`
3. Add examples in `examples.ts`
4. Document in this README
5. Include test cases demonstrating valid and invalid usage

## License

MIT
