# @ts-torch/core Tests

This directory contains integration tests for the @ts-torch/core package.

## Test Structure

```
packages/@ts-torch/core/
├── vitest.config.ts           # Vitest configuration
├── vitest.setup.ts            # Test setup (registers custom matchers)
├── src/
│   ├── test/
│   │   └── utils.ts           # Test utilities and custom matchers
│   └── tensor/
│       └── __tests__/
│           ├── factory.integration.test.ts   # Factory function tests
│           └── tensor.integration.test.ts    # Tensor operation tests
└── __tests__/
    └── integration/
        └── autograd.test.ts   # Autograd/gradient tests
```

## Test Files

### 1. Factory Tests (`factory.integration.test.ts`)
Tests tensor creation functions:
- `zeros()` - Create zero-filled tensors
- `ones()` - Create one-filled tensors
- `empty()` - Create uninitialized tensors
- `randn()` - Create random normal tensors
- `fromArray()` - Create tensors from arrays
- `createArange()` - Create range tensors

### 2. Tensor Operation Tests (`tensor.integration.test.ts`)
Tests tensor operations:
- **Element-wise**: `add`, `sub`, `mul`, `div`
- **Matrix ops**: `matmul`, `transpose`, `reshape`
- **Reductions**: `sum`, `mean`
- **Activations**: `relu`, `sigmoid`, `softmax`, `logSoftmax`, `log`, `exp`, `neg`
- **Scalar ops**: `addScalar`, `subScalar`, `mulScalar`, `divScalar`
- **Memory ops**: `clone`, `item`

### 3. Autograd Tests (`autograd.test.ts`)
Tests automatic differentiation:
- `requiresGrad` property getter/setter
- `backward()` gradient computation
- `zeroGrad()` gradient clearing
- `detach()` gradient removal
- Gradient accumulation
- Complex gradient flows

## Custom Matchers

The test suite includes custom vitest matchers for tensor testing:

```typescript
// Check tensor shape
expect(tensor).toHaveShape([2, 3]);

// Check tensor values (with tolerance)
expect(tensor).toBeCloseTo([1.0, 2.0, 3.0], 1e-5);

// Check all values are finite
expect(tensor).toBeFinite();
expect(tensor).toAllBeFinite();
```

## Test Utilities

### `scopedTest(fn)`
Wraps test functions to automatically manage tensor memory:

```typescript
it('should add tensors', () => scopedTest(() => {
  const a = torch.ones([2, 3]);
  const b = torch.ones([2, 3]);
  const c = a.add(b);
  expect(c).toHaveShape([2, 3]);
}));
```

### Helper Functions
- `expectTensorValues(tensor, expected, tolerance)` - Compare tensor values
- `expectZeros(tensor)` - Verify tensor is all zeros
- `expectOnes(tensor)` - Verify tensor is all ones

## Running Tests

```bash
# Run all tests
bun test

# Run tests in watch mode
bun test:watch

# Run with coverage
bun test:coverage

# Run only @ts-torch/core tests
cd packages/@ts-torch/core
bun test

# Run specific test file
bun test factory.integration.test.ts
```

## Configuration

### Vitest Config
- **Name**: `@ts-torch/core`
- **Environment**: Node.js
- **Timeout**: 30 seconds (for FFI operations)
- **Pool**: Forks with single fork (required for FFI)
- **Coverage**: V8 provider with 70% thresholds

### Setup
The `vitest.setup.ts` file automatically:
1. Registers custom tensor matchers
2. Configures FFI library paths via root setup

## Test Coverage

The test suite covers:
- ✓ Tensor factory functions (zeros, ones, randn, etc.)
- ✓ Element-wise arithmetic operations
- ✓ Matrix operations (matmul, transpose, reshape)
- ✓ Reduction operations (sum, mean)
- ✓ Activation functions (relu, sigmoid, softmax, etc.)
- ✓ Scalar operations
- ✓ Autograd (backward, gradients, detach)
- ✓ Memory management (clone, escape)
- ✓ Shape and dtype validation

## Notes

- Tests use `scopedTest()` wrapper for automatic memory management
- FFI operations require single-fork pool configuration
- Custom matchers provide tensor-specific assertions
- Integration tests verify native library bindings work correctly
