# @ts-torch/test-utils

Shared test utilities for the ts-torch monorepo.

## Features

- **Custom Vitest Matchers** - Tensor-specific assertions for testing
- **Mock Implementations** - MockTensor and MockFFI for testing without native bindings
- **Test Fixtures** - Pre-defined tensor data patterns for common test scenarios
- **Scope Helpers** - Automatic tensor lifecycle management in tests

## Installation

This is a workspace-internal package. Add it to your package dependencies:

```json
{
  "devDependencies": {
    "@ts-torch/test-utils": "workspace:*"
  }
}
```

## Usage

### Custom Matchers

```typescript
import { setupTensorMatchers } from '@ts-torch/test-utils';
import { describe, test, beforeAll } from 'vitest';

beforeAll(() => {
  setupTensorMatchers();
});

test('tensor assertions', () => {
  const tensor = torch.zeros([2, 3]);

  expect(tensor).toHaveShape([2, 3]);
  expect(tensor).toHaveDtype('float32');
  expect(tensor).toBeCloseTo([0, 0, 0, 0, 0, 0]);
});
```

Available matchers:
- `toHaveShape(expected: number[])` - Assert tensor shape
- `toBeCloseTo(expected: number[], tolerance?: number)` - Assert tensor values are close
- `toBeFinite()` - Assert all values are finite
- `toHaveDtype(expected: string)` - Assert tensor dtype
- `toRequireGrad()` - Assert tensor requires gradients
- `toHaveGrad()` - Assert tensor has gradient
- `toEqualTensor(expected: Tensor, tolerance?: number)` - Compare two tensors

### Mock Tensor

```typescript
import { MockTensor, mockTensorFactories } from '@ts-torch/test-utils/mocks';

// Create mock tensors
const zeros = mockTensorFactories.zeros([2, 3]);
const ones = mockTensorFactories.ones([2, 3]);
const random = mockTensorFactories.randn([2, 3]);
const custom = mockTensorFactories.fromArray([1, 2, 3], [3]);

// Use mock tensors
const sum = zeros.add(ones);
const product = ones.mul(random);
```

### Mock FFI

```typescript
import { createMockFFI } from '@ts-torch/test-utils/mocks';

const mockFFI = createMockFFI();

// Configure mock behavior
mockFFI.symbols.torch_zeros.mockReturnValue(/* ... */);

// Reset all mocks
mockFFI.reset();
```

### Test Fixtures

```typescript
import { TensorFixtures, createTestTensor } from '@ts-torch/test-utils/fixtures';

// Use pre-defined fixtures
const identity = TensorFixtures.matrices.identity3x3;
const batch = TensorFixtures.batched.batchVectors;

// Create custom fixtures
const sequential = createTestTensor('sequential', [3, 4]);
const zeros = createTestTensor('zeros', [2, 2]);
```

### Scope Helper

```typescript
import { scopedTest, ScopeTestHelper } from '@ts-torch/test-utils/helpers';

test('with automatic cleanup', () => {
  scopedTest((scope) => {
    const t1 = scope.register(torch.zeros([2, 3]));
    const t2 = scope.register(torch.ones([2, 3]));

    // Test logic here

    // Tensors are automatically freed at the end
  });
});

test('with async cleanup', async () => {
  await scopedTestAsync(async (scope) => {
    const t1 = scope.register(torch.zeros([2, 3]));

    // Async test logic here
    await someAsyncOperation();

    // Tensors are automatically freed at the end
  });
});
```

## API Reference

### Matchers

See [tensor-matchers.ts](./src/matchers/tensor-matchers.ts) for full API documentation.

### Mocks

See [tensor-mock.ts](./src/mocks/tensor-mock.ts) and [ffi-mock.ts](./src/mocks/ffi-mock.ts) for full API documentation.

### Fixtures

See [tensor-fixtures.ts](./src/fixtures/tensor-fixtures.ts) for all available fixtures.

### Helpers

See [scope-helper.ts](./src/helpers/scope-helper.ts) for scope management utilities.

## License

MIT
