/**
 * Example usage of @ts-torch/test-utils
 * This file demonstrates all the utilities provided by the package
 */

import { describe, test, expect, beforeAll } from 'vitest';
import {
  setupTensorMatchers,
  mockTensorFactories,
  createMockFFI,
  TensorFixtures,
  createTestTensor,
  scopedTest,
  scopedTestAsync,
} from '../src/index.js';

// Setup custom matchers
beforeAll(() => {
  setupTensorMatchers();
});

describe('Custom Matchers', () => {
  test('toHaveShape matcher', () => {
    const tensor = mockTensorFactories.zeros([2, 3]);
    expect(tensor).toHaveShape([2, 3]);
  });

  test('toBeCloseTo matcher', () => {
    const tensor = mockTensorFactories.fromArray([1.0, 2.0, 3.0], [3]);
    expect(tensor).toBeCloseTo([1.0, 2.0, 3.0]);
    expect(tensor).toBeCloseTo([1.001, 2.001, 3.001], 0.01);
  });

  test('toBeFinite matcher', () => {
    const tensor = mockTensorFactories.ones([3]);
    expect(tensor).toBeFinite();
  });

  test('toHaveDtype matcher', () => {
    const tensor = mockTensorFactories.zeros([2]);
    expect(tensor).toHaveDtype('float32');
  });

  test('toRequireGrad matcher', () => {
    const tensor = mockTensorFactories.zeros([2], true);
    expect(tensor).toRequireGrad();
  });

  test('toHaveGrad matcher', () => {
    const tensor = mockTensorFactories.fromArray([5.0], [], true);
    tensor.backward();
    expect(tensor).toHaveGrad();
  });

  test('toEqualTensor matcher', () => {
    const tensor1 = mockTensorFactories.ones([2, 2]);
    const tensor2 = mockTensorFactories.ones([2, 2]);
    expect(tensor1).toEqualTensor(tensor2);
  });
});

describe('Mock Tensor', () => {
  test('create zeros tensor', () => {
    const zeros = mockTensorFactories.zeros([2, 3]);
    expect(zeros.shape).toEqual([2, 3]);
    expect(zeros.numel()).toBe(6);
    expect(zeros.toArray()).toEqual([0, 0, 0, 0, 0, 0]);
  });

  test('create ones tensor', () => {
    const ones = mockTensorFactories.ones([3]);
    expect(ones.toArray()).toEqual([1, 1, 1]);
  });

  test('tensor operations', () => {
    const a = mockTensorFactories.fromArray([1, 2, 3], [3]);
    const b = mockTensorFactories.fromArray([4, 5, 6], [3]);

    const sum = a.add(b);
    expect(sum.toArray()).toEqual([5, 7, 9]);

    const diff = a.sub(b);
    expect(diff.toArray()).toEqual([-3, -3, -3]);

    const product = a.mul(b);
    expect(product.toArray()).toEqual([4, 10, 18]);
  });

  test('activation functions', () => {
    const tensor = mockTensorFactories.fromArray([-1, 0, 1], [3]);

    const relu = tensor.relu();
    expect(relu.toArray()).toEqual([0, 0, 1]);

    const sigmoid = mockTensorFactories.fromArray([0], []).sigmoid();
    expect(sigmoid.toArray()[0]).toBeCloseTo(0.5);
  });

  test('gradient operations', () => {
    const tensor = mockTensorFactories.fromArray([5.0], [], true);

    expect(tensor.requiresGrad).toBe(true);
    expect(tensor.grad).toBeNull();

    tensor.backward();
    expect(tensor.grad).not.toBeNull();
    expect(tensor.grad?.toArray()).toEqual([1]);

    tensor.zeroGrad();
    expect(tensor.grad).toBeNull();
  });
});

describe('Mock FFI', () => {
  test('create and use mock FFI', () => {
    const mockFFI = createMockFFI();

    // Configure mock behavior
    mockFFI.symbols.torch_zeros.mockReturnValue(123);
    mockFFI.symbols.torch_ones.mockReturnValue(456);

    // Call mocked functions
    const zerosResult = mockFFI.symbols.torch_zeros();
    const onesResult = mockFFI.symbols.torch_ones();

    expect(zerosResult).toBe(123);
    expect(onesResult).toBe(456);

    // Verify calls
    expect(mockFFI.symbols.torch_zeros).toHaveBeenCalled();
    expect(mockFFI.symbols.torch_ones).toHaveBeenCalled();

    // Reset mocks
    mockFFI.reset();
    expect(mockFFI.symbols.torch_zeros).not.toHaveBeenCalled();
  });
});

describe('Test Fixtures', () => {
  test('use predefined fixtures', () => {
    const scalar = TensorFixtures.small.scalar;
    expect(scalar.data).toEqual([42]);
    expect(scalar.shape).toEqual([]);

    const matrix = TensorFixtures.small.matrix2x2;
    expect(matrix.data).toEqual([1, 2, 3, 4]);
    expect(matrix.shape).toEqual([2, 2]);

    const identity = TensorFixtures.matrices.identity3x3;
    expect(identity.data).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  test('create custom fixtures', () => {
    const zeros = createTestTensor('zeros', [2, 3]);
    expect(zeros.data).toEqual([0, 0, 0, 0, 0, 0]);

    const ones = createTestTensor('ones', [3, 2]);
    expect(ones.data).toEqual([1, 1, 1, 1, 1, 1]);

    const sequential = createTestTensor('sequential', [2, 2]);
    expect(sequential.data).toEqual([0, 1, 2, 3]);

    const identity = createTestTensor('identity', [3, 3]);
    expect(identity.data).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });
});

describe('Scope Helper', () => {
  test('scopedTest manages tensor lifecycle', () => {
    scopedTest((scope) => {
      const t1 = mockTensorFactories.zeros([2, 3]);
      const t2 = mockTensorFactories.ones([3, 2]);

      scope.register(t1);
      scope.register(t2);

      expect(scope.count()).toBe(2);
      expect(scope.isFreed()).toBe(false);

      // Perform operations
      expect(t1.numel()).toBe(6);
      expect(t2.numel()).toBe(6);

      // Scope will auto-free after this block
    });
  });

  test('scopedTestAsync manages async tensor lifecycle', async () => {
    await scopedTestAsync(async (scope) => {
      const t1 = mockTensorFactories.randn([4, 4]);
      scope.register(t1);

      // Simulate async operation
      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(t1.shape).toEqual([4, 4]);
      expect(scope.count()).toBe(1);

      // Scope will auto-free after this block
    });
  });

  test('manual scope management', () => {
    const t1 = mockTensorFactories.zeros([2, 2]);
    const t2 = mockTensorFactories.ones([2, 2]);

    expect(t1.isFreed).toBe(false);
    expect(t2.isFreed).toBe(false);

    // Manual cleanup
    t1.free();
    t2.free();

    expect(t1.isFreed).toBe(true);
    expect(t2.isFreed).toBe(true);
  });
});

describe('Integration Example', () => {
  test('complete workflow with all utilities', () => {
    scopedTest((scope) => {
      // Use fixtures
      const fixture = TensorFixtures.matrices.identity3x3;
      const tensor = mockTensorFactories.fromArray(
        fixture.data,
        fixture.shape,
        true
      );
      scope.register(tensor);

      // Use custom matchers
      expect(tensor).toHaveShape([3, 3]);
      expect(tensor).toHaveDtype('float32');
      expect(tensor).toRequireGrad();

      // Perform operations
      const doubled = tensor.mulScalar(2);
      scope.register(doubled);

      expect(doubled).toBeCloseTo([2, 0, 0, 0, 2, 0, 0, 0, 2]);

      // Check gradient
      const loss = mockTensorFactories.fromArray([5.0], [], true);
      scope.register(loss);

      loss.backward();
      expect(loss).toHaveGrad();
    });
  });
});
