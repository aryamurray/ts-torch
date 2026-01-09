/**
 * Integration tests for autograd (automatic differentiation)
 *
 * Tests gradient computation, backpropagation, and gradient tracking
 * to ensure automatic differentiation works correctly.
 */

import { describe, it, expect } from 'vitest';
import { zeros, ones, fromArray } from '../../src/tensor/factory.js';
import { DType } from '../../src/types/dtype.js';
import { scopedTest } from '../../src/test/utils.js';

describe('Autograd - Integration', () => {
  describe('requiresGrad property', () => {
    it('should be false by default', () =>
      scopedTest(() => {
        const tensor = zeros([2, 2] as const);
        expect(tensor.requiresGrad).toBe(false);
      }));

    it('should be true when explicitly set in factory', () =>
      scopedTest(() => {
        const tensor = zeros([2, 2] as const, DType.float32, true);
        expect(tensor.requiresGrad).toBe(true);
      }));

    it('should be settable after creation', () =>
      scopedTest(() => {
        const tensor = zeros([2, 2] as const);
        expect(tensor.requiresGrad).toBe(false);

        tensor.requiresGrad = true;
        expect(tensor.requiresGrad).toBe(true);

        tensor.requiresGrad = false;
        expect(tensor.requiresGrad).toBe(false);
      }));

    it('should persist across operations', () =>
      scopedTest(() => {
        const a = zeros([2, 2] as const, DType.float32, true);
        const b = ones([2, 2] as const);
        const c = a.add(b);

        expect(a.requiresGrad).toBe(true);
        expect(c.requiresGrad).toBe(true);
      }));
  });

  describe('backward() - gradient computation', () => {
    it('should compute gradient for simple scalar multiplication', () =>
      scopedTest(() => {
        // f(x) = x^2, df/dx = 2x
        const x = fromArray([2], [1] as const, DType.float32, true);
        const y = x.mul(x);

        y.backward();

        const grad = x.grad;
        expect(grad).not.toBeNull();
        expect(grad!.item()).toBeCloseTo(4, 1e-5); // 2 * x = 2 * 2 = 4
      }));

    it('should compute gradient for linear function', () =>
      scopedTest(() => {
        // f(x) = 3x + 2, df/dx = 3
        const x = fromArray([5], [1] as const, DType.float32, true);
        const y = x.mulScalar(3).addScalar(2);

        y.backward();

        const grad = x.grad;
        expect(grad).not.toBeNull();
        expect(grad!.item()).toBeCloseTo(3, 1e-5);
      }));

    it('should compute gradient for addition', () =>
      scopedTest(() => {
        // f(a, b) = a + b, df/da = 1, df/db = 1
        const a = fromArray([1], [1] as const, DType.float32, true);
        const b = fromArray([2], [1] as const, DType.float32, true);
        const c = a.add(b);

        c.backward();

        expect(a.grad!.item()).toBeCloseTo(1, 1e-5);
        expect(b.grad!.item()).toBeCloseTo(1, 1e-5);
      }));

    it('should compute gradient for subtraction', () =>
      scopedTest(() => {
        // f(a, b) = a - b, df/da = 1, df/db = -1
        const a = fromArray([5], [1] as const, DType.float32, true);
        const b = fromArray([3], [1] as const, DType.float32, true);
        const c = a.sub(b);

        c.backward();

        expect(a.grad!.item()).toBeCloseTo(1, 1e-5);
        expect(b.grad!.item()).toBeCloseTo(-1, 1e-5);
      }));

    it('should compute gradient for multiplication', () =>
      scopedTest(() => {
        // f(a, b) = a * b, df/da = b, df/db = a
        const a = fromArray([3], [1] as const, DType.float32, true);
        const b = fromArray([4], [1] as const, DType.float32, true);
        const c = a.mul(b);

        c.backward();

        expect(a.grad!.item()).toBeCloseTo(4, 1e-5); // b
        expect(b.grad!.item()).toBeCloseTo(3, 1e-5); // a
      }));

    it('should compute gradient for division', () =>
      scopedTest(() => {
        // f(a, b) = a / b, df/da = 1/b, df/db = -a/b^2
        const a = fromArray([8], [1] as const, DType.float32, true);
        const b = fromArray([2], [1] as const, DType.float32, true);
        const c = a.div(b);

        c.backward();

        expect(a.grad!.item()).toBeCloseTo(0.5, 1e-5); // 1/2
        expect(b.grad!.item()).toBeCloseTo(-2, 1e-5); // -8/4
      }));

    it('should compute gradient through ReLU', () =>
      scopedTest(() => {
        // ReLU gradient is 1 for positive, 0 for negative
        const x1 = fromArray([2], [1] as const, DType.float32, true);
        const y1 = x1.relu();
        y1.backward();
        expect(x1.grad!.item()).toBeCloseTo(1, 1e-5);

        const x2 = fromArray([-2], [1] as const, DType.float32, true);
        const y2 = x2.relu();
        y2.backward();
        expect(x2.grad!.item()).toBeCloseTo(0, 1e-5);
      }));

    it('should compute gradient through sigmoid', () =>
      scopedTest(() => {
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        const x = fromArray([0], [1] as const, DType.float32, true);
        const y = x.sigmoid();

        y.backward();

        // sigmoid(0) = 0.5, gradient = 0.5 * 0.5 = 0.25
        expect(x.grad!.item()).toBeCloseTo(0.25, 1e-5);
      }));

    it('should compute gradient for chained operations', () =>
      scopedTest(() => {
        // f(x) = (x + 1) * 2
        // f(x) = 2x + 2
        // df/dx = 2
        const x = fromArray([3], [1] as const, DType.float32, true);
        const y = x.addScalar(1).mulScalar(2);

        y.backward();

        expect(x.grad!.item()).toBeCloseTo(2, 1e-5);
      }));

    it('should compute gradient for nested operations', () =>
      scopedTest(() => {
        // f(x) = ((x * 2) + 3) * 4
        // f(x) = 8x + 12
        // df/dx = 8
        const x = fromArray([1], [1] as const, DType.float32, true);
        const y = x.mulScalar(2).addScalar(3).mulScalar(4);

        y.backward();

        expect(x.grad!.item()).toBeCloseTo(8, 1e-5);
      }));

    it('should work with matrix operations', () =>
      scopedTest(() => {
        const x = fromArray([1, 2], [2, 1] as const, DType.float32, true);
        const w = fromArray([3, 4], [1, 2] as const, DType.float32, true);
        const y = x.matmul(w); // [1, 2] x [3, 4] = [11]

        y.backward();

        // Gradients should be computed
        expect(x.grad).not.toBeNull();
        expect(w.grad).not.toBeNull();
      }));
  });

  describe('zeroGrad() - gradient clearing', () => {
    it('should clear gradients', () =>
      scopedTest(() => {
        const x = fromArray([2], [1] as const, DType.float32, true);
        const y = x.mul(x);

        y.backward();
        expect(x.grad).not.toBeNull();
        expect(x.grad!.item()).toBeCloseTo(4, 1e-5);

        x.zeroGrad();
        const gradAfterZero = x.grad;
        // After zeroing, gradient should either be null or zero
        if (gradAfterZero !== null) {
          expect(gradAfterZero.item()).toBe(0);
        }
      }));

    it('should allow multiple backward passes with zeroGrad', () =>
      scopedTest(() => {
        const x = fromArray([3], [1] as const, DType.float32, true);

        // First pass
        const y1 = x.mul(x);
        y1.backward();
        const grad1 = x.grad!.item();
        expect(grad1).toBeCloseTo(6, 1e-5); // 2 * 3 = 6

        // Clear gradients
        x.zeroGrad();

        // Second pass with different computation
        const y2 = x.mulScalar(2);
        y2.backward();
        const grad2 = x.grad!.item();
        expect(grad2).toBeCloseTo(2, 1e-5); // Should be 2, not accumulated
      }));
  });

  describe('detach() - gradient removal', () => {
    it('should create tensor without gradient tracking', () =>
      scopedTest(() => {
        const x = fromArray([2], [1] as const, DType.float32, true);
        expect(x.requiresGrad).toBe(true);

        const y = x.detach();
        expect(y.requiresGrad).toBe(false);
      }));

    it('should copy values but not gradients', () =>
      scopedTest(() => {
        const x = fromArray([3], [1] as const, DType.float32, true);
        const y = x.detach();

        expect(y.item()).toBe(3);
        expect(y.requiresGrad).toBe(false);
      }));

    it('should break gradient flow', () =>
      scopedTest(() => {
        const x = fromArray([2], [1] as const, DType.float32, true);
        const y = x.mul(x);
        const z = y.detach(); // Detach breaks gradient flow
        const w = z.addScalar(1);

        // This should only compute gradients up to y, not x
        w.backward();

        // x should not receive gradients through detached tensor
        expect(x.grad).toBeNull();
      }));

    it('should allow mix of tracked and non-tracked tensors', () =>
      scopedTest(() => {
        const tracked = fromArray([2], [1] as const, DType.float32, true);
        const notTracked = fromArray([3], [1] as const);

        const result = tracked.mul(notTracked);
        expect(result.requiresGrad).toBe(true);

        result.backward();
        expect(tracked.grad).not.toBeNull();
      }));
  });

  describe('grad property', () => {
    it('should be null before backward', () =>
      scopedTest(() => {
        const x = fromArray([2], [1] as const, DType.float32, true);
        expect(x.grad).toBeNull();
      }));

    it('should be non-null after backward', () =>
      scopedTest(() => {
        const x = fromArray([2], [1] as const, DType.float32, true);
        const y = x.mul(x);

        y.backward();
        expect(x.grad).not.toBeNull();
      }));

    it('should have same shape as tensor', () =>
      scopedTest(() => {
        const x = fromArray([1, 2, 3, 4], [2, 2] as const, DType.float32, true);
        const y = x.sum();

        y.backward();

        const grad = x.grad;
        expect(grad).not.toBeNull();
        expect(grad!).toHaveShape([2, 2]);
      }));
  });

  describe('gradient accumulation', () => {
    it('should accumulate gradients from multiple backward passes', () =>
      scopedTest(() => {
        const x = fromArray([1], [1] as const, DType.float32, true);

        // First computation
        const y1 = x.mulScalar(2);
        y1.backward();
        const grad1 = x.grad!.item();

        // Second computation without zeroGrad
        const y2 = x.mulScalar(3);
        y2.backward();
        const grad2 = x.grad!.item();

        // Gradients should accumulate: 2 + 3 = 5
        expect(grad2).toBeGreaterThan(grad1);
      }));
  });

  describe('complex gradient computation', () => {
    it('should compute gradients for neural network-like operations', () =>
      scopedTest(() => {
        // Simple y = W*x + b computation
        const x = fromArray([1, 2], [2, 1] as const, DType.float32, true);
        const W = fromArray([0.5, 0.3], [1, 2] as const, DType.float32, true);
        const b = fromArray([0.1], [1, 1] as const, DType.float32, true);

        const y = x.matmul(W).add(b); // y = [1, 2] * [0.5, 0.3]^T + 0.1 = 1.1 + 0.1 = 1.2

        y.backward();

        // All parameters should have gradients
        expect(x.grad).not.toBeNull();
        expect(W.grad).not.toBeNull();
        expect(b.grad).not.toBeNull();
      }));

    it('should compute gradients through activation functions', () =>
      scopedTest(() => {
        // f(x) = sigmoid(x * 2)
        const x = fromArray([0.5], [1] as const, DType.float32, true);
        const y = x.mulScalar(2).sigmoid();

        y.backward();

        const grad = x.grad;
        expect(grad).not.toBeNull();
        expect(grad!).toBeFinite();
        expect(grad!.item()).toBeGreaterThan(0);
      }));

    it('should handle reduction operations', () =>
      scopedTest(() => {
        const x = fromArray([1, 2, 3, 4], [2, 2] as const, DType.float32, true);
        const y = x.sum();

        y.backward();

        const grad = x.grad;
        expect(grad).not.toBeNull();
        expect(grad!).toHaveShape([2, 2]);

        // Sum gradient should be all ones
        const gradData = Array.from(grad!.toArray()).map(Number);
        gradData.forEach((val) => {
          expect(val).toBeCloseTo(1, 1e-5);
        });
      }));

    it('should handle mean reduction', () =>
      scopedTest(() => {
        const x = fromArray([2, 4, 6, 8], [2, 2] as const, DType.float32, true);
        const y = x.mean();

        y.backward();

        const grad = x.grad;
        expect(grad).not.toBeNull();

        // Mean gradient should be 1/n for each element
        const gradData = Array.from(grad!.toArray()).map(Number);
        gradData.forEach((val) => {
          expect(val).toBeCloseTo(0.25, 1e-5); // 1/4
        });
      }));
  });
});
