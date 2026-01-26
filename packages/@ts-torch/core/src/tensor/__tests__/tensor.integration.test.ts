/**
 * Integration tests for Tensor operations
 *
 * Tests tensor arithmetic, linear algebra, activations, and other operations
 * to ensure the FFI bindings work correctly with the native library.
 */

import { describe, it, expect } from 'vitest';
import { zeros, ones, fromArray } from '../factory.js';
import { scopedTest } from '../../test/utils.js';

describe('Tensor Operations - Integration', () => {
  describe('element-wise operations', () => {
    describe('add', () => {
      it('should add two tensors element-wise', () =>
        scopedTest(() => {
          const a = ones([2, 3] as const);
          const b = ones([2, 3] as const);
          const c = a.add(b);

          expect(c).toHaveShape([2, 3]);
          expect(c).toBeCloseTo([2, 2, 2, 2, 2, 2]); // All values should be 2
        }));

      it('should add tensors with different values', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const b = fromArray([10, 20, 30, 40], [2, 2] as const);
          const c = a.add(b);

          expect(c).toBeCloseTo([11, 22, 33, 44]);
        }));

      it('should work with 1D tensors', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3], [3] as const);
          const b = fromArray([4, 5, 6], [3] as const);
          const c = a.add(b);

          expect(c).toBeCloseTo([5, 7, 9]);
        }));
    });

    describe('sub', () => {
      it('should subtract two tensors element-wise', () =>
        scopedTest(() => {
          const a = fromArray([5, 6, 7, 8], [2, 2] as const);
          const b = ones([2, 2] as const);
          const c = a.sub(b);

          expect(c).toBeCloseTo([4, 5, 6, 7]);
        }));

      it('should handle negative results', () =>
        scopedTest(() => {
          const a = ones([2, 2] as const);
          const b = fromArray([2, 3, 4, 5], [2, 2] as const);
          const c = a.sub(b);

          expect(c).toBeCloseTo([-1, -2, -3, -4]);
        }));
    });

    describe('mul', () => {
      it('should multiply two tensors element-wise', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const b = fromArray([2, 2, 2, 2], [2, 2] as const);
          const c = a.mul(b);

          expect(c).toBeCloseTo([2, 4, 6, 8]);
        }));

      it('should handle zero multiplication', () =>
        scopedTest(() => {
          const a = ones([2, 2] as const);
          const b = zeros([2, 2] as const);
          const c = a.mul(b);

          expect(c).toBeCloseTo([0, 0, 0, 0]);
        }));
    });

    describe('div', () => {
      it('should divide two tensors element-wise', () =>
        scopedTest(() => {
          const a = fromArray([2, 4, 6, 8], [2, 2] as const);
          const b = fromArray([2, 2, 2, 2], [2, 2] as const);
          const c = a.div(b);

          expect(c).toBeCloseTo([1, 2, 3, 4]);
        }));

      it('should handle fractional division', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const b = fromArray([2, 4, 6, 8], [2, 2] as const);
          const c = a.div(b);

          expect(c).toBeCloseTo([0.5, 0.5, 0.5, 0.5]);
        }));
    });
  });

  describe('matrix operations', () => {
    describe('matmul', () => {
      it('should multiply 2x3 and 3x2 matrices', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4, 5, 6], [2, 3] as const);
          const b = fromArray([1, 2, 3, 4, 5, 6], [3, 2] as const);
          const c = a.matmul(b);

          expect(c).toHaveShape([2, 2]);
          // [1,2,3]   [1,2]     [22, 28]
          // [4,5,6] x [3,4]  =  [49, 64]
          //           [5,6]
          expect(c).toBeCloseTo([22, 28, 49, 64]);
        }));

      it('should multiply square matrices', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const b = fromArray([5, 6, 7, 8], [2, 2] as const);
          const c = a.matmul(b);

          expect(c).toHaveShape([2, 2]);
          // [1,2] x [5,6]  =  [19, 22]
          // [3,4]   [7,8]     [43, 50]
          expect(c).toBeCloseTo([19, 22, 43, 50]);
        }));

      it('should handle identity matrix multiplication', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const identity = fromArray([1, 0, 0, 1], [2, 2] as const);
          const c = a.matmul(identity);

          expect(c).toBeCloseTo([1, 2, 3, 4]);
        }));
    });

    describe('transpose', () => {
      it('should transpose 2D matrix', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4, 5, 6], [2, 3] as const);
          const b = a.transpose(0, 1);

          expect(b).toHaveShape([3, 2]);
          // [1,2,3]  =>  [1,4]
          // [4,5,6]      [2,5]
          //              [3,6]
          expect(b).toBeCloseTo([1, 4, 2, 5, 3, 6]);
        }));

      it('should transpose square matrix', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const b = a.transpose(0, 1);

          expect(b).toHaveShape([2, 2]);
          expect(b).toBeCloseTo([1, 3, 2, 4]);
        }));

      it('should handle 3D tensor transpose', () =>
        scopedTest(() => {
          const a = zeros([2, 3, 4] as const);
          const b = a.transpose(0, 2);

          expect(b).toHaveShape([4, 3, 2]);
        }));
    });

    describe('reshape', () => {
      it('should reshape 1D to 2D', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4, 5, 6], [6] as const);
          const b = a.reshape([2, 3] as const);

          expect(b).toHaveShape([2, 3]);
          expect(b).toBeCloseTo([1, 2, 3, 4, 5, 6]);
        }));

      it('should reshape 2D to 1D', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const b = a.reshape([4] as const);

          expect(b).toHaveShape([4]);
          expect(b).toBeCloseTo([1, 2, 3, 4]);
        }));

      it('should reshape to different 2D shape', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4, 5, 6], [2, 3] as const);
          const b = a.reshape([3, 2] as const);

          expect(b).toHaveShape([3, 2]);
          expect(b).toBeCloseTo([1, 2, 3, 4, 5, 6]);
        }));
    });
  });

  describe('shape operations', () => {
    it('should squeeze singleton dimensions', () =>
      scopedTest(() => {
        const a = ones([1, 2, 1, 3] as const);
        const b = a.squeeze();

        expect(b).toHaveShape([2, 3]);
      }));

    it('should unsqueeze along a dimension', () =>
      scopedTest(() => {
        const a = ones([2, 3] as const);
        const b = (a as any).unsqueeze(1) as typeof a;

        expect(b).toHaveShape([2, 1, 3]);
      }));

    it('should flatten a range of dimensions', () =>
      scopedTest(() => {
        const a = ones([2, 3, 4] as const);
        const b = a.flatten(1, 2);

        expect(b).toHaveShape([2, 12]);
        expect(b.numel).toBe(24);
      }));

    it('should permute dimensions', () =>
      scopedTest(() => {
        const a = fromArray([1, 2, 3, 4, 5, 6], [2, 3] as const);
        const b = a.permute([1, 0]);

        expect(b).toHaveShape([3, 2]);
        expect(b).toBeCloseTo([1, 4, 2, 5, 3, 6]);
      }));

    it('should split tensors into chunks', () =>
      scopedTest(() => {
        const a = fromArray([1, 2, 3, 4, 5, 6], [6] as const);
        const parts = a.split(2);

        expect(parts).toHaveLength(3);
        expect(parts[0]).toBeCloseTo([1, 2]);
        expect(parts[1]).toBeCloseTo([3, 4]);
        expect(parts[2]).toBeCloseTo([5, 6]);
      }));
  });

  describe('reduction operations', () => {
    describe('sum', () => {
      it('should sum all elements to scalar', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const s = a.sum();

          expect(s).toHaveShape([]);
          expect(s.numel).toBe(1);
          expect(s.item()).toBe(10);
        }));

      it('should sum ones tensor', () =>
        scopedTest(() => {
          const a = ones([10, 10] as const);
          const s = a.sum();

          expect(s.item()).toBe(100);
        }));

      it('should sum zeros tensor', () =>
        scopedTest(() => {
          const a = zeros([5, 5] as const);
          const s = a.sum();

          expect(s.item()).toBe(0);
        }));
    });

    describe('mean', () => {
      it('should compute mean of all elements', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const m = a.mean();

          expect(m).toHaveShape([]);
          expect(Math.abs(m.item() - 2.5)).toBeLessThan(1e-5);
        }));

      it('should compute mean of ones', () =>
        scopedTest(() => {
          const a = ones([10, 10] as const);
          const m = a.mean();

          expect(Math.abs(m.item() - 1)).toBeLessThan(1e-5);
        }));

      it('should compute mean of zeros', () =>
        scopedTest(() => {
          const a = zeros([5, 5] as const);
          const m = a.mean();

          expect(m.item()).toBe(0);
        }));
    });
  });

  describe('activation functions', () => {
    describe('relu', () => {
      it('should apply ReLU activation', () =>
        scopedTest(() => {
          const a = fromArray([-2, -1, 0, 1, 2], [5] as const);
          const b = a.relu();

          expect(b).toBeCloseTo([0, 0, 0, 1, 2]);
        }));

      it('should not affect positive values', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const b = a.relu();

          expect(b).toBeCloseTo([1, 2, 3, 4]);
        }));

      it('should zero out negative values', () =>
        scopedTest(() => {
          const a = fromArray([-1, -2, -3, -4], [2, 2] as const);
          const b = a.relu();

          expect(b).toBeCloseTo([0, 0, 0, 0]);
        }));
    });

    describe('sigmoid', () => {
      it('should apply sigmoid activation', () =>
        scopedTest(() => {
          const a = fromArray([0], [1] as const);
          const b = a.sigmoid();

          expect(Math.abs(b.item() - 0.5)).toBeLessThan(1e-5);
        }));

      it('should map large positive values close to 1', () =>
        scopedTest(() => {
          const a = fromArray([10], [1] as const);
          const b = a.sigmoid();

          expect(Math.abs(b.item() - 0.9999)).toBeLessThan(1e-4);
        }));

      it('should map large negative values close to 0', () =>
        scopedTest(() => {
          const a = fromArray([-10], [1] as const);
          const b = a.sigmoid();

          expect(Math.abs(b.item() - 0.0001)).toBeLessThan(1e-4);
        }));

      it('should work with tensors', () =>
        scopedTest(() => {
          const a = fromArray([-2, -1, 0, 1, 2], [5] as const);
          const b = a.sigmoid();

          expect(b).toBeFinite();
          const data = Array.from(b.toArray() as Iterable<number | bigint>).map(Number);
          expect(Math.abs(data[2] - 0.5)).toBeLessThan(1e-5); // sigmoid(0) = 0.5
        }));
    });

    describe('softmax', () => {
      it('should apply softmax along dimension', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3], [3] as const);
          const b = a.softmax(0);

          expect(b).toBeFinite();
          const data = Array.from(b.toArray() as Iterable<number | bigint>).map(Number);
          const sum = data.reduce((acc, val) => acc + val, 0);
          expect(Math.abs(sum - 1)).toBeLessThan(1e-5); // Probabilities should sum to 1
        }));

      it('should apply softmax to 2D tensor', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const b = a.softmax(1);

          expect(b).toBeFinite();
          expect(b).toHaveShape([2, 2]);
        }));
    });

    describe('logSoftmax', () => {
      it('should apply log-softmax', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3], [3] as const);
          const b = a.logSoftmax(0);

          expect(b).toBeFinite();
          // Log-softmax should produce negative values
          const data = Array.from(b.toArray() as Iterable<number | bigint>).map(Number);
          expect(data.every((v) => v <= 0)).toBe(true);
        }));
    });

    describe('log and exp', () => {
      it('should compute natural logarithm', () =>
        scopedTest(() => {
          const a = fromArray([1, Math.E, Math.E ** 2], [3] as const);
          const b = a.log();

          expect(b).toBeCloseTo([0, 1, 2], 1e-5);
        }));

      it('should compute exponential', () =>
        scopedTest(() => {
          const a = fromArray([0, 1, 2], [3] as const);
          const b = a.exp();

          expect(b).toBeCloseTo([1, Math.E, Math.E ** 2], 1e-5);
        }));

      it('should satisfy exp(log(x)) = x', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const b = a.log().exp();

          expect(b).toBeCloseTo([1, 2, 3, 4], 1e-5);
        }));
    });

    describe('neg', () => {
      it('should negate tensor values', () =>
        scopedTest(() => {
          const a = fromArray([1, -2, 3, -4], [2, 2] as const);
          const b = a.neg();

          expect(b).toBeCloseTo([-1, 2, -3, 4]);
        }));
    });
  });

  describe('scalar operations', () => {
    describe('addScalar', () => {
      it('should add scalar to all elements', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const b = a.addScalar(10);

          expect(b).toBeCloseTo([11, 12, 13, 14]);
        }));

      it('should handle negative scalars', () =>
        scopedTest(() => {
          const a = ones([2, 2] as const);
          const b = a.addScalar(-5);

          expect(b).toBeCloseTo([-4, -4, -4, -4]);
        }));
    });

    describe('subScalar', () => {
      it('should subtract scalar from all elements', () =>
        scopedTest(() => {
          const a = fromArray([10, 20, 30, 40], [2, 2] as const);
          const b = a.subScalar(5);

          expect(b).toBeCloseTo([5, 15, 25, 35]);
        }));
    });

    describe('mulScalar', () => {
      it('should multiply all elements by scalar', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const b = a.mulScalar(2);

          expect(b).toBeCloseTo([2, 4, 6, 8]);
        }));

      it('should handle zero multiplication', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const b = a.mulScalar(0);

          expect(b).toBeCloseTo([0, 0, 0, 0]);
        }));
    });

    describe('divScalar', () => {
      it('should divide all elements by scalar', () =>
        scopedTest(() => {
          const a = fromArray([2, 4, 6, 8], [2, 2] as const);
          const b = a.divScalar(2);

          expect(b).toBeCloseTo([1, 2, 3, 4]);
        }));

      it('should handle fractional division', () =>
        scopedTest(() => {
          const a = ones([2, 2] as const);
          const b = a.divScalar(2);

          expect(b).toBeCloseTo([0.5, 0.5, 0.5, 0.5]);
        }));
    });
  });

  describe('memory operations', () => {
    describe('clone', () => {
      it('should create independent copy', () =>
        scopedTest(() => {
          const a = fromArray([1, 2, 3, 4], [2, 2] as const);
          const b = a.clone();

          expect(b).toHaveShape([2, 2]);
          expect(b).toBeCloseTo([1, 2, 3, 4]);

          // Verify they are independent
          const c = b.addScalar(10);
          expect(c).toBeCloseTo([11, 12, 13, 14]);
          expect(a).toBeCloseTo([1, 2, 3, 4]); // Original unchanged
        }));
    });

    describe('item', () => {
      it('should extract scalar value', () =>
        scopedTest(() => {
          const a = fromArray([42], [1] as const);
          expect(a.item()).toBe(42);
        }));

      it('should throw error for non-scalar tensor', () =>
        scopedTest(() => {
          const a = fromArray([1, 2], [2] as const);
          expect(() => a.item()).toThrow();
        }));
    });
  });
});
