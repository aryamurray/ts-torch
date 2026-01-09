/**
 * Integration tests for tensor factory functions
 *
 * Tests the creation of tensors using various factory functions,
 * verifying shapes, dtypes, and initial values.
 */

import { describe, it, expect } from 'vitest';
import { zeros, ones, randn, fromArray, createArange, empty } from '../factory.js';
import { DType } from '../../types/dtype.js';
import { scopedTest, expectZeros, expectOnes } from '../../test/utils.js';

describe('Tensor Factory Functions - Integration', () => {
  describe('zeros', () => {
    it('should create 1D tensor filled with zeros', () =>
      scopedTest(() => {
        const tensor = zeros([5] as const);

        expect(tensor).toHaveShape([5]);
        expect(tensor.dtype.name).toBe('float32');
        expectZeros(tensor);
      }));

    it('should create 2D tensor filled with zeros', () =>
      scopedTest(() => {
        const tensor = zeros([2, 3] as const);

        expect(tensor).toHaveShape([2, 3]);
        expect(tensor.numel).toBe(6);
        expectZeros(tensor);
      }));

    it('should create 3D tensor filled with zeros', () =>
      scopedTest(() => {
        const tensor = zeros([2, 3, 4] as const);

        expect(tensor).toHaveShape([2, 3, 4]);
        expect(tensor.numel).toBe(24);
        expect(tensor.ndim).toBe(3);
        expectZeros(tensor);
      }));

    it('should create tensor with specified dtype', () =>
      scopedTest(() => {
        const float64Tensor = zeros([2, 2] as const, DType.float64);
        expect(float64Tensor.dtype.name).toBe('float64');

        const int32Tensor = zeros([2, 2] as const, DType.int32);
        expect(int32Tensor.dtype.name).toBe('int32');
      }));

    it('should create tensor with requires_grad flag', () =>
      scopedTest(() => {
        const tensor = zeros([2, 2] as const, DType.float32, false);
        expect(tensor.requiresGrad).toBe(false);

        const gradTensor = zeros([2, 2] as const, DType.float32, true);
        expect(gradTensor.requiresGrad).toBe(true);
      }));
  });

  describe('ones', () => {
    it('should create 1D tensor filled with ones', () =>
      scopedTest(() => {
        const tensor = ones([5] as const);

        expect(tensor).toHaveShape([5]);
        expect(tensor.dtype.name).toBe('float32');
        expectOnes(tensor);
      }));

    it('should create 2D tensor filled with ones', () =>
      scopedTest(() => {
        const tensor = ones([3, 4] as const);

        expect(tensor).toHaveShape([3, 4]);
        expect(tensor.numel).toBe(12);
        expectOnes(tensor);
      }));

    it('should create tensor with specified dtype', () =>
      scopedTest(() => {
        const tensor = ones([2, 2] as const, DType.float64);
        expect(tensor.dtype.name).toBe('float64');
        expectOnes(tensor);
      }));

    it('should support requires_grad', () =>
      scopedTest(() => {
        const tensor = ones([2, 2] as const, DType.float32, true);
        expect(tensor.requiresGrad).toBe(true);
      }));
  });

  describe('empty', () => {
    it('should create uninitialized tensor with correct shape', () =>
      scopedTest(() => {
        const tensor = empty([10, 20] as const);

        expect(tensor).toHaveShape([10, 20]);
        expect(tensor.dtype.name).toBe('float32');
        expect(tensor.numel).toBe(200);
      }));

    it('should create tensor with specified dtype', () =>
      scopedTest(() => {
        const tensor = empty([5, 5] as const, DType.float64);
        expect(tensor.dtype.name).toBe('float64');
      }));

    it('should allocate correct amount of memory', () =>
      scopedTest(() => {
        const tensor = empty([100, 100] as const);
        expect(tensor.toArray()).toHaveLength(10000);
      }));
  });

  describe('randn', () => {
    it('should create tensor with random normal distribution', () =>
      scopedTest(() => {
        const tensor = randn([100] as const);

        expect(tensor).toHaveShape([100]);
        expect(tensor.dtype.name).toBe('float32');

        // Check that values are finite and vary
        expect(tensor).toBeFinite();

        const data = Array.from(tensor.toArray()).map(Number);
        const mean = data.reduce((a, b) => a + b, 0) / data.length;
        const variance =
          data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / data.length;

        // Normal distribution should have mean ~0 and variance ~1 (with tolerance)
        expect(Math.abs(mean)).toBeLessThan(0.3);
        expect(Math.abs(variance - 1)).toBeLessThan(0.5);
      }));

    it('should create 2D tensor with random values', () =>
      scopedTest(() => {
        const tensor = randn([10, 10] as const);

        expect(tensor).toHaveShape([10, 10]);
        expect(tensor.numel).toBe(100);
        expect(tensor).toBeFinite();

        // Verify values are different (not all the same)
        const data = Array.from(tensor.toArray()).map(Number);
        const unique = new Set(data);
        expect(unique.size).toBeGreaterThan(50); // At least 50 unique values
      }));

    it('should support different dtypes', () =>
      scopedTest(() => {
        const tensor = randn([5, 5] as const, DType.float64);
        expect(tensor.dtype.name).toBe('float64');
        expect(tensor).toBeFinite();
      }));

    it('should support requires_grad', () =>
      scopedTest(() => {
        const tensor = randn([2, 2] as const, DType.float32, true);
        expect(tensor.requiresGrad).toBe(true);
      }));
  });

  describe('fromArray', () => {
    it('should create 1D tensor from array', () =>
      scopedTest(() => {
        const data = [1, 2, 3, 4, 5];
        const tensor = fromArray(data, [5] as const);

        expect(tensor).toHaveShape([5]);
        expect(tensor).toBeCloseTo(data);
      }));

    it('should create 2D tensor from flat array', () =>
      scopedTest(() => {
        const data = [1, 2, 3, 4, 5, 6];
        const tensor = fromArray(data, [2, 3] as const);

        expect(tensor).toHaveShape([2, 3]);
        expect(tensor.numel).toBe(6);
        expect(tensor).toBeCloseTo(data);
      }));

    it('should create 3D tensor from flat array', () =>
      scopedTest(() => {
        const data = [1, 2, 3, 4, 5, 6, 7, 8];
        const tensor = fromArray(data, [2, 2, 2] as const);

        expect(tensor).toHaveShape([2, 2, 2]);
        expect(tensor.numel).toBe(8);
        expect(tensor).toBeCloseTo(data);
      }));

    it('should work with Float32Array', () =>
      scopedTest(() => {
        const data = new Float32Array([1.5, 2.5, 3.5, 4.5]);
        const tensor = fromArray(data, [2, 2] as const);

        expect(tensor).toHaveShape([2, 2]);
        expect(tensor).toBeCloseTo([1.5, 2.5, 3.5, 4.5]);
      }));

    it('should work with Float64Array', () =>
      scopedTest(() => {
        const data = new Float64Array([1.1, 2.2, 3.3]);
        const tensor = fromArray(data, [3] as const, DType.float64);

        expect(tensor.dtype.name).toBe('float64');
        expect(tensor).toBeCloseTo([1.1, 2.2, 3.3]);
      }));

    it('should work with Int32Array', () =>
      scopedTest(() => {
        const data = new Int32Array([10, 20, 30, 40]);
        const tensor = fromArray(data, [2, 2] as const, DType.int32);

        expect(tensor.dtype.name).toBe('int32');
        expect(tensor).toBeCloseTo([10, 20, 30, 40]);
      }));

    it('should throw error if data length does not match shape', () =>
      scopedTest(() => {
        const data = [1, 2, 3];
        expect(() => fromArray(data, [2, 2] as const)).toThrow();
      }));

    it('should support requires_grad', () =>
      scopedTest(() => {
        const data = [1, 2, 3, 4];
        const tensor = fromArray(data, [2, 2] as const, DType.float32, true);
        expect(tensor.requiresGrad).toBe(true);
      }));
  });

  describe('createArange', () => {
    it('should create range from 0 to 10', () =>
      scopedTest(() => {
        const tensor = createArange(0, 10);

        expect(tensor).toHaveShape([10]);
        expect(tensor).toBeCloseTo([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      }));

    it('should create range with custom start', () =>
      scopedTest(() => {
        const tensor = createArange(5, 10);

        expect(tensor).toHaveShape([5]);
        expect(tensor).toBeCloseTo([5, 6, 7, 8, 9]);
      }));

    it('should create range with custom step', () =>
      scopedTest(() => {
        const tensor = createArange(0, 10, 2);

        expect(tensor).toHaveShape([5]);
        expect(tensor).toBeCloseTo([0, 2, 4, 6, 8]);
      }));

    it('should create range with fractional step', () =>
      scopedTest(() => {
        const tensor = createArange(0, 1, 0.1);

        expect(tensor).toHaveShape([10]);
        const data = Array.from(tensor.toArray()).map(Number);
        expect(data[0]).toBeCloseTo(0, 1e-5);
        expect(data[5]).toBeCloseTo(0.5, 1e-5);
        expect(data[9]).toBeCloseTo(0.9, 1e-5);
      }));

    it('should support negative ranges', () =>
      scopedTest(() => {
        const tensor = createArange(-5, 0);

        expect(tensor).toHaveShape([5]);
        expect(tensor).toBeCloseTo([-5, -4, -3, -2, -1]);
      }));

    it('should support different dtypes', () =>
      scopedTest(() => {
        const tensor = createArange(0, 5, 1, DType.float64);
        expect(tensor.dtype.name).toBe('float64');
      }));

    it('should throw error for zero step', () =>
      scopedTest(() => {
        expect(() => createArange(0, 10, 0)).toThrow('Step cannot be zero');
      }));

    it('should throw error for invalid range', () =>
      scopedTest(() => {
        expect(() => createArange(10, 0, 1)).toThrow('Invalid range');
      }));
  });

  describe('tensor properties', () => {
    it('should correctly report ndim', () =>
      scopedTest(() => {
        expect(zeros([5] as const).ndim).toBe(1);
        expect(zeros([2, 3] as const).ndim).toBe(2);
        expect(zeros([2, 3, 4] as const).ndim).toBe(3);
        expect(zeros([2, 3, 4, 5] as const).ndim).toBe(4);
      }));

    it('should correctly calculate numel', () =>
      scopedTest(() => {
        expect(zeros([5] as const).numel).toBe(5);
        expect(zeros([2, 3] as const).numel).toBe(6);
        expect(zeros([2, 3, 4] as const).numel).toBe(24);
        expect(zeros([10, 10, 10] as const).numel).toBe(1000);
      }));
  });
});
