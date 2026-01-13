/**
 * Test utilities for @ts-torch/core
 *
 * Provides custom matchers and helpers for testing tensor operations
 */

import { expect } from 'vitest';
import type { Tensor } from '../tensor/tensor.js';
import type { Shape } from '../types/shape.js';
import type { DType } from '../types/dtype.js';
import { run } from '../memory/scope.js';

/**
 * Custom matcher interface for TypeScript
 */
interface CustomMatchers<R = unknown> {
  toHaveShape(expected: readonly number[]): R;
  toBeCloseTo(expected: number | number[], tolerance?: number): R;
  toBeFinite(): R;
  toAllBeFinite(): R;
}

declare module 'vitest' {
  interface Assertion<T = any> extends CustomMatchers<T> {}
  interface AsymmetricMatchersContaining extends CustomMatchers {}
}

/**
 * Setup custom tensor matchers for vitest
 * Call this in your vitest.setup.ts file
 */
export function setupTensorMatchers(): void {
  expect.extend({
    /**
     * Check if tensor has expected shape
     */
    toHaveShape(received: Tensor<Shape, DType<string>>, expected: readonly number[]) {
      const { isNot } = this;
      const pass = received.shape.length === expected.length &&
        received.shape.every((dim, i) => dim === expected[i]);

      return {
        pass,
        message: () =>
          isNot
            ? `Expected tensor not to have shape [${expected.join(', ')}], but it does`
            : `Expected tensor to have shape [${expected.join(', ')}], but got [${received.shape.join(', ')}]`,
      };
    },

    /**
     * Check if tensor values are close to expected values (within tolerance)
     * Also handles plain numbers by falling back to built-in behavior
     */
    toBeCloseTo(
      received: Tensor<Shape, DType<string>> | number,
      expected: number | number[],
      tolerance = 1e-5,
    ) {
      const { isNot } = this;

      // If received is a plain number, use simple comparison
      if (typeof received === 'number') {
        const expectedNum = Array.isArray(expected) ? expected[0]! : expected;
        const pass = Math.abs(received - expectedNum) <= tolerance;
        return {
          pass,
          message: () =>
            isNot
              ? `Expected ${received} not to be close to ${expectedNum} (tolerance: ${tolerance})`
              : `Expected ${received} to be close to ${expectedNum} (tolerance: ${tolerance})`,
        };
      }

      const data = received.toArray();
      const expectedArray = Array.isArray(expected) ? expected : [expected];

      if (data.length !== expectedArray.length && !Array.isArray(expected)) {
        // If expected is a single number, check if all values are close to it
        const allClose = Array.from(data as Iterable<number | bigint>).every((val) =>
          Math.abs(Number(val) - expected) <= tolerance,
        );

        return {
          pass: allClose,
          message: () =>
            isNot
              ? `Expected tensor values not to be close to ${expected} (tolerance: ${tolerance})`
              : `Expected all tensor values to be close to ${expected} (tolerance: ${tolerance}), but some values differ`,
        };
      }

      const allClose = Array.from(data as Iterable<number | bigint>).every((val, i) =>
        Math.abs(Number(val) - expectedArray[i]!) <= tolerance,
      );

      return {
        pass: allClose,
        message: () =>
          isNot
            ? `Expected tensor values not to be close to [${expectedArray.join(', ')}] (tolerance: ${tolerance})`
            : `Expected tensor values to be close to [${expectedArray.join(', ')}] (tolerance: ${tolerance}), but got [${Array.from(data as Iterable<number | bigint>).join(', ')}]`,
      };
    },

    /**
     * Check if tensor contains only finite values (no NaN or Infinity)
     */
    toBeFinite(received: Tensor<Shape, DType<string>>) {
      const { isNot } = this;
      const data = received.toArray();
      const allFinite = Array.from(data as Iterable<number | bigint>).every((val) => Number.isFinite(Number(val)));

      return {
        pass: allFinite,
        message: () =>
          isNot
            ? `Expected tensor to contain non-finite values, but all values are finite`
            : `Expected all tensor values to be finite, but found NaN or Infinity`,
      };
    },

    /**
     * Alias for toBeFinite for clarity
     */
    toAllBeFinite(received: Tensor<Shape, DType<string>>) {
      return (this as any).toBeFinite(received);
    },
  });
}

/**
 * Wrapper for test functions that automatically manages tensor memory scopes
 *
 * @param fn - Test function to execute
 * @returns Test result
 *
 * @example
 * ```ts
 * import { device } from '@ts-torch/core'
 * const cpu = device.cpu()
 * it('should add tensors', () => scopedTest(() => {
 *   const a = cpu.ones([2, 3] as const)
 *   const b = cpu.ones([2, 3] as const)
 *   const c = a.add(b)
 *   expect(c).toHaveShape([2, 3])
 * }))
 * ```
 */
export function scopedTest<T>(fn: () => T): T {
  return run(() => {
    const result = fn();
    // If result is a tensor, escape it so it survives the scope
    if (result && typeof result === 'object' && 'escape' in result) {
      (result as any).escape();
    }
    return result;
  });
}

/**
 * Wrapper for async test functions with scoped memory management
 *
 * @param fn - Async test function to execute
 * @returns Promise of test result
 *
 * @example
 * ```ts
 * import { device } from '@ts-torch/core'
 * const cpu = device.cpu()
 * it('should process async', async () => await scopedTestAsync(async () => {
 *   const data = await loadData()
 *   const tensor = cpu.fromArray(data, [10, 10] as const)
 *   expect(tensor).toHaveShape([10, 10])
 * }))
 * ```
 */
export async function scopedTestAsync<T>(fn: () => Promise<T>): Promise<T> {
  return run(() => fn());
}

/**
 * Helper to create a tensor and compare its values
 *
 * @param tensor - Tensor to compare
 * @param expectedValues - Expected values (flat array)
 * @param tolerance - Tolerance for floating point comparison
 */
export function expectTensorValues(
  tensor: Tensor<Shape, DType<string>>,
  expectedValues: number[],
  tolerance = 1e-5,
): void {
  const data = Array.from(tensor.toArray() as Iterable<number | bigint>).map(Number);
  expect(data).toHaveLength(expectedValues.length);
  data.forEach((val, i) => {
    expect(Math.abs(val - expectedValues[i]!)).toBeLessThanOrEqual(tolerance);
  });
}

/**
 * Helper to check if tensor is all zeros
 */
export function expectZeros(tensor: Tensor<Shape, DType<string>>): void {
  const data = Array.from(tensor.toArray() as Iterable<number | bigint>).map(Number);
  data.forEach((val) => {
    expect(val).toBe(0);
  });
}

/**
 * Helper to check if tensor is all ones
 */
export function expectOnes(tensor: Tensor<Shape, DType<string>>): void {
  const data = Array.from(tensor.toArray() as Iterable<number | bigint>).map(Number);
  data.forEach((val) => {
    expect(val).toBe(1);
  });
}
