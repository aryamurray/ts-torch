import { expect } from 'vitest';
import type { Tensor } from '@ts-torch/core';

/**
 * Convert a typed array to a number array
 */
function toNumberArray(arr: ArrayLike<number | bigint>): number[] {
  const result: number[] = [];
  for (let i = 0; i < arr.length; i++) {
    const val = arr[i];
    result.push(typeof val === 'bigint' ? Number(val) : (val as number));
  }
  return result;
}

/**
 * Custom Vitest matchers for tensor assertions
 */

declare module 'vitest' {
  interface Assertion<T> {
    /**
     * Assert that a tensor has the expected shape
     */
    toHaveShape(expected: readonly number[]): T;

    /**
     * Assert that a tensor's values are close to expected values
     */
    toBeCloseTo(expected: number[], tolerance?: number): T;

    /**
     * Assert that all tensor values are finite
     */
    toBeFinite(): T;

    /**
     * Assert that a tensor has the expected dtype
     */
    toHaveDtype(expected: string): T;

    /**
     * Assert that a tensor requires gradients
     */
    toRequireGrad(): T;

    /**
     * Assert that a tensor has a gradient attached
     */
    toHaveGrad(): T;

    /**
     * Assert that two tensors are equal (shape and values)
     */
    toEqualTensor(expected: Tensor, tolerance?: number): T;
  }

  interface AsymmetricMatchersContaining {
    toHaveShape(expected: readonly number[]): unknown;
    toBeCloseTo(expected: number[], tolerance?: number): unknown;
    toBeFinite(): unknown;
    toHaveDtype(expected: string): unknown;
    toRequireGrad(): unknown;
    toHaveGrad(): unknown;
    toEqualTensor(expected: Tensor, tolerance?: number): unknown;
  }
}

function arraysEqual(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((val, idx) => val === b[idx]);
}

function arraysClose(
  a: number[],
  b: number[],
  tolerance: number = 1e-5
): boolean {
  if (a.length !== b.length) return false;
  return a.every((val, idx) => {
    const bVal = b[idx];
    return bVal !== undefined && Math.abs(val - bVal) <= tolerance;
  });
}

export const tensorMatchers = {
  toHaveShape(received: Tensor, expected: readonly number[]) {
    const pass = arraysEqual(received.shape, expected);

    return {
      pass,
      message: () =>
        pass
          ? `Expected tensor not to have shape [${expected.join(', ')}], but it does`
          : `Expected tensor to have shape [${expected.join(', ')}], but got [${received.shape.join(', ')}]`,
      actual: received.shape,
      expected,
    };
  },

  toBeCloseTo(received: Tensor | number, expected: number[] | number, numDigitsOrTolerance: number = 2) {
    // Handle number case (fallback to default behavior with vitest semantics)
    // In vitest, the third arg is numDigits (default 2), not tolerance
    if (typeof received === 'number' && typeof expected === 'number') {
      const numDigits = numDigitsOrTolerance;
      const diff = Math.abs(received - expected);
      const pass = diff < Math.pow(10, -numDigits) / 2;
      return {
        pass,
        message: () =>
          pass
            ? `Expected ${received} not to be close to ${expected}`
            : `Expected ${received} to be close to ${expected} (difference: ${diff})`,
        actual: received,
        expected,
      };
    }

    // Handle tensor case - here the third arg is a tolerance value
    const tolerance = numDigitsOrTolerance < 1 ? numDigitsOrTolerance : 1e-5;
    const actual = toNumberArray((received as Tensor).toArray());
    const expectedArr = Array.isArray(expected) ? expected : [expected];
    const pass = arraysClose(actual, expectedArr, tolerance);

    return {
      pass,
      message: () =>
        pass
          ? `Expected tensor values not to be close to [${expectedArr.join(', ')}] within tolerance ${tolerance}, but they are`
          : `Expected tensor values to be close to [${expectedArr.join(', ')}] within tolerance ${tolerance}, but got [${actual.join(', ')}]`,
      actual,
      expected: expectedArr,
    };
  },

  toBeFinite(received: Tensor) {
    const values = toNumberArray(received.toArray());
    const pass = values.every((val) => Number.isFinite(val));

    const nonFiniteValues = values.filter((val) => !Number.isFinite(val));

    return {
      pass,
      message: () =>
        pass
          ? `Expected tensor to contain non-finite values, but all values are finite`
          : `Expected all tensor values to be finite, but found: [${nonFiniteValues.join(', ')}]`,
      actual: values,
    };
  },

  toHaveDtype(received: Tensor, expected: string) {
    const actual = received.dtype.name;
    const pass = actual === expected;

    return {
      pass,
      message: () =>
        pass
          ? `Expected tensor not to have dtype "${expected}", but it does`
          : `Expected tensor to have dtype "${expected}", but got "${actual}"`,
      actual,
      expected,
    };
  },

  toRequireGrad(received: Tensor) {
    const pass = received.requiresGrad === true;

    return {
      pass,
      message: () =>
        pass
          ? `Expected tensor not to require gradients, but it does`
          : `Expected tensor to require gradients, but requiresGrad is ${received.requiresGrad}`,
      actual: received.requiresGrad,
      expected: true,
    };
  },

  toHaveGrad(received: Tensor) {
    const pass = received.grad !== null && received.grad !== undefined;

    return {
      pass,
      message: () =>
        pass
          ? `Expected tensor not to have a gradient, but it does`
          : `Expected tensor to have a gradient, but grad is ${received.grad}`,
      actual: received.grad,
    };
  },

  toEqualTensor(received: Tensor, expected: Tensor, tolerance: number = 1e-5) {
    const shapeMatch = arraysEqual(received.shape, expected.shape);

    if (!shapeMatch) {
      return {
        pass: false,
        message: () =>
          `Expected tensors to have equal shapes, but got [${received.shape.join(', ')}] and [${expected.shape.join(', ')}]`,
        actual: received.shape,
        expected: expected.shape,
      };
    }

    const actualValues = toNumberArray(received.toArray());
    const expectedValues = toNumberArray(expected.toArray());
    const valuesMatch = arraysClose(actualValues, expectedValues, tolerance);

    return {
      pass: shapeMatch && valuesMatch,
      message: () =>
        valuesMatch
          ? `Expected tensors not to be equal, but they are`
          : `Expected tensor values to be close within tolerance ${tolerance}, but got:\nActual:   [${actualValues.join(', ')}]\nExpected: [${expectedValues.join(', ')}]`,
      actual: actualValues,
      expected: expectedValues,
    };
  },
};

/**
 * Setup tensor matchers for Vitest
 * Call this in your test setup file or at the beginning of test suites
 */
export function setupTensorMatchers(): void {
  expect.extend(tensorMatchers);
}
