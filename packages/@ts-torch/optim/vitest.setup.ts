/**
 * Vitest setup file for @ts-torch/optim
 *
 * This file sets up custom matchers and utilities for testing optimizers.
 */

import { expect } from 'vitest'
import type { Tensor } from '@ts-torch/core'

/**
 * Custom matchers for tensor comparisons
 */
interface CustomMatchers<R = unknown> {
  toBeCloseTo(expected: Tensor, precision?: number): R
  toHaveShape(expected: readonly number[]): R
}

declare module 'vitest' {
  interface Assertion<T = any> extends CustomMatchers<T> {}
  interface AsymmetricMatchersContaining extends CustomMatchers {}
}

/**
 * Setup custom tensor matchers
 *
 * Note: This is a placeholder implementation. In a real setup, you would
 * implement these matchers based on your tensor library's API.
 */
export function setupTensorMatchers() {
  expect.extend({
    toBeCloseTo(received: Tensor, expected: Tensor, precision = 1e-6) {
      // Placeholder implementation
      // In a real scenario, you would compare tensor values element-wise
      const pass = true // Implement actual comparison logic

      return {
        pass,
        message: () =>
          pass
            ? `expected tensor not to be close to ${expected}`
            : `expected tensor to be close to ${expected} within precision ${precision}`,
      }
    },

    toHaveShape(received: Tensor, expected: readonly number[]) {
      // Placeholder implementation
      const receivedShape = (received as any).shape || []
      const pass = JSON.stringify(receivedShape) === JSON.stringify(expected)

      return {
        pass,
        message: () =>
          pass
            ? `expected tensor not to have shape ${JSON.stringify(expected)}`
            : `expected tensor to have shape ${JSON.stringify(expected)}, but got ${JSON.stringify(receivedShape)}`,
      }
    },
  })
}

// Call setup automatically
setupTensorMatchers()
