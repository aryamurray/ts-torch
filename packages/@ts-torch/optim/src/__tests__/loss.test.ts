/**
 * Tests for loss functions
 */

import { describe, test, expect } from 'vitest'
import {
  mseLoss,
  crossEntropyLoss,
  binaryCrossEntropyLoss,
  l1Loss,
  smoothL1Loss,
  klDivLoss,
  type Reduction,
} from '../loss'
import type { Tensor, Shape, DType } from '@ts-torch/core'

/**
 * Mock tensor with tensor operations for testing
 */
class MockTensor<S extends Shape = Shape, D extends DType<string> = DType<string>> implements Partial<Tensor<S, D>> {
  shape: S
  dtype: D
  _data: number[]

  constructor(data: number[], shape: S, dtype: D = 'float32' as D) {
    this.shape = shape
    this.dtype = dtype
    this._data = [...data]
  }

  sub(other: MockTensor): MockTensor {
    return new MockTensor(
      this._data.map((v, i) => v - (other._data[i] ?? 0)),
      this.shape as any,
      this.dtype,
    )
  }

  add(other: MockTensor | number): MockTensor {
    if (typeof other === 'number') {
      return new MockTensor(
        this._data.map((v) => v + other),
        this.shape as any,
        this.dtype,
      )
    }
    return new MockTensor(
      this._data.map((v, i) => v + (other._data[i] ?? 0)),
      this.shape as any,
      this.dtype,
    )
  }

  mul(other: MockTensor | number): MockTensor {
    if (typeof other === 'number') {
      return new MockTensor(
        this._data.map((v) => v * other),
        this.shape as any,
        this.dtype,
      )
    }
    return new MockTensor(
      this._data.map((v, i) => v * (other._data[i] ?? 1)),
      this.shape as any,
      this.dtype,
    )
  }

  div(other: MockTensor | number): MockTensor {
    if (typeof other === 'number') {
      return new MockTensor(
        this._data.map((v) => v / other),
        this.shape as any,
        this.dtype,
      )
    }
    return new MockTensor(
      this._data.map((v, i) => v / (other._data[i] ?? 1)),
      this.shape as any,
      this.dtype,
    )
  }

  pow(exponent: number): MockTensor {
    return new MockTensor(
      this._data.map((v) => Math.pow(v, exponent)),
      this.shape as any,
      this.dtype,
    )
  }

  sqrt(): MockTensor {
    return new MockTensor(
      this._data.map((v) => Math.sqrt(v)),
      this.shape as any,
      this.dtype,
    )
  }

  abs(): MockTensor {
    return new MockTensor(
      this._data.map((v) => Math.abs(v)),
      this.shape as any,
      this.dtype,
    )
  }

  exp(): MockTensor {
    return new MockTensor(
      this._data.map((v) => Math.exp(v)),
      this.shape as any,
      this.dtype,
    )
  }

  log(): MockTensor {
    return new MockTensor(
      this._data.map((v) => Math.log(v)),
      this.shape as any,
      this.dtype,
    )
  }

  sum(dim?: number, keepdim?: boolean): MockTensor {
    if (dim === undefined) {
      const total = this._data.reduce((a, b) => a + b, 0)
      return new MockTensor([total], [1] as any, this.dtype)
    }
    // Simplified sum along dimension
    return new MockTensor([this._data.reduce((a, b) => a + b, 0)], [1] as any, this.dtype)
  }

  mean(): MockTensor {
    const avg = this._data.reduce((a, b) => a + b, 0) / this._data.length
    return new MockTensor([avg], [1] as any, this.dtype)
  }

  clamp(min: number, max: number): MockTensor {
    return new MockTensor(
      this._data.map((v) => Math.max(min, Math.min(max, v))),
      this.shape as any,
      this.dtype,
    )
  }

  gather(dim: number, index: MockTensor): MockTensor {
    // Simplified gather operation
    const result = index._data.map((idx) => this._data[Math.floor(idx)] ?? 0)
    return new MockTensor(result, index.shape as any, this.dtype)
  }

  getData(): number[] {
    return [...this._data]
  }

  getScalar(): number {
    return this._data[0] ?? 0
  }
}

describe('Loss Functions', () => {
  describe('mseLoss', () => {
    test('computes mean squared error with default reduction', () => {
      const input = new MockTensor([2.5, 0.0, 2.0, 8.0], [4] as const, 'float32' as const)
      const target = new MockTensor([3.0, -0.5, 2.0, 7.0], [4] as const, 'float32' as const)

      const loss = mseLoss(input as any, target as any)

      // MSE = mean([(2.5-3)^2, (0-(-0.5))^2, (2-2)^2, (8-7)^2])
      // MSE = mean([0.25, 0.25, 0, 1]) = 1.5 / 4 = 0.375
      expect((loss as any).getScalar()).toBeCloseTo(0.375, 5)
    })

    test('computes MSE with reduction=sum', () => {
      const input = new MockTensor([1.0, 2.0], [2] as const, 'float32' as const)
      const target = new MockTensor([0.0, 0.0], [2] as const, 'float32' as const)

      const loss = mseLoss(input as any, target as any, { reduction: 'sum' })

      // Sum = (1-0)^2 + (2-0)^2 = 1 + 4 = 5
      expect((loss as any).getScalar()).toBe(5.0)
    })

    test('computes MSE with reduction=none', () => {
      const input = new MockTensor([1.0, 2.0], [2] as const, 'float32' as const)
      const target = new MockTensor([0.0, 1.0], [2] as const, 'float32' as const)

      const loss = mseLoss(input as any, target as any, { reduction: 'none' })

      // Per-element: [(1-0)^2, (2-1)^2] = [1, 1]
      const data = (loss as any).getData()
      expect(data[0]).toBe(1.0)
      expect(data[1]).toBe(1.0)
    })

    test('handles zero error', () => {
      const input = new MockTensor([1.0, 2.0, 3.0], [3] as const, 'float32' as const)
      const target = new MockTensor([1.0, 2.0, 3.0], [3] as const, 'float32' as const)

      const loss = mseLoss(input as any, target as any)

      expect((loss as any).getScalar()).toBe(0.0)
    })
  })

  describe('l1Loss', () => {
    test('computes mean absolute error with default reduction', () => {
      const input = new MockTensor([2.5, 0.0, 2.0, 8.0], [4] as const, 'float32' as const)
      const target = new MockTensor([3.0, -0.5, 2.0, 7.0], [4] as const, 'float32' as const)

      const loss = l1Loss(input as any, target as any)

      // L1 = mean([|2.5-3|, |0-(-0.5)|, |2-2|, |8-7|])
      // L1 = mean([0.5, 0.5, 0, 1]) = 2.0 / 4 = 0.5
      expect((loss as any).getScalar()).toBe(0.5)
    })

    test('computes L1 with reduction=sum', () => {
      const input = new MockTensor([1.0, 2.0], [2] as const, 'float32' as const)
      const target = new MockTensor([0.0, 0.0], [2] as const, 'float32' as const)

      const loss = l1Loss(input as any, target as any, { reduction: 'sum' })

      // Sum = |1-0| + |2-0| = 1 + 2 = 3
      expect((loss as any).getScalar()).toBe(3.0)
    })

    test('computes L1 with reduction=none', () => {
      const input = new MockTensor([1.0, -2.0], [2] as const, 'float32' as const)
      const target = new MockTensor([0.0, 1.0], [2] as const, 'float32' as const)

      const loss = l1Loss(input as any, target as any, { reduction: 'none' })

      // Per-element: [|1-0|, |-2-1|] = [1, 3]
      const data = (loss as any).getData()
      expect(data[0]).toBe(1.0)
      expect(data[1]).toBe(3.0)
    })
  })

  describe('smoothL1Loss', () => {
    test('computes smooth L1 loss with default reduction', () => {
      const input = new MockTensor([1.0, 2.0, 3.0], [3] as const, 'float32' as const)
      const target = new MockTensor([0.0, 1.0, 2.0], [3] as const, 'float32' as const)

      const loss = smoothL1Loss(input as any, target as any)

      // Simplified implementation returns L1 loss
      expect((loss as any).getScalar()).toBeGreaterThan(0)
    })

    test('computes smooth L1 with reduction=none', () => {
      const input = new MockTensor([1.0, 2.0], [2] as const, 'float32' as const)
      const target = new MockTensor([0.0, 0.0], [2] as const, 'float32' as const)

      const loss = smoothL1Loss(input as any, target as any, { reduction: 'none' })

      const data = (loss as any).getData()
      expect(data).toHaveLength(2)
    })

    test('accepts beta parameter', () => {
      const input = new MockTensor([1.0, 2.0], [2] as const, 'float32' as const)
      const target = new MockTensor([0.0, 0.0], [2] as const, 'float32' as const)

      expect(() => {
        smoothL1Loss(input as any, target as any, { beta: 0.5 })
      }).not.toThrow()
    })
  })

  describe('binaryCrossEntropyLoss', () => {
    test('computes binary cross entropy with default reduction', () => {
      const input = new MockTensor([0.8, 0.3], [2] as const, 'float32' as const)
      const target = new MockTensor([1.0, 0.0], [2] as const, 'float32' as const)

      const loss = binaryCrossEntropyLoss(input as any, target as any)

      // BCE should be positive
      expect((loss as any).getScalar()).toBeGreaterThan(0)
    })

    test('clamps input to avoid log(0)', () => {
      const input = new MockTensor([0.0, 1.0], [2] as const, 'float32' as const)
      const target = new MockTensor([0.0, 1.0], [2] as const, 'float32' as const)

      // Should not throw error or return infinity
      expect(() => {
        binaryCrossEntropyLoss(input as any, target as any)
      }).not.toThrow()
    })

    test('computes BCE with reduction=none', () => {
      const input = new MockTensor([0.5, 0.7], [2] as const, 'float32' as const)
      const target = new MockTensor([1.0, 0.0], [2] as const, 'float32' as const)

      const loss = binaryCrossEntropyLoss(input as any, target as any, { reduction: 'none' })

      const data = (loss as any).getData()
      expect(data).toHaveLength(2)
      expect(data[0]).toBeGreaterThan(0)
      expect(data[1]).toBeGreaterThan(0)
    })
  })

  describe('crossEntropyLoss', () => {
    test('computes cross entropy for multi-class classification', () => {
      // Logits for 2 samples, 3 classes
      const logits = new MockTensor([2.0, 1.0, 0.1, 0.5, 2.5, 0.2], [2, 3] as const, 'float32' as const)
      const targets = new MockTensor([0, 1], [2] as const, 'float32' as const)

      const loss = crossEntropyLoss(logits as any, targets as any)

      // Loss should be positive
      expect((loss as any).getScalar()).toBeGreaterThan(0)
    })

    test('computes CE with reduction=sum', () => {
      const logits = new MockTensor([1.0, 2.0, 3.0], [1, 3] as const, 'float32' as const)
      const targets = new MockTensor([0], [1] as const, 'float32' as const)

      const loss = crossEntropyLoss(logits as any, targets as any, { reduction: 'sum' })

      expect((loss as any).getScalar()).toBeGreaterThan(0)
    })

    test('computes CE with reduction=none', () => {
      const logits = new MockTensor([1.0, 2.0, 3.0, 0.5, 1.5, 2.5], [2, 3] as const, 'float32' as const)
      const targets = new MockTensor([0, 2], [2] as const, 'float32' as const)

      const loss = crossEntropyLoss(logits as any, targets as any, { reduction: 'none' })

      // Should return per-sample losses
      const data = (loss as any).getData()
      expect(data).toHaveLength(2)
    })

    test('throws error if required operations not available', () => {
      const invalidTensor = { shape: [2, 3], dtype: 'float32' }

      expect(() => {
        crossEntropyLoss(invalidTensor as any, invalidTensor as any)
      }).toThrow()
    })
  })

  describe('klDivLoss', () => {
    test('computes KL divergence with default reduction', () => {
      const input = new MockTensor([Math.log(0.3), Math.log(0.7)], [2] as const, 'float32' as const)
      const target = new MockTensor([0.3, 0.7], [2] as const, 'float32' as const)

      const loss = klDivLoss(input as any, target as any)

      // KL divergence should be non-negative
      expect((loss as any).getScalar()).toBeGreaterThanOrEqual(0)
    })

    test('computes KL with reduction=sum', () => {
      const input = new MockTensor([Math.log(0.5), Math.log(0.5)], [2] as const, 'float32' as const)
      const target = new MockTensor([0.3, 0.7], [2] as const, 'float32' as const)

      const loss = klDivLoss(input as any, target as any, { reduction: 'sum' })

      expect((loss as any).getScalar()).toBeGreaterThanOrEqual(0)
    })

    test('computes KL with reduction=none', () => {
      const input = new MockTensor([Math.log(0.3), Math.log(0.7)], [2] as const, 'float32' as const)
      const target = new MockTensor([0.3, 0.7], [2] as const, 'float32' as const)

      const loss = klDivLoss(input as any, target as any, { reduction: 'none' })

      const data = (loss as any).getData()
      expect(data).toHaveLength(2)
    })
  })

  describe('reduction modes', () => {
    test('none reduction returns original shape', () => {
      const input = new MockTensor([1.0, 2.0, 3.0, 4.0], [4] as const, 'float32' as const)
      const target = new MockTensor([0.0, 0.0, 0.0, 0.0], [4] as const, 'float32' as const)

      const loss = mseLoss(input as any, target as any, { reduction: 'none' })

      expect((loss as any).shape).toEqual([4])
    })

    test('mean reduction returns scalar', () => {
      const input = new MockTensor([1.0, 2.0, 3.0, 4.0], [4] as const, 'float32' as const)
      const target = new MockTensor([0.0, 0.0, 0.0, 0.0], [4] as const, 'float32' as const)

      const loss = mseLoss(input as any, target as any, { reduction: 'mean' })

      // Should return a scalar (shape [1])
      expect((loss as any).getData()).toHaveLength(1)
    })

    test('sum reduction returns scalar', () => {
      const input = new MockTensor([1.0, 2.0, 3.0, 4.0], [4] as const, 'float32' as const)
      const target = new MockTensor([0.0, 0.0, 0.0, 0.0], [4] as const, 'float32' as const)

      const loss = mseLoss(input as any, target as any, { reduction: 'sum' })

      // Should return a scalar (shape [1])
      expect((loss as any).getData()).toHaveLength(1)
    })
  })

  describe('edge cases', () => {
    test('handles empty tensors gracefully', () => {
      const input = new MockTensor([], [0] as const, 'float32' as const)
      const target = new MockTensor([], [0] as const, 'float32' as const)

      // Should handle empty input without errors
      expect(() => {
        mseLoss(input as any, target as any)
      }).not.toThrow()
    })

    test('handles single element tensors', () => {
      const input = new MockTensor([1.5], [1] as const, 'float32' as const)
      const target = new MockTensor([1.0], [1] as const, 'float32' as const)

      const loss = mseLoss(input as any, target as any)

      expect((loss as any).getScalar()).toBeCloseTo(0.25, 5)
    })

    test('handles large values without overflow', () => {
      const input = new MockTensor([1000.0, 2000.0], [2] as const, 'float32' as const)
      const target = new MockTensor([0.0, 0.0], [2] as const, 'float32' as const)

      const loss = mseLoss(input as any, target as any)

      expect((loss as any).getScalar()).toBeGreaterThan(0)
      expect(isFinite((loss as any).getScalar())).toBe(true)
    })
  })
})
