/**
 * Tests for SGD optimizer
 */

import { describe, test, expect } from 'vitest'
import { SGD } from '../sgd'
import type { Tensor } from '@ts-torch/core'

/**
 * Mock tensor with tensor operations for testing
 */
class MockTensor implements Partial<Tensor> {
  shape: readonly number[]
  dtype: string
  grad: MockTensor | null = null
  _handle: number
  _data: number[]

  constructor(data: number[], shape: readonly number[] = [data.length], dtype: string = 'float32') {
    this.shape = shape
    this.dtype = dtype
    this._handle = Math.random()
    this._data = [...data]
  }

  clone(): MockTensor {
    return new MockTensor([...this._data], this.shape, this.dtype)
  }

  add(other: MockTensor): MockTensor {
    const result = new MockTensor(
      this._data.map((v, i) => v + (other._data[i] ?? 0)),
      this.shape,
      this.dtype,
    )
    return result
  }

  sub(other: MockTensor): MockTensor {
    const result = new MockTensor(
      this._data.map((v, i) => v - (other._data[i] ?? 0)),
      this.shape,
      this.dtype,
    )
    return result
  }

  mulScalar(scalar: number): MockTensor {
    const result = new MockTensor(
      this._data.map((v) => v * scalar),
      this.shape,
      this.dtype,
    )
    return result
  }

  mul(other: number | MockTensor): MockTensor {
    if (typeof other === 'number') {
      return this.mulScalar(other)
    }
    const result = new MockTensor(
      this._data.map((v, i) => v * (other._data[i] ?? 0)),
      this.shape,
      this.dtype,
    )
    return result
  }

  zeroGrad(): void {
    this.grad = null
  }

  getData(): number[] {
    return [...this._data]
  }
}

describe('SGD', () => {
  describe('constructor', () => {
    test('creates optimizer with default options', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new SGD(params, { lr: 0.01 })

      expect(optimizer.learningRate).toBe(0.01)
      expect(optimizer.defaults.momentum).toBe(0)
      expect(optimizer.defaults.weightDecay).toBe(0)
    })

    test('creates optimizer with momentum', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new SGD(params, { lr: 0.01, momentum: 0.9 })

      expect(optimizer.learningRate).toBe(0.01)
      expect(optimizer.defaults.momentum).toBe(0.9)
    })

    test('creates optimizer with weight decay', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new SGD(params, { lr: 0.01, weightDecay: 0.0001 })

      expect(optimizer.learningRate).toBe(0.01)
      expect(optimizer.defaults.weightDecay).toBe(0.0001)
    })

    test('creates optimizer with all options', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new SGD(params, {
        lr: 0.1,
        momentum: 0.95,
        weightDecay: 0.001,
      })

      expect(optimizer.learningRate).toBe(0.1)
      expect(optimizer.defaults.momentum).toBe(0.95)
      expect(optimizer.defaults.weightDecay).toBe(0.001)
    })
  })

  describe('step', () => {
    test('updates parameters with gradient', () => {
      const param = new MockTensor([1.0, 2.0, 3.0])
      param.grad = new MockTensor([0.1, 0.2, 0.3])

      const optimizer = new SGD([param as unknown as Tensor], { lr: 0.1 })
      optimizer.step()

      // Expected: param = param - lr * grad
      // [1.0, 2.0, 3.0] - 0.1 * [0.1, 0.2, 0.3] = [0.99, 1.98, 2.97]
      const data = param.getData()
      expect(data[0]).toBeCloseTo(0.99, 5)
      expect(data[1]).toBeCloseTo(1.98, 5)
      expect(data[2]).toBeCloseTo(2.97, 5)
    })

    test('skips parameters without gradients', () => {
      const param = new MockTensor([1.0, 2.0, 3.0])
      const initialData = param.getData()

      const optimizer = new SGD([param as unknown as Tensor], { lr: 0.1 })
      optimizer.step()

      // Parameters should remain unchanged
      const data = param.getData()
      expect(data).toEqual(initialData)
    })

    test('applies weight decay (L2 regularization)', () => {
      const param = new MockTensor([1.0, 2.0, 3.0])
      param.grad = new MockTensor([0.1, 0.2, 0.3])

      const optimizer = new SGD([param as unknown as Tensor], { lr: 0.1, weightDecay: 0.01 })
      optimizer.step()

      // With weight decay: grad' = grad + weight_decay * param
      // grad' = [0.1, 0.2, 0.3] + 0.01 * [1.0, 2.0, 3.0] = [0.11, 0.22, 0.33]
      // param = param - lr * grad' = [1.0, 2.0, 3.0] - 0.1 * [0.11, 0.22, 0.33]
      const data = param.getData()
      expect(data[0]).toBeCloseTo(0.989, 5)
      expect(data[1]).toBeCloseTo(1.978, 5)
      expect(data[2]).toBeCloseTo(2.967, 5)
    })

    test('applies momentum', () => {
      const param = new MockTensor([1.0, 2.0, 3.0])
      param.grad = new MockTensor([0.1, 0.2, 0.3])

      const optimizer = new SGD([param as unknown as Tensor], { lr: 0.1, momentum: 0.9 })

      // First step: velocity = gradient
      optimizer.step()
      const data1 = param.getData()

      // Second step with new gradient
      param.grad = new MockTensor([0.05, 0.1, 0.15])
      optimizer.step()
      const data2 = param.getData()

      // Velocity should accumulate across steps
      expect(data2[0]).toBeLessThan(data1[0])
      expect(data2[1]).toBeLessThan(data1[1])
      expect(data2[2]).toBeLessThan(data1[2])
    })

    test('handles multiple parameters', () => {
      const param1 = new MockTensor([1.0, 2.0])
      const param2 = new MockTensor([3.0, 4.0])
      param1.grad = new MockTensor([0.1, 0.2])
      param2.grad = new MockTensor([0.3, 0.4])

      const optimizer = new SGD([param1 as unknown as Tensor, param2 as unknown as Tensor], { lr: 0.1 })
      optimizer.step()

      const data1 = param1.getData()
      const data2 = param2.getData()

      expect(data1[0]).toBeCloseTo(0.99, 5)
      expect(data1[1]).toBeCloseTo(1.98, 5)
      expect(data2[0]).toBeCloseTo(2.97, 5)
      expect(data2[1]).toBeCloseTo(3.96, 5)
    })
  })

  describe('zeroGrad', () => {
    test('zeros all parameter gradients', () => {
      const param1 = new MockTensor([1.0, 2.0])
      const param2 = new MockTensor([3.0, 4.0])
      param1.grad = new MockTensor([0.1, 0.2])
      param2.grad = new MockTensor([0.3, 0.4])

      const optimizer = new SGD([param1 as unknown as Tensor, param2 as unknown as Tensor], { lr: 0.1 })
      optimizer.zeroGrad()

      expect(param1.grad).toBeNull()
      expect(param2.grad).toBeNull()
    })

    test('handles parameters without gradients', () => {
      const param = new MockTensor([1.0, 2.0])
      const optimizer = new SGD([param as unknown as Tensor], { lr: 0.1 })

      expect(() => {
        optimizer.zeroGrad()
      }).not.toThrow()
    })
  })

  describe('learningRate getter/setter', () => {
    test('gets learning rate', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new SGD(params, { lr: 0.01 })

      expect(optimizer.learningRate).toBe(0.01)
    })

    test('sets learning rate', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new SGD(params, { lr: 0.01 })

      optimizer.learningRate = 0.001
      expect(optimizer.learningRate).toBe(0.001)
    })

    test('affects parameter updates', () => {
      const param = new MockTensor([1.0, 2.0, 3.0])
      param.grad = new MockTensor([0.1, 0.2, 0.3])

      const optimizer = new SGD([param as unknown as Tensor], { lr: 0.1 })
      optimizer.learningRate = 0.01
      optimizer.step()

      // With lr=0.01: param = [1.0, 2.0, 3.0] - 0.01 * [0.1, 0.2, 0.3]
      const data = param.getData()
      expect(data[0]).toBeCloseTo(0.999, 5)
      expect(data[1]).toBeCloseTo(1.998, 5)
      expect(data[2]).toBeCloseTo(2.997, 5)
    })
  })

  describe('toString', () => {
    test('returns string representation', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new SGD(params, { lr: 0.01, momentum: 0.9, weightDecay: 0.0001 })

      const str = optimizer.toString()
      expect(str).toContain('SGD')
      expect(str).toContain('lr=0.01')
      expect(str).toContain('momentum=0.9')
      expect(str).toContain('weight_decay=0.0001')
    })

    test('shows default values when not specified', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new SGD(params, { lr: 0.01 })

      const str = optimizer.toString()
      expect(str).toContain('momentum=0')
      expect(str).toContain('weight_decay=0')
    })
  })

  describe('integration scenarios', () => {
    test('complete training step workflow', () => {
      const param = new MockTensor([1.0, 2.0, 3.0, 4.0])
      const optimizer = new SGD([param as unknown as Tensor], { lr: 0.1, momentum: 0.9 })

      // Simulate multiple training steps
      for (let i = 0; i < 3; i++) {
        // Simulate gradient computation
        param.grad = new MockTensor([0.1, 0.1, 0.1, 0.1])

        // Optimization step
        optimizer.step()

        // Zero gradients for next iteration
        optimizer.zeroGrad()
        expect(param.grad).toBeNull()
      }

      // Parameters should have been updated
      const data = param.getData()
      expect(data[0]).toBeLessThan(1.0)
      expect(data[1]).toBeLessThan(2.0)
      expect(data[2]).toBeLessThan(3.0)
      expect(data[3]).toBeLessThan(4.0)
    })

    test('momentum accumulates velocity correctly', () => {
      const param = new MockTensor([10.0])
      const optimizer = new SGD([param as unknown as Tensor], { lr: 1.0, momentum: 0.5 })

      // First step: v = g = 1.0, param = 10 - 1*1 = 9
      param.grad = new MockTensor([1.0])
      optimizer.step()
      expect(param.getData()[0]).toBeCloseTo(9.0, 5)

      // Second step: v = 0.5*1 + 1 = 1.5, param = 9 - 1*1.5 = 7.5
      param.grad = new MockTensor([1.0])
      optimizer.step()
      expect(param.getData()[0]).toBeCloseTo(7.5, 5)

      // Third step: v = 0.5*1.5 + 1 = 1.75, param = 7.5 - 1*1.75 = 5.75
      param.grad = new MockTensor([1.0])
      optimizer.step()
      expect(param.getData()[0]).toBeCloseTo(5.75, 5)
    })
  })
})
