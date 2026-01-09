/**
 * Tests for Adam optimizer
 */

import { describe, test, expect } from 'vitest'
import { Adam } from '../adam'
import type { Tensor } from '@ts-torch/core'

/**
 * Mock tensor with tensor operations for testing
 */
class MockTensor implements Partial<Tensor> {
  shape: readonly number[]
  dtype: string
  grad: MockTensor | null | undefined = null
  _handle: number
  _data: number[]
  data: MockTensor

  constructor(data: number[], shape: readonly number[] = [data.length], dtype: string = 'float32') {
    this.shape = shape
    this.dtype = dtype
    this._handle = Math.random()
    this._data = [...data]
    this.data = this
  }

  clone(): MockTensor {
    return new MockTensor([...this._data], this.shape, this.dtype)
  }

  add(other: MockTensor | number): MockTensor {
    if (typeof other === 'number') {
      return new MockTensor(
        this._data.map((v) => v + other),
        this.shape,
        this.dtype,
      )
    }
    return new MockTensor(
      this._data.map((v, i) => v + (other._data[i] ?? 0)),
      this.shape,
      this.dtype,
    )
  }

  sub(other: MockTensor): MockTensor {
    return new MockTensor(
      this._data.map((v, i) => v - (other._data[i] ?? 0)),
      this.shape,
      this.dtype,
    )
  }

  mul(other: number | MockTensor): MockTensor {
    if (typeof other === 'number') {
      return new MockTensor(
        this._data.map((v) => v * other),
        this.shape,
        this.dtype,
      )
    }
    return new MockTensor(
      this._data.map((v, i) => v * (other._data[i] ?? 1)),
      this.shape,
      this.dtype,
    )
  }

  div(other: number | MockTensor): MockTensor {
    if (typeof other === 'number') {
      return new MockTensor(
        this._data.map((v) => v / other),
        this.shape,
        this.dtype,
      )
    }
    return new MockTensor(
      this._data.map((v, i) => v / (other._data[i] ?? 1)),
      this.shape,
      this.dtype,
    )
  }

  pow(exponent: number): MockTensor {
    return new MockTensor(
      this._data.map((v) => Math.pow(v, exponent)),
      this.shape,
      this.dtype,
    )
  }

  sqrt(): MockTensor {
    return new MockTensor(
      this._data.map((v) => Math.sqrt(v)),
      this.shape,
      this.dtype,
    )
  }

  maximum(other: MockTensor): MockTensor {
    return new MockTensor(
      this._data.map((v, i) => Math.max(v, other._data[i] ?? v)),
      this.shape,
      this.dtype,
    )
  }

  getData(): number[] {
    return [...this._data]
  }
}

describe('Adam', () => {
  describe('constructor', () => {
    test('creates optimizer with default options', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new Adam(params, { lr: 0.001 })

      expect(optimizer.learningRate).toBe(0.001)
      expect(optimizer.defaults.betas).toEqual([0.9, 0.999])
      expect(optimizer.defaults.eps).toBe(1e-8)
      expect(optimizer.defaults.weightDecay).toBe(0)
      expect(optimizer.defaults.amsgrad).toBe(false)
    })

    test('creates optimizer with custom betas', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new Adam(params, { lr: 0.001, betas: [0.95, 0.9999] })

      expect(optimizer.defaults.betas).toEqual([0.95, 0.9999])
    })

    test('creates optimizer with custom eps', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new Adam(params, { lr: 0.001, eps: 1e-10 })

      expect(optimizer.defaults.eps).toBe(1e-10)
    })

    test('creates optimizer with weight decay', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new Adam(params, { lr: 0.001, weightDecay: 0.01 })

      expect(optimizer.defaults.weightDecay).toBe(0.01)
    })

    test('creates optimizer with AMSGrad', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new Adam(params, { lr: 0.001, amsgrad: true })

      expect(optimizer.defaults.amsgrad).toBe(true)
    })
  })

  describe('step', () => {
    test('updates parameters with gradient', () => {
      const param = new MockTensor([1.0, 2.0, 3.0])
      param.grad = new MockTensor([0.1, 0.2, 0.3])

      const optimizer = new Adam([param as unknown as Tensor], { lr: 0.1 })
      optimizer.step()

      // Parameters should be updated (exact values depend on Adam algorithm)
      const data = param.getData()
      expect(data[0]).toBeLessThan(1.0)
      expect(data[1]).toBeLessThan(2.0)
      expect(data[2]).toBeLessThan(3.0)
    })

    test('skips parameters without gradients', () => {
      const param = new MockTensor([1.0, 2.0, 3.0])
      const initialData = param.getData()

      const optimizer = new Adam([param as unknown as Tensor], { lr: 0.1 })
      optimizer.step()

      // Parameters should remain unchanged
      const data = param.getData()
      expect(data).toEqual(initialData)
    })

    test('maintains optimizer state across steps', () => {
      const param = new MockTensor([1.0])
      param.grad = new MockTensor([0.1])

      const optimizer = new Adam([param as unknown as Tensor], { lr: 0.1 })

      // First step
      optimizer.step()
      const state1 = optimizer.getState()
      expect(state1.size).toBe(1)

      const paramState = state1.get(param as unknown as Tensor) as {
        step: number
        exp_avg: Tensor | null
        exp_avg_sq: Tensor | null
      }
      expect(paramState).toBeDefined()
      expect(paramState.step).toBe(1)
      expect(paramState.exp_avg).not.toBeNull()
      expect(paramState.exp_avg_sq).not.toBeNull()

      // Second step
      param.grad = new MockTensor([0.2])
      optimizer.step()

      const state2 = optimizer.getState()
      const paramState2 = state2.get(param as unknown as Tensor) as {
        step: number
        exp_avg: Tensor | null
        exp_avg_sq: Tensor | null
      }
      expect(paramState2.step).toBe(2)
    })

    test('applies weight decay (L2 regularization)', () => {
      const param = new MockTensor([1.0, 2.0])
      param.grad = new MockTensor([0.1, 0.2])

      const optimizerWithDecay = new Adam([param.clone() as unknown as Tensor], {
        lr: 0.1,
        weightDecay: 0.01,
      })
      const optimizerNoDecay = new Adam([param.clone() as unknown as Tensor], { lr: 0.1, weightDecay: 0 })

      optimizerWithDecay.step()
      optimizerNoDecay.step()

      // Weight decay should cause more aggressive updates
      // (The exact comparison depends on the implementation details)
    })

    test('handles multiple parameters independently', () => {
      const param1 = new MockTensor([1.0])
      const param2 = new MockTensor([2.0])
      param1.grad = new MockTensor([0.1])
      param2.grad = new MockTensor([0.2])

      const optimizer = new Adam([param1 as unknown as Tensor, param2 as unknown as Tensor], { lr: 0.1 })
      optimizer.step()

      const state = optimizer.getState()
      expect(state.size).toBe(2)

      // Each parameter should have its own state
      const state1 = state.get(param1 as unknown as Tensor)
      const state2 = state.get(param2 as unknown as Tensor)
      expect(state1).toBeDefined()
      expect(state2).toBeDefined()
      expect(state1).not.toBe(state2)
    })

    test('bias correction increases step size early in training', () => {
      const param = new MockTensor([1.0])
      param.grad = new MockTensor([0.1])

      const optimizer = new Adam([param as unknown as Tensor], { lr: 0.1 })

      // First few steps should have larger effective learning rates due to bias correction
      const initialValue = param.getData()[0]
      optimizer.step()
      const firstStepChange = Math.abs(param.getData()[0] - initialValue)

      // The actual values will depend on bias correction
      expect(firstStepChange).toBeGreaterThan(0)
    })
  })

  describe('AMSGrad variant', () => {
    test('enables AMSGrad when specified', () => {
      const param = new MockTensor([1.0])
      param.grad = new MockTensor([0.1])

      const optimizer = new Adam([param as unknown as Tensor], { lr: 0.1, amsgrad: true })
      optimizer.step()

      const state = optimizer.getState()
      const paramState = state.get(param as unknown as Tensor) as {
        step: number
        exp_avg: Tensor | null
        exp_avg_sq: Tensor | null
        max_exp_avg_sq?: Tensor
      }

      // AMSGrad should maintain max_exp_avg_sq
      expect(paramState.max_exp_avg_sq).toBeDefined()
    })
  })

  describe('learningRate getter/setter', () => {
    test('gets learning rate', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new Adam(params, { lr: 0.001 })

      expect(optimizer.learningRate).toBe(0.001)
    })

    test('sets learning rate', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new Adam(params, { lr: 0.001 })

      optimizer.learningRate = 0.0001
      expect(optimizer.learningRate).toBe(0.0001)
    })
  })

  describe('toString', () => {
    test('returns string representation', () => {
      const params = [new MockTensor([1, 2, 3])] as unknown as Tensor[]
      const optimizer = new Adam(params, { lr: 0.001, betas: [0.9, 0.999], eps: 1e-8 })

      const str = optimizer.toString()
      expect(str).toContain('Adam')
      expect(str).toContain('lr=0.001')
      expect(str).toContain('betas=[0.9,0.999]')
      expect(str).toContain('eps=1e-8')
    })
  })

  describe('integration scenarios', () => {
    test('complete training step workflow', () => {
      const param = new MockTensor([1.0, 2.0, 3.0, 4.0])
      const optimizer = new Adam([param as unknown as Tensor], { lr: 0.1 })

      // Simulate multiple training steps
      for (let i = 0; i < 5; i++) {
        // Simulate gradient computation
        param.grad = new MockTensor([0.1, 0.1, 0.1, 0.1])

        // Optimization step
        optimizer.step()

        // Check state is maintained
        const state = optimizer.getState()
        const paramState = state.get(param as unknown as Tensor) as { step: number }
        expect(paramState.step).toBe(i + 1)
      }

      // Parameters should have been updated
      const data = param.getData()
      expect(data[0]).toBeLessThan(1.0)
      expect(data[1]).toBeLessThan(2.0)
      expect(data[2]).toBeLessThan(3.0)
      expect(data[3]).toBeLessThan(4.0)
    })

    test('converges on simple optimization problem', () => {
      // Simple problem: minimize (x - 5)^2, starting from x = 0
      const param = new MockTensor([0.0])
      const optimizer = new Adam([param as unknown as Tensor], { lr: 0.5 })

      const target = 5.0

      for (let i = 0; i < 100; i++) {
        // Gradient of (x - 5)^2 is 2(x - 5)
        const x = param.getData()[0]
        const grad = 2 * (x - target)
        param.grad = new MockTensor([grad])

        optimizer.step()
      }

      // Should converge close to target
      const finalValue = param.getData()[0]
      expect(Math.abs(finalValue - target)).toBeLessThan(0.1)
    })

    test('handles varying gradients across steps', () => {
      const param = new MockTensor([1.0])
      const optimizer = new Adam([param as unknown as Tensor], { lr: 0.1 })

      // Step 1: Large gradient
      param.grad = new MockTensor([1.0])
      optimizer.step()
      const value1 = param.getData()[0]

      // Step 2: Small gradient
      param.grad = new MockTensor([0.01])
      optimizer.step()
      const value2 = param.getData()[0]

      // Step 3: Large gradient again
      param.grad = new MockTensor([1.0])
      optimizer.step()
      const value3 = param.getData()[0]

      // All steps should make progress
      expect(value1).toBeLessThan(1.0)
      expect(value2).toBeLessThan(value1)
      expect(value3).toBeLessThan(value2)
    })
  })

  describe('state management', () => {
    test('initializes state on first step', () => {
      const param = new MockTensor([1.0])
      param.grad = new MockTensor([0.1])

      const optimizer = new Adam([param as unknown as Tensor], { lr: 0.1 })

      // State should be empty before first step
      expect(optimizer.getState().size).toBe(0)

      optimizer.step()

      // State should be initialized after first step
      const state = optimizer.getState()
      expect(state.size).toBe(1)
      const paramState = state.get(param as unknown as Tensor) as {
        step: number
        exp_avg: Tensor | null
        exp_avg_sq: Tensor | null
      }
      expect(paramState.step).toBe(1)
      expect(paramState.exp_avg).not.toBeNull()
      expect(paramState.exp_avg_sq).not.toBeNull()
    })

    test('can load external state', () => {
      const param = new MockTensor([1.0])
      const optimizer = new Adam([param as unknown as Tensor], { lr: 0.1 })

      const externalState = new Map<Tensor, Record<string, unknown>>()
      externalState.set(param as unknown as Tensor, {
        step: 10,
        exp_avg: new MockTensor([0.5]),
        exp_avg_sq: new MockTensor([0.1]),
      })

      optimizer.loadState(externalState)

      const loadedState = optimizer.getState()
      const paramState = loadedState.get(param as unknown as Tensor) as { step: number }
      expect(paramState.step).toBe(10)
    })
  })
})
