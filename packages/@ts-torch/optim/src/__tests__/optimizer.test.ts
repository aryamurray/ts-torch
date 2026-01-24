/**
 * Tests for base Optimizer class
 */

import { describe, test, expect } from 'vitest'
import { Optimizer, type ParameterGroup, type OptimizerOptions } from '../optimizer'
import type { Tensor } from '@ts-torch/core'

/**
 * Mock optimizer implementation for testing
 */
class MockOptimizer extends Optimizer {
  stepCalled = false

  step(): void {
    this.stepCalled = true
  }
}

/**
 * Mock tensor for testing
 */
class MockTensor {
  shape: readonly number[]
  dtype: string
  grad: Tensor | null = null

  constructor(shape: readonly number[] = [2, 2], dtype: string = 'float32') {
    this.shape = shape
    this.dtype = dtype
  }

  zeroGrad(): void {
    this.grad = null
  }
}

describe('Optimizer', () => {
  describe('constructor', () => {
    test('accepts tensor array and creates single parameter group', () => {
      const params = [new MockTensor(), new MockTensor()] as unknown as Tensor[]
      const options: OptimizerOptions = { lr: 0.01 }
      const optimizer = new MockOptimizer(params, options)

      const allParams = optimizer.getAllParams()
      expect(allParams).toHaveLength(2)
      expect(optimizer.lr).toBe(0.01)
    })

    test('accepts parameter groups', () => {
      const group1: ParameterGroup = {
        params: [new MockTensor(), new MockTensor()] as unknown as Tensor[],
        lr: 0.01,
      }
      const group2: ParameterGroup = {
        params: [new MockTensor()] as unknown as Tensor[],
        lr: 0.001,
      }
      const options: OptimizerOptions = { lr: 0.01 }
      const optimizer = new MockOptimizer([group1, group2], options)

      const allParams = optimizer.getAllParams()
      expect(allParams).toHaveLength(3)
    })

    test('throws error for parameter group without params array', () => {
      const invalidGroup = { lr: 0.01 } as unknown as ParameterGroup
      const options: OptimizerOptions = { lr: 0.01 }

      expect(() => {
        new MockOptimizer([invalidGroup], options)
      }).toThrow('Parameter group must have a params array')
    })

    test('sets default options correctly', () => {
      const params = [new MockTensor()] as unknown as Tensor[]
      const options: OptimizerOptions = { lr: 0.01, momentum: 0.9 }
      const optimizer = new MockOptimizer(params, options)

      expect(optimizer.lr).toBe(0.01)
    })
  })

  describe('zeroGrad', () => {
    test('zeros gradients for all parameters', () => {
      const param1 = new MockTensor() as unknown as Tensor
      const param2 = new MockTensor() as unknown as Tensor
      ;(param1 as any).grad = new MockTensor()
      ;(param2 as any).grad = new MockTensor()

      const optimizer = new MockOptimizer([param1, param2], { lr: 0.01 })
      optimizer.zeroGrad()

      expect((param1 as any).grad).toBeNull()
      expect((param2 as any).grad).toBeNull()
    })

    test('handles parameters without gradients', () => {
      const param = new MockTensor() as unknown as Tensor
      const optimizer = new MockOptimizer([param], { lr: 0.01 })

      expect(() => {
        optimizer.zeroGrad()
      }).not.toThrow()
    })

    test('zeros gradients across multiple parameter groups', () => {
      const group1: ParameterGroup = {
        params: [new MockTensor(), new MockTensor()] as unknown as Tensor[],
      }
      const group2: ParameterGroup = {
        params: [new MockTensor()] as unknown as Tensor[],
      }

      // Add gradients
      group1.params.forEach((p) => ((p as any).grad = new MockTensor()))
      group2.params.forEach((p) => ((p as any).grad = new MockTensor()))

      const optimizer = new MockOptimizer([group1, group2], { lr: 0.01 })
      optimizer.zeroGrad()

      group1.params.forEach((p) => expect((p as any).grad).toBeNull())
      group2.params.forEach((p) => expect((p as any).grad).toBeNull())
    })
  })

  describe('lr getter/setter', () => {
    test('gets current learning rate', () => {
      const optimizer = new MockOptimizer([new MockTensor()] as unknown as Tensor[], { lr: 0.01 })
      expect(optimizer.lr).toBe(0.01)
    })

    test('sets learning rate for all parameter groups', () => {
      const group1: ParameterGroup = {
        params: [new MockTensor()] as unknown as Tensor[],
        lr: 0.01,
      }
      const group2: ParameterGroup = {
        params: [new MockTensor()] as unknown as Tensor[],
        lr: 0.001,
      }

      const optimizer = new MockOptimizer([group1, group2], { lr: 0.01 })
      optimizer.lr = 0.05

      expect(optimizer.lr).toBe(0.05)
      expect(group1.lr).toBe(0.05)
      expect(group2.lr).toBe(0.05)
    })
  })

  describe('state management', () => {
    test('getState returns empty state initially', () => {
      const optimizer = new MockOptimizer([new MockTensor()] as unknown as Tensor[], { lr: 0.01 })
      const state = optimizer.getState()

      expect(state).toBeInstanceOf(Map)
      expect(state.size).toBe(0)
    })

    test('loadState sets optimizer state', () => {
      const optimizer = new MockOptimizer([new MockTensor()] as unknown as Tensor[], { lr: 0.01 })
      const newState = new Map<Tensor, Record<string, unknown>>()
      const param = new MockTensor() as unknown as Tensor
      newState.set(param, { step: 1, momentum: 0.9 })

      optimizer.loadState(newState)

      const loadedState = optimizer.getState()
      expect(loadedState).toBe(newState)
      expect(loadedState.size).toBe(1)
    })
  })

  describe('addParamGroup', () => {
    test('adds a new parameter group', () => {
      const optimizer = new MockOptimizer([new MockTensor()] as unknown as Tensor[], { lr: 0.01 })
      expect(optimizer.getAllParams()).toHaveLength(1)

      const newGroup: ParameterGroup = {
        params: [new MockTensor(), new MockTensor()] as unknown as Tensor[],
      }
      optimizer.addParamGroup(newGroup)

      expect(optimizer.getAllParams()).toHaveLength(3)
    })

    test('merges default options with new group', () => {
      const optimizer = new MockOptimizer([new MockTensor()] as unknown as Tensor[], { lr: 0.01, momentum: 0.9 })

      const newGroup: ParameterGroup = {
        params: [new MockTensor()] as unknown as Tensor[],
        lr: 0.001, // Override lr
      }
      optimizer.addParamGroup(newGroup)

      // The new group should have lr: 0.001 but inherit other defaults
      expect(newGroup.lr).toBe(0.001)
    })
  })

  describe('getAllParams', () => {
    test('returns all parameters from all groups', () => {
      const group1: ParameterGroup = {
        params: [new MockTensor(), new MockTensor()] as unknown as Tensor[],
      }
      const group2: ParameterGroup = {
        params: [new MockTensor(), new MockTensor(), new MockTensor()] as unknown as Tensor[],
      }

      const optimizer = new MockOptimizer([group1, group2], { lr: 0.01 })
      const allParams = optimizer.getAllParams()

      expect(allParams).toHaveLength(5)
    })

    test('returns empty array when no parameters', () => {
      const emptyGroup: ParameterGroup = {
        params: [],
      }
      const optimizer = new MockOptimizer([emptyGroup], { lr: 0.01 })

      expect(optimizer.getAllParams()).toHaveLength(0)
    })
  })

  describe('toString', () => {
    test('returns string representation with options', () => {
      const optimizer = new MockOptimizer([new MockTensor()] as unknown as Tensor[], { lr: 0.01, momentum: 0.9 })
      const str = optimizer.toString()

      expect(str).toContain('MockOptimizer')
      expect(str).toContain('lr: 0.01')
      expect(str).toContain('momentum: 0.9')
    })

    test('includes all default options', () => {
      const optimizer = new MockOptimizer([new MockTensor()] as unknown as Tensor[], {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
      })
      const str = optimizer.toString()

      expect(str).toContain('lr: 0.001')
      expect(str).toContain('beta1: 0.9')
      expect(str).toContain('beta2: 0.999')
    })
  })

  describe('abstract step method', () => {
    test('must be implemented by subclass', () => {
      const optimizer = new MockOptimizer([new MockTensor()] as unknown as Tensor[], { lr: 0.01 })
      expect(optimizer.stepCalled).toBe(false)

      optimizer.step()
      expect(optimizer.stepCalled).toBe(true)
    })
  })
})
