/**
 * Tests for Adam optimizer
 */

import { describe, test, expect } from 'vitest'
import { Adam } from '../adam'
import { device, run, float32, type Tensor } from '@ts-torch/core'

const cpu = device.cpu()

describe('Adam', () => {
  describe('constructor', () => {
    test('creates optimizer with default options', () => {
      run(() => {
        const params = [cpu.tensor([1, 2, 3], [3] as const)]
        const optimizer = new Adam(params, { lr: 0.001 })

        expect(optimizer.learningRate).toBe(0.001)
        expect(optimizer.defaults.betas).toEqual([0.9, 0.999])
        expect(optimizer.defaults.eps).toBe(1e-8)
        expect(optimizer.defaults.weightDecay).toBe(0)
        expect(optimizer.defaults.amsgrad).toBe(false)
      })
    })

    test('creates optimizer with custom betas', () => {
      run(() => {
        const params = [cpu.tensor([1, 2, 3], [3] as const)]
        const optimizer = new Adam(params, { lr: 0.001, betas: [0.95, 0.9999] })

        expect(optimizer.defaults.betas).toEqual([0.95, 0.9999])
      })
    })

    test('creates optimizer with custom eps', () => {
      run(() => {
        const params = [cpu.tensor([1, 2, 3], [3] as const)]
        const optimizer = new Adam(params, { lr: 0.001, eps: 1e-10 })

        expect(optimizer.defaults.eps).toBe(1e-10)
      })
    })

    test('creates optimizer with weight decay', () => {
      run(() => {
        const params = [cpu.tensor([1, 2, 3], [3] as const)]
        const optimizer = new Adam(params, { lr: 0.001, weightDecay: 0.01 })

        expect(optimizer.defaults.weightDecay).toBe(0.01)
      })
    })

    test('creates optimizer with AMSGrad', () => {
      run(() => {
        const params = [cpu.tensor([1, 2, 3], [3] as const)]
        const optimizer = new Adam(params, { lr: 0.001, amsgrad: true })

        expect(optimizer.defaults.amsgrad).toBe(true)
      })
    })
  })

  describe('step', () => {
    test('updates parameters with gradient', () => {
      run(() => {
        const param = cpu.tensor([1.0, 2.0, 3.0], [3] as const, float32, true)
        // Simulate gradient by doing a backward pass
        const loss = param.sum()
        loss.backward()

        const initialData = Array.from(param.toArray() as Float32Array)
        const optimizer = new Adam([param], { lr: 0.1 })
        optimizer.step()

        const data = Array.from(param.toArray() as Float32Array)
        // After step, values should have changed
        expect(data[0]).not.toBe(initialData[0])
      })
    })

    test('skips parameters without gradients', () => {
      run(() => {
        const param = cpu.tensor([1.0, 2.0, 3.0], [3] as const)
        const initialData = Array.from(param.toArray() as Float32Array)

        const optimizer = new Adam([param], { lr: 0.1 })
        optimizer.step()

        // Parameters should remain unchanged (no grad)
        const data = Array.from(param.toArray() as Float32Array)
        expect(data).toEqual(initialData)
      })
    })

    test('maintains optimizer state across steps', () => {
      run(() => {
        const param = cpu.tensor([1.0], [1] as const, float32, true)

        const optimizer = new Adam([param], { lr: 0.1 })

        // First step
        const loss1 = param.sum()
        loss1.backward()
        optimizer.step()

        const state1 = optimizer.getState()
        expect(state1.size).toBe(1)

        const paramState = state1.get(param) as {
          step: number
          exp_avg: Tensor | null
          exp_avg_sq: Tensor | null
        }
        expect(paramState).toBeDefined()
        expect(paramState.step).toBe(1)

        // Second step
        param.zeroGrad()
        const loss2 = param.sum()
        loss2.backward()
        optimizer.step()

        const state2 = optimizer.getState()
        const paramState2 = state2.get(param) as { step: number }
        expect(paramState2.step).toBe(2)
      })
    })

    test('handles multiple parameters independently', () => {
      run(() => {
        const param1 = cpu.tensor([1.0], [1] as const, float32, true)
        const param2 = cpu.tensor([2.0], [1] as const, float32, true)

        const optimizer = new Adam([param1, param2], { lr: 0.1 })

        // Compute gradients
        const loss = param1.sum().add(param2.sum())
        loss.backward()

        optimizer.step()

        const state = optimizer.getState()
        expect(state.size).toBe(2)

        // Each parameter should have its own state
        const state1 = state.get(param1)
        const state2 = state.get(param2)
        expect(state1).toBeDefined()
        expect(state2).toBeDefined()
        expect(state1).not.toBe(state2)
      })
    })
  })

  describe('AMSGrad variant', () => {
    test('enables AMSGrad when specified', () => {
      run(() => {
        const param = cpu.tensor([1.0], [1] as const, float32, true)
        const loss = param.sum()
        loss.backward()

        const optimizer = new Adam([param], { lr: 0.1, amsgrad: true })
        optimizer.step()

        const state = optimizer.getState()
        const paramState = state.get(param) as {
          step: number
          max_exp_avg_sq?: Tensor
        }

        // AMSGrad should maintain max_exp_avg_sq
        expect(paramState.max_exp_avg_sq).toBeDefined()
      })
    })
  })

  describe('learningRate getter/setter', () => {
    test('gets learning rate', () => {
      run(() => {
        const params = [cpu.tensor([1, 2, 3], [3] as const)]
        const optimizer = new Adam(params, { lr: 0.001 })

        expect(optimizer.learningRate).toBe(0.001)
      })
    })

    test('sets learning rate', () => {
      run(() => {
        const params = [cpu.tensor([1, 2, 3], [3] as const)]
        const optimizer = new Adam(params, { lr: 0.001 })

        optimizer.learningRate = 0.0001
        expect(optimizer.learningRate).toBe(0.0001)
      })
    })
  })

  describe('toString', () => {
    test('returns string representation', () => {
      run(() => {
        const params = [cpu.tensor([1, 2, 3], [3] as const)]
        const optimizer = new Adam(params, { lr: 0.001, betas: [0.9, 0.999], eps: 1e-8 })

        const str = optimizer.toString()
        expect(str).toContain('Adam')
        expect(str).toContain('lr=0.001')
        expect(str).toContain('betas=[0.9,0.999]')
        expect(str).toContain('eps=1e-8')
      })
    })
  })

  describe('state management', () => {
    test('initializes state on first step', () => {
      run(() => {
        const param = cpu.tensor([1.0], [1] as const, float32, true)
        const loss = param.sum()
        loss.backward()

        const optimizer = new Adam([param], { lr: 0.1 })

        // State should be empty before first step
        expect(optimizer.getState().size).toBe(0)

        optimizer.step()

        // State should be initialized after first step
        const state = optimizer.getState()
        expect(state.size).toBe(1)
        const paramState = state.get(param) as {
          step: number
          exp_avg: Tensor | null
          exp_avg_sq: Tensor | null
        }
        expect(paramState.step).toBe(1)
      })
    })
  })
})
