/**
 * Tests for SGD optimizer
 */

import { describe, test, expect } from 'vitest'
import { SGD } from '../sgd'
import { torch } from '@ts-torch/core'

describe('SGD', () => {
  describe('constructor', () => {
    test('creates optimizer with default options', () => {
      torch.run(() => {
        const params = [torch.tensor([1, 2, 3], [3] as const)]
        const optimizer = new SGD(params, { lr: 0.01 })

        expect(optimizer.learningRate).toBe(0.01)
        expect(optimizer.defaults.momentum).toBe(0)
        expect(optimizer.defaults.weightDecay).toBe(0)
      })
    })

    test('creates optimizer with momentum', () => {
      torch.run(() => {
        const params = [torch.tensor([1, 2, 3], [3] as const)]
        const optimizer = new SGD(params, { lr: 0.01, momentum: 0.9 })

        expect(optimizer.learningRate).toBe(0.01)
        expect(optimizer.defaults.momentum).toBe(0.9)
      })
    })

    test('creates optimizer with weight decay', () => {
      torch.run(() => {
        const params = [torch.tensor([1, 2, 3], [3] as const)]
        const optimizer = new SGD(params, { lr: 0.01, weightDecay: 0.0001 })

        expect(optimizer.learningRate).toBe(0.01)
        expect(optimizer.defaults.weightDecay).toBe(0.0001)
      })
    })

    test('creates optimizer with all options', () => {
      torch.run(() => {
        const params = [torch.tensor([1, 2, 3], [3] as const)]
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
  })

  describe('step', () => {
    test('updates parameters with gradient', () => {
      torch.run(() => {
        const param = torch.tensor([1.0, 2.0, 3.0], [3] as const, torch.float32, true)
        // Simulate gradient by doing a backward pass
        const loss = param.sum()
        loss.backward()

        const initialData = Array.from(param.toArray() as Float32Array)
        const optimizer = new SGD([param], { lr: 0.1 })
        optimizer.step()

        const data = Array.from(param.toArray() as Float32Array)
        // After step, values should have changed
        expect(data[0]).not.toBe(initialData[0])
      })
    })

    test('skips parameters without gradients', () => {
      torch.run(() => {
        const param = torch.tensor([1.0, 2.0, 3.0], [3] as const)
        const initialData = Array.from(param.toArray() as Float32Array)

        const optimizer = new SGD([param], { lr: 0.1 })
        optimizer.step()

        // Parameters should remain unchanged (no grad)
        const data = Array.from(param.toArray() as Float32Array)
        expect(data).toEqual(initialData)
      })
    })

    test('applies weight decay', () => {
      torch.run(() => {
        const param = torch.tensor([1.0, 2.0, 3.0], [3] as const, torch.float32, true)
        const loss = param.sum()
        loss.backward()

        const initialData = Array.from(param.toArray() as Float32Array)
        const optimizer = new SGD([param], { lr: 0.1, weightDecay: 0.01 })
        optimizer.step()

        // With weight decay, parameters should still be updated
        const data = Array.from(param.toArray() as Float32Array)
        expect(data[0]).not.toBe(initialData[0])
      })
    })

    test('handles multiple parameters', () => {
      torch.run(() => {
        const param1 = torch.tensor([1.0, 2.0], [2] as const, torch.float32, true)
        const param2 = torch.tensor([3.0, 4.0], [2] as const, torch.float32, true)

        const initial1 = Array.from(param1.toArray() as Float32Array)
        const initial2 = Array.from(param2.toArray() as Float32Array)

        const loss = param1.sum().add(param2.sum())
        loss.backward()

        const optimizer = new SGD([param1, param2], { lr: 0.1 })
        optimizer.step()

        const data1 = Array.from(param1.toArray() as Float32Array)
        const data2 = Array.from(param2.toArray() as Float32Array)

        // Both parameters should have been updated
        expect(data1[0]).not.toBe(initial1[0])
        expect(data2[0]).not.toBe(initial2[0])
      })
    })
  })

  describe('zeroGrad', () => {
    test('zeros all parameter gradients', () => {
      torch.run(() => {
        const param1 = torch.tensor([1.0, 2.0], [2] as const, torch.float32, true)
        const param2 = torch.tensor([3.0, 4.0], [2] as const, torch.float32, true)

        const loss = param1.sum().add(param2.sum())
        loss.backward()

        const optimizer = new SGD([param1, param2], { lr: 0.1 })
        optimizer.zeroGrad()

        // After zeroGrad, gradients should be zeroed (or null)
        // The underlying implementation may keep the gradient tensor with zero values
        // or may set it to null depending on the native library
        if (param1.grad !== null) {
          const grad1 = Array.from(param1.grad.toArray() as Float32Array)
          expect(grad1.every((v) => v === 0)).toBe(true)
        }
        if (param2.grad !== null) {
          const grad2 = Array.from(param2.grad.toArray() as Float32Array)
          expect(grad2.every((v) => v === 0)).toBe(true)
        }
      })
    })

    test('handles parameters without gradients', () => {
      torch.run(() => {
        const param = torch.tensor([1.0, 2.0], [2] as const)
        const optimizer = new SGD([param], { lr: 0.1 })

        expect(() => {
          optimizer.zeroGrad()
        }).not.toThrow()
      })
    })
  })

  describe('learningRate getter/setter', () => {
    test('gets learning rate', () => {
      torch.run(() => {
        const params = [torch.tensor([1, 2, 3], [3] as const)]
        const optimizer = new SGD(params, { lr: 0.01 })

        expect(optimizer.learningRate).toBe(0.01)
      })
    })

    test('sets learning rate', () => {
      torch.run(() => {
        const params = [torch.tensor([1, 2, 3], [3] as const)]
        const optimizer = new SGD(params, { lr: 0.01 })

        optimizer.learningRate = 0.001
        expect(optimizer.learningRate).toBe(0.001)
      })
    })

    test('affects parameter updates', () => {
      torch.run(() => {
        const param = torch.tensor([1.0, 2.0, 3.0], [3] as const, torch.float32, true)
        const loss = param.sum()
        loss.backward()

        const initialData = Array.from(param.toArray() as Float32Array)
        const optimizer = new SGD([param], { lr: 0.1 })
        optimizer.learningRate = 0.01
        optimizer.step()

        // With smaller lr, changes should be smaller
        const data = Array.from(param.toArray() as Float32Array)
        expect(data[0]).not.toBe(initialData[0])
      })
    })
  })

  describe('toString', () => {
    test('returns string representation', () => {
      torch.run(() => {
        const params = [torch.tensor([1, 2, 3], [3] as const)]
        const optimizer = new SGD(params, { lr: 0.01, momentum: 0.9, weightDecay: 0.0001 })

        const str = optimizer.toString()
        expect(str).toContain('SGD')
        expect(str).toContain('lr=0.01')
        expect(str).toContain('momentum=0.9')
        expect(str).toContain('weight_decay=0.0001')
      })
    })

    test('shows default values when not specified', () => {
      torch.run(() => {
        const params = [torch.tensor([1, 2, 3], [3] as const)]
        const optimizer = new SGD(params, { lr: 0.01 })

        const str = optimizer.toString()
        expect(str).toContain('momentum=0')
        expect(str).toContain('weight_decay=0')
      })
    })
  })
})
