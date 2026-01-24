/**
 * Tests for Declarative Trainer
 *
 * These are unit tests for the Trainer class configuration.
 * Integration tests requiring native bindings are skipped
 * when the native library is not available.
 */

import { describe, test, expect } from 'vitest'
import { Trainer } from '../trainer'
import { Adam, SGD } from '../optimizers'
import type { Module, Parameter } from '@ts-torch/nn'
import type { DeviceType, Tensor } from '@ts-torch/core'

/**
 * Mock tensor for testing
 */
class MockTensor {
  shape: readonly number[]
  dtype = 'float32'
  grad: MockTensor | null = null
  private values: number[]
  device = 'cpu'

  constructor(values: number[] = [0], shape: readonly number[] = [1]) {
    this.values = values
    this.shape = shape
  }

  item(): number {
    return this.values[0]
  }

  toArray(): number[] {
    return this.values
  }

  zeroGrad(): void {
    this.grad = null
  }

  free(): void {}

  escape(): void {}
}

/**
 * Mock parameter for testing
 */
class MockParameter implements Partial<Parameter> {
  data: Tensor
  requiresGrad = true

  constructor() {
    this.data = new MockTensor([0.1, 0.2, 0.3, 0.4], [2, 2]) as unknown as Tensor
  }
}

/**
 * Mock model for testing
 */
class MockModel implements Partial<Module<any, any, any, DeviceType>> {
  private params: MockParameter[]
  training = true

  constructor(numParams = 2) {
    this.params = Array.from({ length: numParams }, () => new MockParameter())
  }

  parameters(): Parameter[] {
    return this.params as unknown as Parameter[]
  }

  forward(_input: any): any {
    return new MockTensor([0.3, 0.7, 0.6, 0.4], [2, 2])
  }

  train(): this {
    this.training = true
    return this
  }

  eval(): this {
    this.training = false
    return this
  }
}

describe('Trainer', () => {
  describe('constructor', () => {
    test('creates trainer with model', () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer(model)

      expect(trainer).toBeDefined()
    })

    test('accepts default config with optimizer', () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer(model, {
        optimizer: Adam({ lr: 0.001 }),
      })

      expect(trainer).toBeDefined()
    })

    test('accepts default config with loss', () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer(model, {
        loss: 'crossEntropy',
      })

      expect(trainer).toBeDefined()
    })

    test('accepts default config with both optimizer and loss', () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer(model, {
        optimizer: Adam({ lr: 0.001 }),
        loss: 'mse',
      })

      expect(trainer).toBeDefined()
    })
  })

  describe('fit options validation', () => {
    test('requires optimizer in fit() or constructor', async () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer(model)

      const dataLoader = {
        async *[Symbol.asyncIterator]() {
          yield { data: new MockTensor(), label: new MockTensor() }
        },
      }

      await expect(
        trainer.fit(dataLoader, {
          epochs: 1,
        }),
      ).rejects.toThrow('Optimizer must be provided')
    })
  })

  describe('optimizer configuration', () => {
    test('Adam config has correct structure', () => {
      const config = Adam({ lr: 0.001 })
      expect(config.name).toContain('Adam')
      expect(config.name).toContain('lr=0.001')
      expect(typeof config.create).toBe('function')
    })

    test('SGD config has correct structure', () => {
      const config = SGD({ lr: 0.01, momentum: 0.9 })
      expect(config.name).toContain('SGD')
      expect(config.name).toContain('lr=0.01')
      expect(config.name).toContain('momentum=0.9')
      expect(typeof config.create).toBe('function')
    })
  })

  describe('loss function types', () => {
    test('supports crossEntropy loss type', () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer(model, { loss: 'crossEntropy' })
      expect(trainer).toBeDefined()
    })

    test('supports mse loss type', () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer(model, { loss: 'mse' })
      expect(trainer).toBeDefined()
    })

    test('supports nll loss type', () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer(model, { loss: 'nll' })
      expect(trainer).toBeDefined()
    })

    test('supports custom loss function', () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const customLoss = (_pred: Tensor, _target: Tensor) => new MockTensor([0.5]) as unknown as Tensor
      const trainer = new Trainer(model, { loss: customLoss })
      expect(trainer).toBeDefined()
    })
  })
})
