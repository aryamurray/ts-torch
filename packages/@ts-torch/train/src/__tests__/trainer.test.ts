/**
 * Tests for Declarative Trainer
 *
 * These are unit tests for the Trainer class configuration.
 * Integration tests requiring native bindings are skipped
 * when the native library is not available.
 */

import { describe, test, expect } from 'vitest'
import { Trainer } from '../trainer'
import { loss } from '../loss'
import { Adam, SGD } from '../optimizers'
import type { Module, Parameter } from '@ts-torch/nn'
import type { DeviceType, Tensor } from '@ts-torch/core'
import type { Batch } from '../trainer'

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

function makeMockDataLoader(): AsyncIterable<Batch> {
  return {
    async *[Symbol.asyncIterator]() {
      yield {
        input: new MockTensor([1, 2, 3, 4], [2, 2]) as unknown as Tensor,
        target: new MockTensor([0, 1], [2]) as unknown as Tensor,
      }
    },
  }
}

describe('Trainer', () => {
  describe('constructor', () => {
    test('creates trainer with unified options', () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer({
        model,
        data: makeMockDataLoader(),
        epochs: 5,
        optimizer: Adam({ lr: 1e-3 }),
        loss: loss.crossEntropy(),
      })

      expect(trainer).toBeDefined()
    })

    test('accepts metrics as array', () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer({
        model,
        data: makeMockDataLoader(),
        epochs: 5,
        optimizer: Adam({ lr: 1e-3 }),
        loss: loss.crossEntropy(),
        metrics: ['loss', 'accuracy'],
      })

      expect(trainer).toBeDefined()
    })

    test('accepts validation data', () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer({
        model,
        data: makeMockDataLoader(),
        epochs: 5,
        optimizer: Adam({ lr: 1e-3 }),
        loss: loss.crossEntropy(),
        validation: makeMockDataLoader(),
      })

      expect(trainer).toBeDefined()
    })

    test('accepts callbacks array', () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer({
        model,
        data: makeMockDataLoader(),
        epochs: 5,
        optimizer: Adam({ lr: 1e-3 }),
        loss: loss.crossEntropy(),
        callbacks: [{ onEpochEnd: () => {} }],
      })

      expect(trainer).toBeDefined()
    })

    test('accepts onEpochEnd shorthand', () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer({
        model,
        data: makeMockDataLoader(),
        epochs: 5,
        optimizer: Adam({ lr: 1e-3 }),
        loss: loss.crossEntropy(),
        onEpochEnd: () => {},
      })

      expect(trainer).toBeDefined()
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

  describe('loss configuration', () => {
    test('loss.crossEntropy() creates correct config', () => {
      const config = loss.crossEntropy()
      expect(config).toEqual({ kind: 'crossEntropy' })
    })

    test('loss.crossEntropy() accepts labelSmoothing', () => {
      const config = loss.crossEntropy({ labelSmoothing: 0.1 })
      expect(config).toEqual({ kind: 'crossEntropy', labelSmoothing: 0.1 })
    })

    test('loss.mse() creates correct config', () => {
      const config = loss.mse()
      expect(config).toEqual({ kind: 'mse' })
    })

    test('loss.nll() creates correct config', () => {
      const config = loss.nll()
      expect(config).toEqual({ kind: 'nll' })
    })

    test('loss.custom() creates correct config', () => {
      const fn = (_pred: Tensor, _target: Tensor) => new MockTensor([0.5]) as unknown as Tensor
      const config = loss.custom('myLoss', fn)
      expect(config.kind).toBe('custom')
      expect((config as any).name).toBe('myLoss')
      expect((config as any).fn).toBe(fn)
    })
  })

  describe('evaluate', () => {
    test('throws if no validation data configured and called zero-arg', async () => {
      const model = new MockModel() as unknown as Module<any, any, any, DeviceType>
      const trainer = new Trainer({
        model,
        data: makeMockDataLoader(),
        epochs: 1,
        optimizer: Adam({ lr: 1e-3 }),
        loss: loss.crossEntropy(),
      })

      await expect(trainer.evaluate()).rejects.toThrow('No validation data configured')
    })
  })
})
