/**
 * Tests for Declarative Optimizer Factories
 *
 * These are unit tests for the factory functions.
 * They test that the factories return correct configurations
 * without requiring native bindings.
 */

import { describe, test, expect } from 'vitest'
import { Adam, SGD, AdamW, RMSprop, type OptimizerConfig } from '../optimizers'

describe('Adam', () => {
  test('returns correct OptimizerConfig structure', () => {
    const config = Adam({ lr: 0.001 })

    expect(config.name).toContain('Adam')
    expect(config.name).toContain('lr=0.001')
    expect(typeof config.create).toBe('function')
  })

  test('includes lr in name with different values', () => {
    const config1 = Adam({ lr: 0.001 })
    const config2 = Adam({ lr: 0.0001 })

    expect(config1.name).toContain('lr=0.001')
    expect(config2.name).toContain('lr=0.0001')
  })

  test('accepts all Adam options', () => {
    // Should not throw
    const config = Adam({
      lr: 0.001,
      betas: [0.9, 0.999],
      eps: 1e-8,
      weightDecay: 0.01,
      amsgrad: true,
    })

    expect(config.name).toContain('lr=0.001')
  })
})

describe('SGD', () => {
  test('returns correct OptimizerConfig structure', () => {
    const config = SGD({ lr: 0.01 })

    expect(config.name).toContain('SGD')
    expect(config.name).toContain('lr=0.01')
    expect(typeof config.create).toBe('function')
  })

  test('includes momentum in name', () => {
    const config = SGD({ lr: 0.01, momentum: 0.9 })
    expect(config.name).toContain('momentum=0.9')
  })

  test('uses default momentum of 0', () => {
    const config = SGD({ lr: 0.01 })
    expect(config.name).toContain('momentum=0')
  })

  test('accepts all SGD options', () => {
    const config = SGD({
      lr: 0.01,
      momentum: 0.9,
      weightDecay: 0.0001,
      dampening: 0.1,
      nesterov: true,
    })

    expect(config.name).toContain('lr=0.01')
    expect(config.name).toContain('momentum=0.9')
  })
})

describe('AdamW', () => {
  test('returns correct OptimizerConfig structure', () => {
    const config = AdamW({ lr: 0.001 })

    expect(config.name).toContain('AdamW')
    expect(config.name).toContain('lr=0.001')
    expect(typeof config.create).toBe('function')
  })

  test('includes weightDecay in name', () => {
    const config = AdamW({ lr: 0.001, weightDecay: 0.05 })
    expect(config.name).toContain('weightDecay=0.05')
  })

  test('uses default weightDecay of 0.01', () => {
    const config = AdamW({ lr: 0.001 })
    expect(config.name).toContain('weightDecay=0.01')
  })

  test('accepts all AdamW options', () => {
    const config = AdamW({
      lr: 0.001,
      betas: [0.9, 0.999],
      eps: 1e-8,
      weightDecay: 0.01,
      amsgrad: false,
    })

    expect(config.name).toContain('lr=0.001')
  })
})

describe('RMSprop', () => {
  test('returns correct OptimizerConfig structure', () => {
    const config = RMSprop({ lr: 0.01 })

    expect(config.name).toContain('RMSprop')
    expect(config.name).toContain('lr=0.01')
    expect(typeof config.create).toBe('function')
  })

  test('accepts all RMSprop options', () => {
    const config = RMSprop({
      lr: 0.01,
      alpha: 0.99,
      eps: 1e-8,
      weightDecay: 0.0001,
      momentum: 0.9,
      centered: true,
    })

    expect(config.name).toContain('lr=0.01')
  })
})

describe('OptimizerConfig interface', () => {
  test('all factories return objects conforming to OptimizerConfig', () => {
    const configs: OptimizerConfig[] = [
      Adam({ lr: 0.001 }),
      SGD({ lr: 0.01 }),
      AdamW({ lr: 0.001 }),
      RMSprop({ lr: 0.01 }),
    ]

    for (const config of configs) {
      expect(typeof config.name).toBe('string')
      expect(typeof config.create).toBe('function')
    }
  })
})
