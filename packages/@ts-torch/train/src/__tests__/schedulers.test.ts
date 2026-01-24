/**
 * Tests for Declarative LR Scheduler Factories
 *
 * These are unit tests for the factory functions.
 * They test that the factories return correct configurations
 * without requiring native bindings.
 */

import { describe, test, expect } from 'vitest'
import {
  StepLR,
  MultiStepLR,
  ExponentialLR,
  CosineAnnealingLR,
  CosineAnnealingWarmRestarts,
  ReduceLROnPlateau,
  LinearWarmup,
  type SchedulerConfig,
} from '../schedulers'

describe('StepLR', () => {
  test('returns correct SchedulerConfig structure', () => {
    const config = StepLR({ stepSize: 30 })

    expect(config.name).toContain('StepLR')
    expect(config.name).toContain('stepSize=30')
    expect(typeof config.create).toBe('function')
  })

  test('includes gamma in name', () => {
    const config = StepLR({ stepSize: 30, gamma: 0.5 })
    expect(config.name).toContain('gamma=0.5')
  })

  test('uses default gamma of 0.1', () => {
    const config = StepLR({ stepSize: 30 })
    expect(config.name).toContain('gamma=0.1')
  })

  test('does not set stepOn (defaults to epoch)', () => {
    const config = StepLR({ stepSize: 30 })
    expect(config.stepOn).toBeUndefined()
  })

  test('does not need metrics', () => {
    const config = StepLR({ stepSize: 30 })
    expect(config.needsMetrics).toBeUndefined()
  })
})

describe('MultiStepLR', () => {
  test('returns correct SchedulerConfig structure', () => {
    const config = MultiStepLR({ milestones: [30, 80] })

    expect(config.name).toContain('MultiStepLR')
    expect(config.name).toContain('milestones=[30,80]')
    expect(typeof config.create).toBe('function')
  })

  test('includes gamma in name', () => {
    const config = MultiStepLR({ milestones: [30], gamma: 0.5 })
    expect(config.name).toContain('gamma=0.5')
  })

  test('uses default gamma of 0.1', () => {
    const config = MultiStepLR({ milestones: [30] })
    expect(config.name).toContain('gamma=0.1')
  })
})

describe('ExponentialLR', () => {
  test('returns correct SchedulerConfig structure', () => {
    const config = ExponentialLR({ gamma: 0.95 })

    expect(config.name).toContain('ExponentialLR')
    expect(config.name).toContain('gamma=0.95')
    expect(typeof config.create).toBe('function')
  })
})

describe('CosineAnnealingLR', () => {
  test('returns correct SchedulerConfig structure', () => {
    const config = CosineAnnealingLR({ tMax: 50 })

    expect(config.name).toContain('CosineAnnealingLR')
    expect(config.name).toContain('T_max=50')
    expect(typeof config.create).toBe('function')
  })

  test('includes etaMin in name', () => {
    const config = CosineAnnealingLR({ tMax: 50, etaMin: 0.001 })
    expect(config.name).toContain('eta_min=0.001')
  })

  test('uses default etaMin of 0', () => {
    const config = CosineAnnealingLR({ tMax: 50 })
    expect(config.name).toContain('eta_min=0')
  })
})

describe('CosineAnnealingWarmRestarts', () => {
  test('returns correct SchedulerConfig structure', () => {
    const config = CosineAnnealingWarmRestarts({ t0: 10 })

    expect(config.name).toContain('CosineAnnealingWarmRestarts')
    expect(config.name).toContain('T_0=10')
    expect(typeof config.create).toBe('function')
  })

  test('includes tMult in name', () => {
    const config = CosineAnnealingWarmRestarts({ t0: 10, tMult: 2 })
    expect(config.name).toContain('T_mult=2')
  })

  test('uses default tMult of 1', () => {
    const config = CosineAnnealingWarmRestarts({ t0: 10 })
    expect(config.name).toContain('T_mult=1')
  })
})

describe('ReduceLROnPlateau', () => {
  test('returns correct SchedulerConfig structure', () => {
    const config = ReduceLROnPlateau()

    expect(config.name).toContain('ReduceLROnPlateau')
    expect(typeof config.create).toBe('function')
  })

  test('sets needsMetrics to true', () => {
    const config = ReduceLROnPlateau()
    expect(config.needsMetrics).toBe(true)
  })

  test('includes mode in name', () => {
    const config = ReduceLROnPlateau({ mode: 'max' })
    expect(config.name).toContain('mode=max')
  })

  test('uses default mode of min', () => {
    const config = ReduceLROnPlateau()
    expect(config.name).toContain('mode=min')
  })

  test('includes factor and patience in name', () => {
    const config = ReduceLROnPlateau({ factor: 0.5, patience: 5 })
    expect(config.name).toContain('factor=0.5')
    expect(config.name).toContain('patience=5')
  })

  test('uses default factor and patience', () => {
    const config = ReduceLROnPlateau()
    expect(config.name).toContain('factor=0.1')
    expect(config.name).toContain('patience=10')
  })
})

describe('LinearWarmup', () => {
  test('returns correct SchedulerConfig structure', () => {
    const config = LinearWarmup({ warmupSteps: 1000 })

    expect(config.name).toContain('LinearWarmup')
    expect(config.name).toContain('warmup_steps=1000')
    expect(typeof config.create).toBe('function')
  })

  test('sets stepOn to batch by default', () => {
    const config = LinearWarmup({ warmupSteps: 1000 })
    expect(config.stepOn).toBe('batch')
  })

  test('allows stepOn to be set to epoch', () => {
    const config = LinearWarmup({ warmupSteps: 10, stepOn: 'epoch' })
    expect(config.stepOn).toBe('epoch')
  })
})

describe('SchedulerConfig interface', () => {
  test('all factories return objects conforming to SchedulerConfig', () => {
    const configs: SchedulerConfig[] = [
      StepLR({ stepSize: 30 }),
      MultiStepLR({ milestones: [30, 80] }),
      ExponentialLR({ gamma: 0.95 }),
      CosineAnnealingLR({ tMax: 50 }),
      CosineAnnealingWarmRestarts({ t0: 10 }),
      ReduceLROnPlateau(),
      LinearWarmup({ warmupSteps: 1000 }),
    ]

    for (const config of configs) {
      expect(typeof config.name).toBe('string')
      expect(typeof config.create).toBe('function')
      // needsMetrics and stepOn are optional
      if (config.needsMetrics !== undefined) {
        expect(typeof config.needsMetrics).toBe('boolean')
      }
      if (config.stepOn !== undefined) {
        expect(['epoch', 'batch']).toContain(config.stepOn)
      }
    }
  })
})
