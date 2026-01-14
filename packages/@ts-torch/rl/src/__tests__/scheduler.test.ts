import { describe, it, expect } from 'vitest'
import type { SchedulerConfig, RLFitOptions, MemoryConfig } from '../trainer.js'

/**
 * These tests verify the type definitions and configuration options for
 * scheduler integration in the RL trainer. Full integration testing would
 * require mocking the agent and optimizer.
 */
describe('Scheduler Configuration Types', () => {
  const baseMemory: MemoryConfig = {
    capacity: 1000,
    batchSize: 32,
    warmup: 100,
  }

  describe('StepLR scheduler config', () => {
    it('accepts step scheduler with required fields', () => {
      const config: SchedulerConfig = {
        type: 'step',
        stepSize: 10,
      }
      expect(config.type).toBe('step')
      expect(config.stepSize).toBe(10)
    })

    it('accepts step scheduler with optional gamma', () => {
      const config: SchedulerConfig = {
        type: 'step',
        stepSize: 10,
        gamma: 0.5,
      }
      expect(config.gamma).toBe(0.5)
    })
  })

  describe('ExponentialLR scheduler config', () => {
    it('accepts exponential scheduler', () => {
      const config: SchedulerConfig = {
        type: 'exponential',
        gamma: 0.95,
      }
      expect(config.type).toBe('exponential')
      expect(config.gamma).toBe(0.95)
    })
  })

  describe('CosineAnnealingLR scheduler config', () => {
    it('accepts cosine scheduler with required fields', () => {
      const config: SchedulerConfig = {
        type: 'cosine',
        tMax: 100,
      }
      expect(config.type).toBe('cosine')
      expect(config.tMax).toBe(100)
    })

    it('accepts cosine scheduler with optional etaMin', () => {
      const config: SchedulerConfig = {
        type: 'cosine',
        tMax: 100,
        etaMin: 1e-6,
      }
      expect(config.etaMin).toBe(1e-6)
    })
  })

  describe('ReduceLROnPlateau scheduler config', () => {
    it('accepts plateau scheduler with required fields', () => {
      const config: SchedulerConfig = {
        type: 'plateau',
        patience: 10,
      }
      expect(config.type).toBe('plateau')
      expect(config.patience).toBe(10)
    })

    it('accepts plateau scheduler with optional fields', () => {
      const config: SchedulerConfig = {
        type: 'plateau',
        patience: 10,
        factor: 0.5,
        mode: 'max',
      }
      expect(config.factor).toBe(0.5)
      expect(config.mode).toBe('max')
    })
  })

  describe('LinearWarmup scheduler config', () => {
    it('accepts warmup scheduler', () => {
      const config: SchedulerConfig = {
        type: 'warmup',
        warmupSteps: 1000,
      }
      expect(config.type).toBe('warmup')
      expect(config.warmupSteps).toBe(1000)
    })
  })

  describe('RLFitOptions with scheduler', () => {
    it('accepts fit options with scheduler', () => {
      const options: Partial<RLFitOptions> = {
        episodes: 100,
        scheduler: {
          type: 'step',
          stepSize: 10,
          gamma: 0.9,
        },
        scheduleEvery: 'episode',
      }
      expect(options.scheduler?.type).toBe('step')
      expect(options.scheduleEvery).toBe('episode')
    })

    it('accepts fit options with step scheduling', () => {
      const options: Partial<RLFitOptions> = {
        episodes: 100,
        scheduler: {
          type: 'cosine',
          tMax: 1000,
        },
        scheduleEvery: 'step',
      }
      expect(options.scheduleEvery).toBe('step')
    })

    it('allows omitting scheduler', () => {
      const options: Partial<RLFitOptions> = {
        episodes: 100,
        memory: baseMemory,
      }
      expect(options.scheduler).toBeUndefined()
    })
  })

  describe('MemoryConfig with PER', () => {
    it('accepts memory config with PER options', () => {
      const memory: MemoryConfig = {
        capacity: 10000,
        batchSize: 64,
        warmup: 1000,
        per: {
          prioritized: true,
          alpha: 0.6,
          betaStart: 0.4,
          betaEnd: 1.0,
          betaFrames: 100000,
        },
      }
      expect(memory.per?.prioritized).toBe(true)
      expect(memory.per?.alpha).toBe(0.6)
    })

    it('allows omitting PER config', () => {
      const memory: MemoryConfig = {
        capacity: 10000,
        batchSize: 64,
        warmup: 1000,
      }
      expect(memory.per).toBeUndefined()
    })
  })
})

describe('Scheduler + PER combined config', () => {
  it('accepts full feature config', () => {
    const options: Partial<RLFitOptions> = {
      episodes: 1000,
      maxSteps: 500,
      strategy: {
        type: 'epsilon_greedy',
        start: 1.0,
        end: 0.01,
        decay: 0.995,
      },
      memory: {
        capacity: 50000,
        batchSize: 64,
        warmup: 1000,
        per: {
          prioritized: true,
          alpha: 0.6,
          betaStart: 0.4,
          betaEnd: 1.0,
          betaFrames: 100000,
        },
      },
      trainFreq: 4,
      scheduler: {
        type: 'cosine',
        tMax: 1000,
        etaMin: 1e-5,
      },
      scheduleEvery: 'episode',
      logEvery: 10,
      targetReward: 195,
      targetRewardWindow: 100,
    }

    expect(options.memory?.per?.prioritized).toBe(true)
    expect(options.scheduler?.type).toBe('cosine')
    expect(options.scheduleEvery).toBe('episode')
  })
})
