/**
 * Tests for learning rate schedulers
 */

import { describe, test, expect } from 'vitest'
import { device, run, float32 } from '@ts-torch/core'
import { SGD } from '../sgd'
import {
  StepLR,
  MultiStepLR,
  ExponentialLR,
  CosineAnnealingLR,
  CosineAnnealingWarmRestarts,
  ReduceLROnPlateau,
  LinearWarmup,
} from '../lr_scheduler'

const cpu = device.cpu()

/**
 * Helper function to create a mock optimizer with specified learning rate
 */
function createOptimizer(lr: number = 0.1) {
  const param = cpu.tensor([1.0, 2.0, 3.0], [3] as const, float32, true)
  return new SGD([param], { lr })
}

/**
 * Helper function to create an optimizer with multiple parameter groups
 */
function createMultiGroupOptimizer(lrs: number[] = [0.1, 0.01]) {
  const param1 = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
  const param2 = cpu.tensor([3.0, 4.0], [2] as const, float32, true)
  const optimizer = new SGD([param1], { lr: lrs[0] ?? 0.1 })
  optimizer.addParamGroup({ params: [param2], lr: lrs[1] ?? 0.01 })
  return optimizer
}

describe('StepLR', () => {
  describe('constructor', () => {
    test('creates scheduler with valid options', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new StepLR(optimizer, 10, 0.1)

        expect(scheduler.getLastEpoch()).toBe(0)
        expect(scheduler.getCurrentLr()[0]).toBe(0.1)
      })
    })

    test('throws error for non-positive step size', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        expect(() => new StepLR(optimizer, 0, 0.1)).toThrow('Step size must be positive')
        expect(() => new StepLR(optimizer, -5, 0.1)).toThrow('Step size must be positive')
      })
    })

    test('throws error for invalid gamma', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        expect(() => new StepLR(optimizer, 10, 0)).toThrow('Gamma must be in (0, 1]')
        expect(() => new StepLR(optimizer, 10, -0.1)).toThrow('Gamma must be in (0, 1]')
        expect(() => new StepLR(optimizer, 10, 1.5)).toThrow('Gamma must be in (0, 1]')
      })
    })

    test('accepts gamma equal to 1', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new StepLR(optimizer, 10, 1.0)
        expect(scheduler.getCurrentLr()[0]).toBe(0.1)
      })
    })
  })

  describe('step', () => {
    test('does not decay LR before step_size epochs', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new StepLR(optimizer, 10, 0.1)

        // Initial step already called in constructor, epoch = 0
        for (let i = 1; i < 10; i++) {
          scheduler.step()
          expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)
        }
      })
    })

    test('decays LR at step_size boundary', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new StepLR(optimizer, 10, 0.1)

        // Steps 0-9 should have lr = 0.1
        for (let i = 1; i < 10; i++) {
          scheduler.step()
        }
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)

        // Step 10 should decay to 0.01
        scheduler.step()
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.01, 10)
      })
    })

    test('decays LR multiple times', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new StepLR(optimizer, 5, 0.5)

        // epoch 0: lr = 0.1
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)

        // epochs 1-4: lr = 0.1
        for (let i = 1; i < 5; i++) {
          scheduler.step()
          expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)
        }

        // epoch 5: lr = 0.05
        scheduler.step()
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.05, 10)

        // epochs 6-9: lr = 0.05
        for (let i = 6; i < 10; i++) {
          scheduler.step()
          expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.05, 10)
        }

        // epoch 10: lr = 0.025
        scheduler.step()
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.025, 10)
      })
    })

    test('works with multiple parameter groups', () => {
      run(() => {
        const optimizer = createMultiGroupOptimizer([0.1, 0.01])
        const scheduler = new StepLR(optimizer, 5, 0.1)

        // Initial LRs
        const initialLrs = scheduler.getCurrentLr()
        expect(initialLrs[0]).toBeCloseTo(0.1, 10)
        expect(initialLrs[1]).toBeCloseTo(0.01, 10)

        // Step to epoch 5
        for (let i = 1; i <= 5; i++) {
          scheduler.step()
        }

        // Both should be decayed
        const newLrs = scheduler.getCurrentLr()
        expect(newLrs[0]).toBeCloseTo(0.01, 10)
        expect(newLrs[1]).toBeCloseTo(0.001, 10)
      })
    })
  })
})

describe('MultiStepLR', () => {
  describe('constructor', () => {
    test('creates scheduler with valid options', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new MultiStepLR(optimizer, [10, 20, 30], 0.1)

        expect(scheduler.getLastEpoch()).toBe(0)
        expect(scheduler.getCurrentLr()[0]).toBe(0.1)
      })
    })

    test('throws error for invalid gamma', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        expect(() => new MultiStepLR(optimizer, [10], 0)).toThrow('Gamma must be in (0, 1]')
        expect(() => new MultiStepLR(optimizer, [10], -0.1)).toThrow('Gamma must be in (0, 1]')
        expect(() => new MultiStepLR(optimizer, [10], 1.5)).toThrow('Gamma must be in (0, 1]')
      })
    })
  })

  describe('step', () => {
    test('does not decay LR before first milestone', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new MultiStepLR(optimizer, [10, 20], 0.1)

        for (let i = 1; i < 10; i++) {
          scheduler.step()
          expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)
        }
      })
    })

    test('decays LR at milestone boundaries', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new MultiStepLR(optimizer, [5, 10], 0.1)

        // epochs 0-4: lr = 0.1
        for (let i = 1; i < 5; i++) {
          scheduler.step()
          expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)
        }

        // epoch 5: lr = 0.01 (first milestone)
        scheduler.step()
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.01, 10)

        // epochs 6-9: lr = 0.01
        for (let i = 6; i < 10; i++) {
          scheduler.step()
          expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.01, 10)
        }

        // epoch 10: lr = 0.001 (second milestone)
        scheduler.step()
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.001, 10)
      })
    })

    test('handles unsorted milestones', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        // Milestones provided in unsorted order
        const scheduler = new MultiStepLR(optimizer, [10, 5], 0.1)

        // Should still decay at epoch 5 first
        for (let i = 1; i < 5; i++) {
          scheduler.step()
          expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)
        }

        scheduler.step() // epoch 5
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.01, 10)
      })
    })

    test('handles duplicate milestones', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        // Duplicate milestone at 5
        const scheduler = new MultiStepLR(optimizer, [5, 5, 10], 0.1)

        for (let i = 1; i < 5; i++) {
          scheduler.step()
        }

        scheduler.step() // epoch 5 - should only decay once due to Set
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.01, 10)
      })
    })

    test('works with multiple parameter groups', () => {
      run(() => {
        const optimizer = createMultiGroupOptimizer([0.1, 0.01])
        const scheduler = new MultiStepLR(optimizer, [5], 0.1)

        // Step to milestone
        for (let i = 1; i <= 5; i++) {
          scheduler.step()
        }

        const newLrs = scheduler.getCurrentLr()
        expect(newLrs[0]).toBeCloseTo(0.01, 10)
        expect(newLrs[1]).toBeCloseTo(0.001, 10)
      })
    })
  })
})

describe('ExponentialLR', () => {
  describe('constructor', () => {
    test('creates scheduler with valid options', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new ExponentialLR(optimizer, 0.9)

        expect(scheduler.getLastEpoch()).toBe(0)
        expect(scheduler.getCurrentLr()[0]).toBe(0.1)
      })
    })

    test('throws error for invalid gamma', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        expect(() => new ExponentialLR(optimizer, 0)).toThrow('Gamma must be in (0, 1]')
        expect(() => new ExponentialLR(optimizer, -0.1)).toThrow('Gamma must be in (0, 1]')
        expect(() => new ExponentialLR(optimizer, 1.5)).toThrow('Gamma must be in (0, 1]')
      })
    })
  })

  describe('step', () => {
    test('decays LR exponentially each step', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new ExponentialLR(optimizer, 0.9)

        // epoch 0: lr = 0.1 * 0.9^0 = 0.1
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)

        // epoch 1: lr = 0.1 * 0.9^1 = 0.09
        scheduler.step()
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.09, 10)

        // epoch 2: lr = 0.1 * 0.9^2 = 0.081
        scheduler.step()
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.081, 10)

        // epoch 3: lr = 0.1 * 0.9^3 = 0.0729
        scheduler.step()
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.0729, 10)
      })
    })

    test('maintains constant LR with gamma=1', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new ExponentialLR(optimizer, 1.0)

        for (let i = 0; i < 10; i++) {
          expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)
          scheduler.step()
        }
      })
    })

    test('works with multiple parameter groups', () => {
      run(() => {
        const optimizer = createMultiGroupOptimizer([0.1, 0.01])
        const scheduler = new ExponentialLR(optimizer, 0.5)

        scheduler.step() // epoch 1

        const newLrs = scheduler.getCurrentLr()
        expect(newLrs[0]).toBeCloseTo(0.05, 10)
        expect(newLrs[1]).toBeCloseTo(0.005, 10)
      })
    })
  })
})

describe('CosineAnnealingLR', () => {
  describe('constructor', () => {
    test('creates scheduler with valid options', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new CosineAnnealingLR(optimizer, 50, 0.001)

        expect(scheduler.getLastEpoch()).toBe(0)
        expect(scheduler.getCurrentLr()[0]).toBe(0.1)
      })
    })

    test('throws error for non-positive T_max', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        expect(() => new CosineAnnealingLR(optimizer, 0)).toThrow('T_max must be positive')
        expect(() => new CosineAnnealingLR(optimizer, -10)).toThrow('T_max must be positive')
      })
    })

    test('throws error for negative eta_min', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        expect(() => new CosineAnnealingLR(optimizer, 50, -0.001)).toThrow(
          'eta_min must be non-negative',
        )
      })
    })
  })

  describe('step', () => {
    test('starts at base LR and reaches eta_min at T_max', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new CosineAnnealingLR(optimizer, 10, 0.01)

        // epoch 0: lr = 0.1
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)

        // Step to T_max
        for (let i = 1; i <= 10; i++) {
          scheduler.step()
        }

        // epoch 10: lr = eta_min = 0.01
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.01, 10)
      })
    })

    test('follows cosine curve at midpoint', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new CosineAnnealingLR(optimizer, 10, 0.0)

        // Step to midpoint (epoch 5)
        for (let i = 1; i <= 5; i++) {
          scheduler.step()
        }

        // At midpoint, lr should be around (0.1 + 0) / 2 = 0.05
        // cos(pi * 5 / 10) = cos(pi/2) = 0
        // lr = 0 + (0.1 - 0) * (1 + 0) / 2 = 0.05
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.05, 5)
      })
    })

    test('handles eta_min = 0', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new CosineAnnealingLR(optimizer, 10)

        // Step to T_max
        for (let i = 1; i <= 10; i++) {
          scheduler.step()
        }

        // Should reach eta_min = 0
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0, 10)
      })
    })

    test('works with multiple parameter groups', () => {
      run(() => {
        const optimizer = createMultiGroupOptimizer([0.1, 0.01])
        const scheduler = new CosineAnnealingLR(optimizer, 10, 0.0)

        // Step to T_max
        for (let i = 1; i <= 10; i++) {
          scheduler.step()
        }

        const newLrs = scheduler.getCurrentLr()
        expect(newLrs[0]).toBeCloseTo(0, 10)
        expect(newLrs[1]).toBeCloseTo(0, 10)
      })
    })
  })
})

describe('CosineAnnealingWarmRestarts', () => {
  describe('constructor', () => {
    test('creates scheduler with valid options', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new CosineAnnealingWarmRestarts(optimizer, 10, 2, 0.001)

        expect(scheduler.getLastEpoch()).toBe(0)
        expect(scheduler.getCurrentLr()[0]).toBe(0.1)
      })
    })

    test('throws error for non-positive T_0', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        expect(() => new CosineAnnealingWarmRestarts(optimizer, 0)).toThrow('T_0 must be positive')
        expect(() => new CosineAnnealingWarmRestarts(optimizer, -5)).toThrow('T_0 must be positive')
      })
    })

    test('throws error for T_mult < 1', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        expect(() => new CosineAnnealingWarmRestarts(optimizer, 10, 0.5)).toThrow(
          'T_mult must be >= 1',
        )
      })
    })

    test('throws error for negative eta_min', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        expect(() => new CosineAnnealingWarmRestarts(optimizer, 10, 1, -0.001)).toThrow(
          'eta_min must be non-negative',
        )
      })
    })
  })

  describe('step', () => {
    test('restarts LR after T_0 epochs with T_mult=1', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new CosineAnnealingWarmRestarts(optimizer, 5, 1, 0.0)

        // epoch 0: lr = 0.1
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)

        // Step through first cycle (epochs 1-4)
        for (let i = 1; i < 5; i++) {
          scheduler.step()
        }

        // epoch 5: restart, lr should be back to 0.1
        scheduler.step()
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 5)
      })
    })

    test('increases period with T_mult > 1', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new CosineAnnealingWarmRestarts(optimizer, 5, 2, 0.0)

        // First cycle: 5 epochs
        // Step through first cycle
        for (let i = 1; i <= 5; i++) {
          scheduler.step()
        }
        // After restart, lr should be high
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 5)

        // Second cycle: 10 epochs (T_0 * T_mult = 5 * 2)
        // Step through second cycle
        for (let i = 1; i <= 10; i++) {
          scheduler.step()
        }
        // After second restart
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 5)
      })
    })

    test('reaches eta_min at end of each cycle', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new CosineAnnealingWarmRestarts(optimizer, 5, 1, 0.01)

        // Step to just before restart (epoch 4)
        for (let i = 1; i < 5; i++) {
          scheduler.step()
        }

        // At tCur = T_i - 1 (near end of cycle), LR should be near eta_min
        // Actually at tCur = 4, cos(pi * 4 / 5) is close to -1
        // lr = 0.01 + (0.1 - 0.01) * (1 + cos(pi * 4/5)) / 2
        const lr = scheduler.getCurrentLr()[0]
        expect(lr).toBeLessThan(0.05)
        expect(lr).toBeGreaterThanOrEqual(0.01)
      })
    })

    test('works with multiple parameter groups', () => {
      run(() => {
        const optimizer = createMultiGroupOptimizer([0.1, 0.01])
        const scheduler = new CosineAnnealingWarmRestarts(optimizer, 5, 1, 0.0)

        // Step through first cycle and restart
        for (let i = 1; i <= 5; i++) {
          scheduler.step()
        }

        // Both should restart to base LR
        const newLrs = scheduler.getCurrentLr()
        expect(newLrs[0]).toBeCloseTo(0.1, 5)
        expect(newLrs[1]).toBeCloseTo(0.01, 5)
      })
    })
  })
})

describe('ReduceLROnPlateau', () => {
  describe('constructor', () => {
    test('creates scheduler with default options', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new ReduceLROnPlateau(optimizer, 'min')

        expect(scheduler.getCurrentLr()[0]).toBe(0.1)
      })
    })

    test('creates scheduler with custom options', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new ReduceLROnPlateau(optimizer, 'max', {
          factor: 0.5,
          patience: 5,
          threshold: 0.01,
          cooldown: 2,
          minLr: 0.001,
        })

        expect(scheduler.getCurrentLr()[0]).toBe(0.1)
      })
    })

    test('throws error for factor >= 1', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        expect(() => new ReduceLROnPlateau(optimizer, 'min', { factor: 1.0 })).toThrow(
          'Factor should be < 1.0',
        )
        expect(() => new ReduceLROnPlateau(optimizer, 'min', { factor: 1.5 })).toThrow(
          'Factor should be < 1.0',
        )
      })
    })

    test('throws error for negative patience', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        expect(() => new ReduceLROnPlateau(optimizer, 'min', { patience: -1 })).toThrow(
          'Patience should be non-negative',
        )
      })
    })
  })

  describe('step', () => {
    test('requires metric value', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new ReduceLROnPlateau(optimizer, 'min')

        expect(() => scheduler.step()).toThrow('ReduceLROnPlateau requires a metric value')
      })
    })

    test('does not reduce LR when improving (min mode)', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new ReduceLROnPlateau(optimizer, 'min', { patience: 3 })

        // Continuously improving (decreasing loss)
        scheduler.step(1.0)
        scheduler.step(0.9)
        scheduler.step(0.8)
        scheduler.step(0.7)
        scheduler.step(0.6)

        expect(scheduler.getCurrentLr()[0]).toBe(0.1)
      })
    })

    test('reduces LR after patience epochs without improvement (min mode)', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new ReduceLROnPlateau(optimizer, 'min', { patience: 2, factor: 0.5 })

        // Initial metric
        scheduler.step(1.0)
        // No improvement for patience + 1 epochs
        scheduler.step(1.0)
        scheduler.step(1.0)
        scheduler.step(1.0) // This should trigger reduction

        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.05, 10)
      })
    })

    test('works in max mode', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new ReduceLROnPlateau(optimizer, 'max', { patience: 2, factor: 0.5 })

        // Improving (increasing metric)
        scheduler.step(1.0)
        scheduler.step(1.1)
        scheduler.step(1.2)
        expect(scheduler.getCurrentLr()[0]).toBe(0.1)

        // No improvement
        scheduler.step(1.2)
        scheduler.step(1.2)
        scheduler.step(1.2) // Should trigger reduction

        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.05, 10)
      })
    })

    test('respects cooldown period', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new ReduceLROnPlateau(optimizer, 'min', {
          patience: 1,
          factor: 0.5,
          cooldown: 2,
        })

        // Trigger first reduction
        scheduler.step(1.0)
        scheduler.step(1.0)
        scheduler.step(1.0) // First reduction
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.05, 10)

        // During cooldown, no reduction should occur
        scheduler.step(1.0)
        scheduler.step(1.0)
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.05, 10)

        // After cooldown, can reduce again
        scheduler.step(1.0)
        scheduler.step(1.0)
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.025, 10)
      })
    })

    test('respects minLr', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new ReduceLROnPlateau(optimizer, 'min', {
          patience: 0,
          factor: 0.1,
          minLr: 0.01,
        })

        // Multiple reductions
        scheduler.step(1.0)
        scheduler.step(1.0) // 0.1 -> 0.01
        scheduler.step(1.0) // Should not go below 0.01

        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.01, 10)
      })
    })

    test('uses relative threshold mode correctly', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new ReduceLROnPlateau(optimizer, 'min', {
          patience: 1,
          factor: 0.5,
          threshold: 0.1, // 10% improvement required
          thresholdMode: 'rel',
        })

        // Initial
        scheduler.step(1.0)

        // 5% improvement - not enough
        scheduler.step(0.95)
        scheduler.step(0.95)
        // Should reduce because 5% < 10% required
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.05, 10)
      })
    })

    test('uses absolute threshold mode correctly', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new ReduceLROnPlateau(optimizer, 'min', {
          patience: 1,
          factor: 0.5,
          threshold: 0.1, // 0.1 absolute improvement required
          thresholdMode: 'abs',
        })

        // Initial
        scheduler.step(1.0)

        // Small improvement - not enough
        scheduler.step(0.95)
        scheduler.step(0.95)
        // Should reduce because 0.05 < 0.1 required
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.05, 10)
      })
    })

    test('works with multiple parameter groups and array minLr', () => {
      run(() => {
        const optimizer = createMultiGroupOptimizer([0.1, 0.01])
        const scheduler = new ReduceLROnPlateau(optimizer, 'min', {
          patience: 0,
          factor: 0.1,
          minLr: [0.01, 0.001],
        })

        // Trigger reductions
        scheduler.step(1.0)
        scheduler.step(1.0)
        scheduler.step(1.0)

        const lrs = scheduler.getCurrentLr()
        expect(lrs[0]).toBeCloseTo(0.01, 10) // Capped at minLr[0]
        expect(lrs[1]).toBeCloseTo(0.001, 10) // Capped at minLr[1]
      })
    })
  })
})

describe('LinearWarmup', () => {
  describe('constructor', () => {
    test('creates scheduler with valid options', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new LinearWarmup(optimizer, 100)

        expect(scheduler.getLastEpoch()).toBe(0)
        // At epoch 0, lr = 0.1 * 0/100 = 0
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0, 10)
      })
    })

    test('throws error for non-positive warmup steps', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        expect(() => new LinearWarmup(optimizer, 0)).toThrow('Warmup steps must be positive')
        expect(() => new LinearWarmup(optimizer, -10)).toThrow('Warmup steps must be positive')
      })
    })
  })

  describe('step', () => {
    test('linearly increases LR from 0 to base LR', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new LinearWarmup(optimizer, 10)

        // epoch 0: lr = 0.1 * 0/10 = 0
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0, 10)

        // epoch 1: lr = 0.1 * 1/10 = 0.01
        scheduler.step()
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.01, 10)

        // epoch 5: lr = 0.1 * 5/10 = 0.05
        for (let i = 2; i <= 5; i++) {
          scheduler.step()
        }
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.05, 10)

        // epoch 10: lr = 0.1 (reached base lr)
        for (let i = 6; i <= 10; i++) {
          scheduler.step()
        }
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)
      })
    })

    test('maintains base LR after warmup completes', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new LinearWarmup(optimizer, 5)

        // Complete warmup
        for (let i = 1; i <= 5; i++) {
          scheduler.step()
        }
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)

        // Continue stepping beyond warmup
        scheduler.step()
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)

        scheduler.step()
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)
      })
    })

    test('works with multiple parameter groups', () => {
      run(() => {
        const optimizer = createMultiGroupOptimizer([0.1, 0.01])
        const scheduler = new LinearWarmup(optimizer, 10)

        // epoch 0: both at 0
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0, 10)
        expect(scheduler.getCurrentLr()[1]).toBeCloseTo(0, 10)

        // epoch 5: lr = baseLr * 5/10
        for (let i = 1; i <= 5; i++) {
          scheduler.step()
        }
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.05, 10)
        expect(scheduler.getCurrentLr()[1]).toBeCloseTo(0.005, 10)

        // epoch 10: full base lr
        for (let i = 6; i <= 10; i++) {
          scheduler.step()
        }
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1, 10)
        expect(scheduler.getCurrentLr()[1]).toBeCloseTo(0.01, 10)
      })
    })
  })
})

describe('LRScheduler base class', () => {
  describe('getLastEpoch', () => {
    test('returns correct epoch number', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        const scheduler = new StepLR(optimizer, 10, 0.1)

        expect(scheduler.getLastEpoch()).toBe(0)

        scheduler.step()
        expect(scheduler.getLastEpoch()).toBe(1)

        scheduler.step()
        expect(scheduler.getLastEpoch()).toBe(2)
      })
    })
  })

  describe('getCurrentLr', () => {
    test('returns array of learning rates for all parameter groups', () => {
      run(() => {
        const optimizer = createMultiGroupOptimizer([0.1, 0.01])
        const scheduler = new StepLR(optimizer, 10, 0.1)

        const lrs = scheduler.getCurrentLr()
        expect(lrs).toHaveLength(2)
        expect(lrs[0]).toBe(0.1)
        expect(lrs[1]).toBe(0.01)
      })
    })
  })

  describe('lastEpoch parameter', () => {
    test('StepLR respects lastEpoch in constructor', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        // Start at epoch 14 with step_size=10, gamma=0.1
        // floor(14/10) = 1, so lr = 0.1 * 0.1^1 = 0.01
        const scheduler = new StepLR(optimizer, 10, 0.1, 14)

        expect(scheduler.getLastEpoch()).toBe(14)
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.01, 10)
      })
    })

    test('ExponentialLR respects lastEpoch in constructor', () => {
      run(() => {
        const optimizer = createOptimizer(0.1)
        // Start at epoch 4 with gamma=0.9
        // lr = 0.1 * 0.9^4
        const scheduler = new ExponentialLR(optimizer, 0.9, 4)

        expect(scheduler.getLastEpoch()).toBe(4)
        expect(scheduler.getCurrentLr()[0]).toBeCloseTo(0.1 * Math.pow(0.9, 4), 10)
      })
    })
  })
})
