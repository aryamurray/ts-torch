/**
 * Tests for gradient clipping utilities
 */

import { describe, test, expect } from 'vitest'
import { device, run, float32 } from '@ts-torch/core'
import {
  clipGradNorm,
  clipGradNorm_,
  clipGradValue,
  clipGradValue_,
  getGradNorm,
  checkGradHealth,
} from '../grad_clip'

const cpu = device.cpu()

describe('clipGradNorm', () => {
  test('returns total gradient norm before clipping', () => {
    run(() => {
      // Create parameters with gradients
      const param1 = cpu.tensor([1.0, 2.0, 3.0], [3] as const, float32, true)
      const param2 = cpu.tensor([4.0, 5.0], [2] as const, float32, true)

      // Simulate backward pass by setting grad
      ;(param1 as any)._gradCache = cpu.tensor([3.0, 4.0, 0.0], [3] as const)
      ;(param2 as any)._gradCache = cpu.tensor([0.0, 0.0], [2] as const)

      // L2 norm = sqrt(3^2 + 4^2) = 5
      const totalNorm = clipGradNorm([param1, param2], 10.0)

      expect(totalNorm).toBeCloseTo(5.0, 4)
    })
  })

  test('clips gradients when norm exceeds maxNorm', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      // Set gradient with norm = sqrt(6^2 + 8^2) = 10
      ;(param as any)._gradCache = cpu.tensor([6.0, 8.0], [2] as const)

      const totalNorm = clipGradNorm([param], 5.0)

      expect(totalNorm).toBeCloseTo(10.0, 4)
      // After clipping, gradient norm should be approximately 5.0
      // Gradient is scaled by 5/10 = 0.5
    })
  })

  test('does not clip when norm is below maxNorm', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      // Set gradient with norm = sqrt(3^2 + 4^2) = 5
      ;(param as any)._gradCache = cpu.tensor([3.0, 4.0], [2] as const)

      const totalNorm = clipGradNorm([param], 10.0)

      expect(totalNorm).toBeCloseTo(5.0, 4)
      // Gradient should not be clipped
    })
  })

  test('returns 0 for empty parameter list', () => {
    const totalNorm = clipGradNorm([], 1.0)

    expect(totalNorm).toBe(0.0)
  })

  test('returns 0 when no parameters have gradients', () => {
    run(() => {
      const param1 = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      const param2 = cpu.tensor([3.0, 4.0], [2] as const, float32, true)
      // Don't set any gradients

      const totalNorm = clipGradNorm([param1, param2], 1.0)

      expect(totalNorm).toBe(0.0)
    })
  })

  test('supports infinity norm', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      ;(param as any)._gradCache = cpu.tensor([3.0, 5.0], [2] as const)

      const totalNorm = clipGradNorm([param], 10.0, Infinity)

      expect(totalNorm).toBeCloseTo(5.0, 4) // Max absolute value
    })
  })

  test('supports L1 norm', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      ;(param as any)._gradCache = cpu.tensor([3.0, 4.0], [2] as const)

      const totalNorm = clipGradNorm([param], 10.0, 1.0)

      expect(totalNorm).toBeCloseTo(7.0, 4) // |3| + |4| = 7
    })
  })

  test('throws error for non-finite norm when errorIfNonfinite is true', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      ;(param as any)._gradCache = cpu.tensor([Infinity, 0.0], [2] as const)

      expect(() => clipGradNorm([param], 1.0, 2.0, true)).toThrow('non-finite')
    })
  })

  test('does not throw for non-finite norm when errorIfNonfinite is false', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      ;(param as any)._gradCache = cpu.tensor([Infinity, 0.0], [2] as const)

      expect(() => clipGradNorm([param], 1.0, 2.0, false)).not.toThrow()
    })
  })

  test('clipGradNorm_ is an alias for clipGradNorm', () => {
    expect(clipGradNorm_).toBe(clipGradNorm)
  })
})

describe('clipGradValue', () => {
  test('clips gradient values to specified range', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      ;(param as any)._gradCache = cpu.tensor([10.0, -10.0], [2] as const)

      clipGradValue([param], 5.0)

      // Gradient values should be clipped to [-5, 5]
      const grad = (param as any)._gradCache
      const gradData = grad.toArray()
      expect(gradData[0]).toBeLessThanOrEqual(5.0)
      expect(gradData[1]).toBeGreaterThanOrEqual(-5.0)
    })
  })

  test('throws error for non-positive clipValue', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      ;(param as any)._gradCache = cpu.tensor([1.0, 1.0], [2] as const)

      expect(() => clipGradValue([param], 0)).toThrow('positive')
      expect(() => clipGradValue([param], -1)).toThrow('positive')
    })
  })

  test('does nothing when gradient values are within range', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      ;(param as any)._gradCache = cpu.tensor([0.5, -0.5], [2] as const)

      clipGradValue([param], 1.0)

      const grad = (param as any)._gradCache
      const gradData = grad.toArray()
      expect(gradData[0]).toBeCloseTo(0.5, 4)
      expect(gradData[1]).toBeCloseTo(-0.5, 4)
    })
  })

  test('handles multiple parameters', () => {
    run(() => {
      const param1 = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      const param2 = cpu.tensor([3.0, 4.0], [2] as const, float32, true)
      ;(param1 as any)._gradCache = cpu.tensor([100.0, -100.0], [2] as const)
      ;(param2 as any)._gradCache = cpu.tensor([0.1, -0.1], [2] as const)

      clipGradValue([param1, param2], 1.0)

      const grad1 = (param1 as any)._gradCache
      const grad1Data = grad1.toArray()
      expect(grad1Data[0]).toBeLessThanOrEqual(1.0)
      expect(grad1Data[1]).toBeGreaterThanOrEqual(-1.0)

      // param2 should be unchanged as it's within range
      const grad2 = (param2 as any)._gradCache
      const grad2Data = grad2.toArray()
      expect(grad2Data[0]).toBeCloseTo(0.1, 4)
    })
  })

  test('clipGradValue_ is an alias for clipGradValue', () => {
    expect(clipGradValue_).toBe(clipGradValue)
  })
})

describe('getGradNorm', () => {
  test('computes L2 norm of gradients', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      ;(param as any)._gradCache = cpu.tensor([3.0, 4.0], [2] as const)

      const norm = getGradNorm([param])

      expect(norm).toBeCloseTo(5.0, 4)
    })
  })

  test('returns 0 for empty parameter list', () => {
    const norm = getGradNorm([])

    expect(norm).toBe(0.0)
  })

  test('returns 0 when no parameters have gradients', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)

      const norm = getGradNorm([param])

      expect(norm).toBe(0.0)
    })
  })

  test('computes infinity norm', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      ;(param as any)._gradCache = cpu.tensor([3.0, 7.0], [2] as const)

      const norm = getGradNorm([param], Infinity)

      expect(norm).toBeCloseTo(7.0, 4)
    })
  })

  test('computes L1 norm', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      ;(param as any)._gradCache = cpu.tensor([3.0, 4.0], [2] as const)

      const norm = getGradNorm([param], 1.0)

      expect(norm).toBeCloseTo(7.0, 4)
    })
  })

  test('aggregates across multiple parameters', () => {
    run(() => {
      const param1 = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      const param2 = cpu.tensor([3.0, 4.0], [2] as const, float32, true)
      ;(param1 as any)._gradCache = cpu.tensor([3.0, 0.0], [2] as const)
      ;(param2 as any)._gradCache = cpu.tensor([0.0, 4.0], [2] as const)

      const norm = getGradNorm([param1, param2])

      // sqrt(3^2 + 0 + 0 + 4^2) = sqrt(9 + 16) = 5
      expect(norm).toBeCloseTo(5.0, 4)
    })
  })
})

describe('checkGradHealth', () => {
  test('detects NaN in gradients', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      ;(param as any)._gradCache = cpu.tensor([NaN, 1.0], [2] as const)

      const result = checkGradHealth([['param', param]])

      expect(result.hasNaN).toBe(true)
      expect(result.nanParams).toContain('param')
    })
  })

  test('detects Inf in gradients', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      ;(param as any)._gradCache = cpu.tensor([Infinity, 1.0], [2] as const)

      const result = checkGradHealth([['param', param]])

      expect(result.hasInf).toBe(true)
      expect(result.infParams).toContain('param')
    })
  })

  test('returns healthy status for normal gradients', () => {
    run(() => {
      const param1 = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      const param2 = cpu.tensor([3.0, 4.0], [2] as const, float32, true)
      ;(param1 as any)._gradCache = cpu.tensor([0.5, -0.5], [2] as const)
      ;(param2 as any)._gradCache = cpu.tensor([0.1, -0.1], [2] as const)

      const result = checkGradHealth([
        ['param1', param1],
        ['param2', param2],
      ])

      expect(result.hasNaN).toBe(false)
      expect(result.hasInf).toBe(false)
      expect(result.nanParams).toHaveLength(0)
      expect(result.infParams).toHaveLength(0)
    })
  })

  test('handles parameters without gradients', () => {
    run(() => {
      const param = cpu.tensor([1.0, 2.0], [2] as const, float32, true)

      const result = checkGradHealth([['param', param]])

      expect(result.hasNaN).toBe(false)
      expect(result.hasInf).toBe(false)
    })
  })

  test('identifies multiple problematic parameters', () => {
    run(() => {
      const param1 = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      const param2 = cpu.tensor([3.0, 4.0], [2] as const, float32, true)
      ;(param1 as any)._gradCache = cpu.tensor([NaN, 1.0], [2] as const)
      ;(param2 as any)._gradCache = cpu.tensor([Infinity, 1.0], [2] as const)

      const result = checkGradHealth([
        ['param1', param1],
        ['param2', param2],
      ])

      expect(result.hasNaN).toBe(true)
      expect(result.hasInf).toBe(true)
      expect(result.nanParams).toContain('param1')
      expect(result.infParams).toContain('param2')
    })
  })
})
