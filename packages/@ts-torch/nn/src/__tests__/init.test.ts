/**
 * Tests for weight initialization functions
 */

import { describe, test, expect } from 'vitest'
import {
  init,
  calculateFanInAndFanOut,
  calculateGain,
  constant_,
  zeros_,
  ones_,
  normal_,
  uniform_,
  kaiming_uniform_,
  kaiming_normal_,
  xavier_uniform_,
  xavier_normal_,
  orthogonal_,
  sparse_,
  trunc_normal_,
} from '../init.js'
import { device, run } from '@ts-torch/core'

const cpu = device.cpu()

describe('calculateFanInAndFanOut', () => {
  test('calculates fan for 2D weight tensor', () => {
    run(() => {
      const weight = cpu.randn([64, 128] as const)
      const [fanIn, fanOut] = calculateFanInAndFanOut(weight)

      expect(fanIn).toBe(128)
      expect(fanOut).toBe(64)
    })
  })

  test('calculates fan for conv2d weight tensor', () => {
    run(() => {
      // Conv weight: [out_channels, in_channels, kernel_h, kernel_w]
      const weight = cpu.randn([64, 32, 3, 3] as const)
      const [fanIn, fanOut] = calculateFanInAndFanOut(weight)

      // fan_in = in_channels * kernel_h * kernel_w
      expect(fanIn).toBe(32 * 3 * 3)
      // fan_out = out_channels * kernel_h * kernel_w
      expect(fanOut).toBe(64 * 3 * 3)
    })
  })

  test('throws error for 1D tensor', () => {
    run(() => {
      const weight = cpu.randn([64] as const)

      expect(() => calculateFanInAndFanOut(weight)).toThrow(
        '>= 2 dimensions',
      )
    })
  })
})

describe('calculateGain', () => {
  test('returns 1.0 for linear', () => {
    expect(calculateGain('linear')).toBe(1.0)
  })

  test('returns 1.0 for sigmoid', () => {
    expect(calculateGain('sigmoid')).toBe(1.0)
  })

  test('returns 5/3 for tanh', () => {
    expect(calculateGain('tanh')).toBeCloseTo(5.0 / 3)
  })

  test('returns sqrt(2) for relu', () => {
    expect(calculateGain('relu')).toBeCloseTo(Math.sqrt(2.0))
  })

  test('calculates gain for leaky_relu with default slope', () => {
    const gain = calculateGain('leaky_relu')
    const expected = Math.sqrt(2.0 / (1 + 0.01 * 0.01))

    expect(gain).toBeCloseTo(expected)
  })

  test('calculates gain for leaky_relu with custom slope', () => {
    const gain = calculateGain('leaky_relu', 0.2)
    const expected = Math.sqrt(2.0 / (1 + 0.2 * 0.2))

    expect(gain).toBeCloseTo(expected)
  })

  test('returns 0.75 for selu', () => {
    expect(calculateGain('selu')).toBeCloseTo(0.75)
  })

  test('throws error for unsupported nonlinearity', () => {
    expect(() => calculateGain('unknown')).toThrow('Unsupported nonlinearity')
  })
})

describe('constant_', () => {
  test('fills tensor with constant value', () => {
    run(() => {
      const tensor = cpu.randn([10, 10] as const)
      const result = constant_(tensor, 5.0)
      const data = result.toArray()

      for (const val of data) {
        expect(val).toBeCloseTo(5.0)
      }
    })
  })
})

describe('zeros_', () => {
  test('fills tensor with zeros', () => {
    run(() => {
      const tensor = cpu.randn([10, 10] as const)
      const result = zeros_(tensor)
      const data = result.toArray()

      for (const val of data) {
        expect(val).toBeCloseTo(0.0)
      }
    })
  })
})

describe('ones_', () => {
  test('fills tensor with ones', () => {
    run(() => {
      const tensor = cpu.randn([10, 10] as const)
      const result = ones_(tensor)
      const data = result.toArray()

      for (const val of data) {
        expect(val).toBeCloseTo(1.0)
      }
    })
  })
})

describe('normal_', () => {
  test('fills tensor with normal distribution', () => {
    run(() => {
      const tensor = cpu.randn([1000] as const)
      const result = normal_(tensor, 0, 1)
      const data = result.toArray()

      // Check approximate mean
      const mean = data.reduce((a, b) => a + b, 0) / data.length
      expect(mean).toBeCloseTo(0, 0) // Within 0.5 of 0
    })
  })

  test('respects mean and std parameters', () => {
    run(() => {
      const tensor = cpu.randn([1000] as const)
      const result = normal_(tensor, 10, 0.1)
      const data = result.toArray()

      const mean = data.reduce((a, b) => a + b, 0) / data.length
      expect(mean).toBeCloseTo(10, 0)
    })
  })
})

describe('uniform_', () => {
  test('fills tensor with uniform distribution', () => {
    run(() => {
      const tensor = cpu.randn([1000] as const)
      const result = uniform_(tensor, -1, 1)
      const data = result.toArray()

      // All values should be in range
      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(-1)
        expect(val).toBeLessThanOrEqual(1)
      }
    })
  })

  test('respects custom bounds', () => {
    run(() => {
      const tensor = cpu.randn([1000] as const)
      const result = uniform_(tensor, 5, 10)
      const data = result.toArray()

      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(5)
        expect(val).toBeLessThanOrEqual(10)
      }
    })
  })
})

describe('kaiming_uniform_', () => {
  test('initializes tensor with kaiming uniform distribution', () => {
    run(() => {
      const tensor = cpu.randn([64, 128] as const)
      const result = kaiming_uniform_(tensor)

      expect(result.shape).toEqual([64, 128])
    })
  })

  test('uses fan_in by default', () => {
    run(() => {
      const tensor = cpu.randn([64, 128] as const)
      const result = kaiming_uniform_(tensor, 0, 'fan_in', 'leaky_relu')

      expect(result.shape).toEqual([64, 128])
    })
  })

  test('can use fan_out mode', () => {
    run(() => {
      const tensor = cpu.randn([64, 128] as const)
      const result = kaiming_uniform_(tensor, 0, 'fan_out', 'relu')

      expect(result.shape).toEqual([64, 128])
    })
  })
})

describe('kaiming_normal_', () => {
  test('initializes tensor with kaiming normal distribution', () => {
    run(() => {
      const tensor = cpu.randn([64, 128] as const)
      const result = kaiming_normal_(tensor)

      expect(result.shape).toEqual([64, 128])
    })
  })

  test('supports different modes', () => {
    run(() => {
      const tensor = cpu.randn([64, 128] as const)
      const result = kaiming_normal_(tensor, 0, 'fan_out', 'relu')

      expect(result.shape).toEqual([64, 128])
    })
  })
})

describe('xavier_uniform_', () => {
  test('initializes tensor with xavier uniform distribution', () => {
    run(() => {
      const tensor = cpu.randn([64, 128] as const)
      const result = xavier_uniform_(tensor)

      expect(result.shape).toEqual([64, 128])
    })
  })

  test('accepts gain parameter', () => {
    run(() => {
      const tensor = cpu.randn([64, 128] as const)
      const gain = calculateGain('tanh')
      const result = xavier_uniform_(tensor, gain)

      expect(result.shape).toEqual([64, 128])
    })
  })
})

describe('xavier_normal_', () => {
  test('initializes tensor with xavier normal distribution', () => {
    run(() => {
      const tensor = cpu.randn([64, 128] as const)
      const result = xavier_normal_(tensor)

      expect(result.shape).toEqual([64, 128])
    })
  })

  test('accepts gain parameter', () => {
    run(() => {
      const tensor = cpu.randn([64, 128] as const)
      const result = xavier_normal_(tensor, 2.0)

      expect(result.shape).toEqual([64, 128])
    })
  })
})

describe('orthogonal_', () => {
  test('initializes tensor approximately orthogonal', () => {
    run(() => {
      const tensor = cpu.randn([64, 64] as const)
      const result = orthogonal_(tensor)

      expect(result.shape).toEqual([64, 64])
    })
  })

  test('accepts gain parameter', () => {
    run(() => {
      const tensor = cpu.randn([64, 64] as const)
      const result = orthogonal_(tensor, 2.0)

      expect(result.shape).toEqual([64, 64])
    })
  })

  test('handles non-square matrices', () => {
    run(() => {
      const tensor = cpu.randn([64, 128] as const)
      const result = orthogonal_(tensor)

      expect(result.shape).toEqual([64, 128])
    })
  })

  test('throws error for 1D tensor', () => {
    run(() => {
      const tensor = cpu.randn([64] as const)

      expect(() => orthogonal_(tensor)).toThrow('2 or more dimensions')
    })
  })
})

describe('sparse_', () => {
  test('initializes sparse tensor', () => {
    run(() => {
      const tensor = cpu.randn([64, 128] as const)
      const result = sparse_(tensor, 0.1)

      expect(result.shape).toEqual([64, 128])
    })
  })

  test('throws error for non-2D tensor', () => {
    run(() => {
      const tensor = cpu.randn([64, 64, 64] as const)

      expect(() => sparse_(tensor, 0.1)).toThrow('2D tensors')
    })
  })

  test('throws error for invalid sparsity', () => {
    run(() => {
      const tensor = cpu.randn([64, 128] as const)

      expect(() => sparse_(tensor, -0.1)).toThrow('between 0 and 1')
      expect(() => sparse_(tensor, 1.5)).toThrow('between 0 and 1')
    })
  })
})

describe('trunc_normal_', () => {
  test('initializes tensor with truncated normal distribution', () => {
    run(() => {
      const tensor = cpu.randn([1000] as const)
      const result = trunc_normal_(tensor, 0, 1, -2, 2)

      expect(result.shape).toEqual([1000])
    })
  })

  test('uses default parameters', () => {
    run(() => {
      const tensor = cpu.randn([64, 128] as const)
      const result = trunc_normal_(tensor)

      expect(result.shape).toEqual([64, 128])
    })
  })
})

describe('init namespace', () => {
  test('exports all functions', () => {
    expect(init.calculateFanInAndFanOut).toBe(calculateFanInAndFanOut)
    expect(init.calculateGain).toBe(calculateGain)
    expect(init.constant_).toBe(constant_)
    expect(init.zeros_).toBe(zeros_)
    expect(init.ones_).toBe(ones_)
    expect(init.normal_).toBe(normal_)
    expect(init.uniform_).toBe(uniform_)
    expect(init.kaiming_uniform_).toBe(kaiming_uniform_)
    expect(init.kaiming_normal_).toBe(kaiming_normal_)
    expect(init.xavier_uniform_).toBe(xavier_uniform_)
    expect(init.xavier_normal_).toBe(xavier_normal_)
    expect(init.orthogonal_).toBe(orthogonal_)
    expect(init.sparse_).toBe(sparse_)
    expect(init.trunc_normal_).toBe(trunc_normal_)
  })
})
