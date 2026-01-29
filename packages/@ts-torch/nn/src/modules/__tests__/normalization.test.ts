/**
 * Tests for normalization layers (GroupNorm, InstanceNorm2d)
 */

import { describe, test, expect } from 'vitest'
import {
  BatchNorm2d,
  BatchNorm1d,
  LayerNorm,
  GroupNorm,
  InstanceNorm2d,
} from '../normalization.js'
import { device, run } from '@ts-torch/core'

const cpu = device.cpu()

describe('GroupNorm', () => {
  describe('constructor', () => {
    test('creates GroupNorm with correct parameters', () => {
      const gn = new GroupNorm(8, 32)

      expect(gn.numGroups).toBe(8)
      expect(gn.numChannels).toBe(32)
      expect(gn.eps).toBe(1e-5)
      expect(gn.affine).toBe(true)
    })

    test('throws error when channels not divisible by groups', () => {
      expect(() => new GroupNorm(7, 32)).toThrow('divisible by')
    })

    test('creates affine parameters by default', () => {
      const gn = new GroupNorm(8, 32)

      expect(gn.weight).not.toBeNull()
      expect(gn.biasParam).not.toBeNull()
      expect(gn.weight!.data.shape).toEqual([32])
      expect(gn.biasParam!.data.shape).toEqual([32])
    })

    test('can disable affine parameters', () => {
      const gn = new GroupNorm(8, 32, { affine: false })

      expect(gn.weight).toBeNull()
      expect(gn.biasParam).toBeNull()
    })

    test('accepts custom eps', () => {
      const gn = new GroupNorm(8, 32, { eps: 1e-6 })

      expect(gn.eps).toBe(1e-6)
    })

    test('registers parameters when affine', () => {
      const gn = new GroupNorm(8, 32)
      const params = gn.parameters()

      expect(params).toHaveLength(2)
    })

    test('no parameters when not affine', () => {
      const gn = new GroupNorm(8, 32, { affine: false })
      const params = gn.parameters()

      expect(params).toHaveLength(0)
    })
  })

  describe('forward pass', () => {
    test('normalizes 4D input (N, C, H, W)', () => {
      run(() => {
        const gn = new GroupNorm(4, 16)
        const input = cpu.randn([2, 16, 8, 8] as const)

        const output = gn.forward(input)

        expect(output.shape).toEqual([2, 16, 8, 8])
      })
    })

    test('normalizes 3D input (N, C, L)', () => {
      run(() => {
        const gn = new GroupNorm(4, 16)
        const input = cpu.randn([2, 16, 100] as const)

        const output = gn.forward(input)

        expect(output.shape).toEqual([2, 16, 100])
      })
    })

    test('handles different group configurations', () => {
      run(() => {
        // 1 group per channel (equivalent to InstanceNorm)
        const gn1 = new GroupNorm(32, 32)
        const input = cpu.randn([2, 32, 4, 4] as const)
        const output1 = gn1.forward(input)

        expect(output1.shape).toEqual([2, 32, 4, 4])

        // All channels in one group (equivalent to LayerNorm on C,H,W)
        const gn2 = new GroupNorm(1, 32)
        const output2 = gn2.forward(input)

        expect(output2.shape).toEqual([2, 32, 4, 4])
      })
    })

    test('works without affine transformation', () => {
      run(() => {
        const gn = new GroupNorm(4, 16, { affine: false })
        const input = cpu.randn([2, 16, 8, 8] as const)

        const output = gn.forward(input)

        expect(output.shape).toEqual([2, 16, 8, 8])
      })
    })
  })

  describe('training mode', () => {
    test('is in training mode by default', () => {
      const gn = new GroupNorm(8, 32)

      expect(gn.training).toBe(true)
    })

    test('can switch to eval mode', () => {
      const gn = new GroupNorm(8, 32)

      gn.eval()

      expect(gn.training).toBe(false)
    })
  })

  describe('toString', () => {
    test('returns descriptive string', () => {
      const gn = new GroupNorm(8, 32, { eps: 1e-6 })
      const str = gn.toString()

      expect(str).toContain('GroupNorm')
      expect(str).toContain('8')
      expect(str).toContain('32')
    })
  })
})

describe('InstanceNorm2d', () => {
  describe('constructor', () => {
    test('creates InstanceNorm2d with correct parameters', () => {
      const norm = new InstanceNorm2d(64)

      expect(norm.numFeatures).toBe(64)
      expect(norm.eps).toBe(1e-5)
      expect(norm.affine).toBe(false) // Default is false for InstanceNorm
    })

    test('can enable affine parameters', () => {
      const norm = new InstanceNorm2d(64, { affine: true })

      expect(norm.affine).toBe(true)
      expect(norm.weight).not.toBeNull()
      expect(norm.biasParam).not.toBeNull()
    })

    test('accepts custom options', () => {
      const norm = new InstanceNorm2d(64, {
        eps: 1e-6,
        momentum: 0.2,
        affine: true,
      })

      expect(norm.eps).toBe(1e-6)
      expect(norm.momentum).toBe(0.2)
    })
  })

  describe('forward pass', () => {
    test('normalizes 4D input', () => {
      run(() => {
        const norm = new InstanceNorm2d(64)
        const input = cpu.randn([4, 64, 28, 28] as const)

        const output = norm.forward(input)

        expect(output.shape).toEqual([4, 64, 28, 28])
      })
    })

    test('normalizes each instance independently', () => {
      run(() => {
        const norm = new InstanceNorm2d(16)
        const input = cpu.randn([2, 16, 8, 8] as const)

        const output = norm.forward(input)

        expect(output.shape).toEqual([2, 16, 8, 8])
      })
    })

    test('applies affine transformation when enabled', () => {
      run(() => {
        const norm = new InstanceNorm2d(16, { affine: true })
        const input = cpu.randn([2, 16, 8, 8] as const)

        const output = norm.forward(input)

        expect(output.shape).toEqual([2, 16, 8, 8])
      })
    })
  })

  describe('toString', () => {
    test('returns descriptive string', () => {
      const norm = new InstanceNorm2d(64, { affine: true })
      const str = norm.toString()

      expect(str).toContain('InstanceNorm2d')
      expect(str).toContain('64')
    })
  })
})

describe('LayerNorm', () => {
  describe('constructor', () => {
    test('creates LayerNorm with single dimension', () => {
      const ln = new LayerNorm(768)

      expect(ln.normalizedShape).toEqual([768])
    })

    test('creates LayerNorm with multiple dimensions', () => {
      const ln = new LayerNorm([32, 32])

      expect(ln.normalizedShape).toEqual([32, 32])
    })

    test('creates affine parameters by default', () => {
      const ln = new LayerNorm(768)

      expect(ln.weight).not.toBeNull()
      expect(ln.biasParam).not.toBeNull()
    })

    test('can disable affine', () => {
      const ln = new LayerNorm(768, { elementwiseAffine: false })

      expect(ln.weight).toBeNull()
      expect(ln.biasParam).toBeNull()
    })
  })

  describe('forward pass', () => {
    test('normalizes 2D input', () => {
      run(() => {
        const ln = new LayerNorm(768)
        const input = cpu.randn([32, 768] as const)

        const output = ln.forward(input)

        expect(output.shape).toEqual([32, 768])
      })
    })

    test('normalizes 3D input (transformer-style)', () => {
      run(() => {
        const ln = new LayerNorm(512)
        const input = cpu.randn([16, 128, 512] as const)

        const output = ln.forward(input)

        expect(output.shape).toEqual([16, 128, 512])
      })
    })
  })
})

describe('BatchNorm2d', () => {
  describe('constructor', () => {
    test('creates BatchNorm2d with correct parameters', () => {
      const bn = new BatchNorm2d(64)

      expect(bn.numFeatures).toBe(64)
      expect(bn.eps).toBe(1e-5)
      expect(bn.momentum).toBe(0.1)
      expect(bn.affine).toBe(true)
    })

    test('creates weight and bias parameters', () => {
      const bn = new BatchNorm2d(64)

      expect(bn.weight).not.toBeNull()
      expect(bn.biasParam).not.toBeNull()
    })

    test('can disable affine', () => {
      const bn = new BatchNorm2d(64, { affine: false })

      expect(bn.weight).toBeNull()
      expect(bn.biasParam).toBeNull()
    })
  })

  describe('forward pass', () => {
    test('normalizes 4D input', () => {
      run(() => {
        const bn = new BatchNorm2d(64)
        const input = cpu.randn([32, 64, 28, 28] as const)

        const output = bn.forward(input)

        expect(output.shape).toEqual([32, 64, 28, 28])
      })
    })
  })
})

describe('BatchNorm1d', () => {
  describe('constructor', () => {
    test('creates BatchNorm1d with correct parameters', () => {
      const bn = new BatchNorm1d(128)

      expect(bn.numFeatures).toBe(128)
    })
  })

  describe('forward pass', () => {
    test('normalizes 3D input', () => {
      run(() => {
        const bn = new BatchNorm1d(128)
        const input = cpu.randn([32, 128, 50] as const)

        const output = bn.forward(input)

        expect(output.shape).toEqual([32, 128, 50])
      })
    })
  })
})
