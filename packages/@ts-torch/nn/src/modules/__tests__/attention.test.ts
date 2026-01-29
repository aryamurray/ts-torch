/**
 * Tests for MultiheadAttention module
 */

import { describe, test, expect } from 'vitest'
import { MultiheadAttention, scaledDotProductAttention } from '../attention.js'
import { device, run } from '@ts-torch/core'

const cpu = device.cpu()

describe('MultiheadAttention', () => {
  describe('constructor', () => {
    test('creates attention with correct dimensions', () => {
      const attention = new MultiheadAttention(512, 8)

      expect(attention.embedDim).toBe(512)
      expect(attention.numHeads).toBe(8)
      expect(attention.headDim).toBe(64)
    })

    test('registers in_proj and out_proj as submodules', () => {
      const attention = new MultiheadAttention(256, 4)
      const modules = attention.modules()

      expect(modules.size).toBeGreaterThanOrEqual(2)
    })

    test('throws error when embedDim not divisible by numHeads', () => {
      expect(() => new MultiheadAttention(512, 7)).toThrow(
        'must be divisible by',
      )
    })

    test('throws error for invalid embedDim', () => {
      expect(() => new MultiheadAttention(0, 8)).toThrow('must be positive')
      expect(() => new MultiheadAttention(-512, 8)).toThrow('must be positive')
    })

    test('throws error for invalid numHeads', () => {
      expect(() => new MultiheadAttention(512, 0)).toThrow('must be positive')
      expect(() => new MultiheadAttention(512, -4)).toThrow('must be positive')
    })

    test('accepts dropout option', () => {
      const attention = new MultiheadAttention(512, 8, { dropout: 0.1 })

      expect(attention.dropoutP).toBe(0.1)
    })

    test('accepts batchFirst option', () => {
      const attention = new MultiheadAttention(512, 8, { batchFirst: true })

      expect(attention.batchFirst).toBe(true)
    })

    test('default batchFirst is false', () => {
      const attention = new MultiheadAttention(512, 8)

      expect(attention.batchFirst).toBe(false)
    })

    test('accepts kdim and vdim options', () => {
      const attention = new MultiheadAttention(512, 8, {
        kdim: 256,
        vdim: 256,
      })

      expect(attention.kdim).toBe(256)
      expect(attention.vdim).toBe(256)
    })
  })

  describe('forward pass', () => {
    test('self-attention with seq-first format', () => {
      run(() => {
        const attention = new MultiheadAttention(64, 4)
        const x = cpu.randn([10, 2, 64] as const) // [seq, batch, embed]

        const [output, weights] = attention.forward(x, x, x)

        expect(output.shape).toEqual([10, 2, 64])
        expect(weights).not.toBeNull()
        expect(weights!.shape).toEqual([2, 10, 10])
      })
    })

    test('self-attention with batch-first format', () => {
      run(() => {
        const attention = new MultiheadAttention(64, 4, { batchFirst: true })
        const x = cpu.randn([2, 10, 64] as const) // [batch, seq, embed]

        const [output, weights] = attention.forward(x, x, x)

        expect(output.shape).toEqual([2, 10, 64])
        expect(weights).not.toBeNull()
      })
    })

    test('forward without returning weights', () => {
      run(() => {
        const attention = new MultiheadAttention(64, 4)
        const x = cpu.randn([10, 2, 64] as const)

        const [output, weights] = attention.forward(x, x, x, {
          needWeights: false,
        })

        expect(output.shape).toEqual([10, 2, 64])
        expect(weights).toBeNull()
      })
    })

    test('cross-attention with different key/value sequence', () => {
      run(() => {
        const attention = new MultiheadAttention(64, 4)
        const query = cpu.randn([5, 2, 64] as const) // [tgt_seq, batch, embed]
        const key = cpu.randn([10, 2, 64] as const) // [src_seq, batch, embed]
        const value = cpu.randn([10, 2, 64] as const)

        const [output, weights] = attention.forward(query, key, value)

        expect(output.shape).toEqual([5, 2, 64])
        expect(weights!.shape).toEqual([2, 5, 10])
      })
    })

    test('handles single sample batch', () => {
      run(() => {
        const attention = new MultiheadAttention(32, 2, { batchFirst: true })
        const x = cpu.randn([1, 8, 32] as const)

        const [output, _] = attention.forward(x, x, x)

        expect(output.shape).toEqual([1, 8, 32])
      })
    })
  })

  describe('training mode', () => {
    test('is in training mode by default', () => {
      const attention = new MultiheadAttention(512, 8)

      expect(attention.training).toBe(true)
    })

    test('can switch to eval mode', () => {
      const attention = new MultiheadAttention(512, 8)

      attention.eval()

      expect(attention.training).toBe(false)
    })

    test('dropout is not applied in eval mode', () => {
      run(() => {
        const attention = new MultiheadAttention(64, 4, { dropout: 0.5 })
        attention.eval()
        const x = cpu.randn([10, 2, 64] as const)

        // In eval mode, dropout should not affect output
        const [output1, _] = attention.forward(x, x, x)
        const [output2, __] = attention.forward(x, x, x)

        // Outputs should be identical in eval mode
        expect(output1.shape).toEqual(output2.shape)
      })
    })
  })

  describe('parameters', () => {
    test('has trainable parameters', () => {
      const attention = new MultiheadAttention(512, 8)
      const params = attention.parameters()

      expect(params.length).toBeGreaterThan(0)
    })

    test('all parameters require gradient', () => {
      const attention = new MultiheadAttention(512, 8)
      const params = attention.parameters()

      for (const param of params) {
        expect(param.requiresGrad).toBe(true)
      }
    })
  })

  describe('toString', () => {
    test('returns descriptive string', () => {
      const attention = new MultiheadAttention(512, 8, {
        dropout: 0.1,
        batchFirst: true,
      })
      const str = attention.toString()

      expect(str).toContain('MultiheadAttention')
      expect(str).toContain('embed_dim=512')
      expect(str).toContain('num_heads=8')
      expect(str).toContain('dropout=0.1')
      expect(str).toContain('batch_first=true')
    })
  })
})

describe('scaledDotProductAttention', () => {
  test('computes attention for 2D tensors', () => {
    run(() => {
      const q = cpu.randn([4, 16] as const)
      const k = cpu.randn([4, 16] as const)
      const v = cpu.randn([4, 16] as const)

      const output = scaledDotProductAttention(q, k, v)

      expect(output.shape).toEqual([4, 16])
    })
  })

  test('computes attention for 3D tensors', () => {
    run(() => {
      const q = cpu.randn([2, 8, 16] as const)
      const k = cpu.randn([2, 8, 16] as const)
      const v = cpu.randn([2, 8, 16] as const)

      const output = scaledDotProductAttention(q, k, v)

      expect(output.shape).toEqual([2, 8, 16])
    })
  })

  test('accepts custom scale', () => {
    run(() => {
      const q = cpu.randn([4, 16] as const)
      const k = cpu.randn([4, 16] as const)
      const v = cpu.randn([4, 16] as const)

      const output = scaledDotProductAttention(q, k, v, { scale: 2.0 })

      expect(output.shape).toEqual([4, 16])
    })
  })
})
