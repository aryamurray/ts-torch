/**
 * Tests for Transformer modules
 */

import { describe, test, expect } from 'vitest'
import {
  TransformerEncoderLayer,
  TransformerEncoder,
  TransformerDecoderLayer,
  TransformerDecoder,
  generateSquareSubsequentMask,
} from '../transformer.js'
import { LayerNorm } from '../normalization.js'
import { device, run } from '@ts-torch/core'

const cpu = device.cpu()

describe('TransformerEncoderLayer', () => {
  describe('constructor', () => {
    test('creates layer with correct dimensions', () => {
      const layer = new TransformerEncoderLayer(512, 8)

      expect(layer.dModel).toBe(512)
      expect(layer.nHead).toBe(8)
      expect(layer.dimFeedforward).toBe(2048)
      expect(layer.dropoutP).toBe(0.1)
      expect(layer.activation).toBe('relu')
    })

    test('accepts custom options', () => {
      const layer = new TransformerEncoderLayer(256, 4, {
        dimFeedforward: 1024,
        dropout: 0.2,
        activation: 'gelu',
        batchFirst: true,
        normFirst: true,
      })

      expect(layer.dModel).toBe(256)
      expect(layer.nHead).toBe(4)
      expect(layer.dimFeedforward).toBe(1024)
      expect(layer.dropoutP).toBe(0.2)
      expect(layer.activation).toBe('gelu')
      expect(layer.batchFirst).toBe(true)
      expect(layer.normFirst).toBe(true)
    })

    test('throws error for invalid dModel', () => {
      expect(() => new TransformerEncoderLayer(0, 8)).toThrow('must be positive')
      expect(() => new TransformerEncoderLayer(-512, 8)).toThrow('must be positive')
    })

    test('throws error for invalid nHead', () => {
      expect(() => new TransformerEncoderLayer(512, 0)).toThrow('must be positive')
      expect(() => new TransformerEncoderLayer(512, -4)).toThrow('must be positive')
    })

    test('throws error when dModel not divisible by nHead', () => {
      expect(() => new TransformerEncoderLayer(512, 7)).toThrow('must be divisible by')
    })
  })

  describe('forward pass', () => {
    test('processes sequence with seq-first format', () => {
      run(() => {
        const layer = new TransformerEncoderLayer(64, 4)
        const x = cpu.randn([10, 2, 64] as const) // [seq, batch, embed]

        const output = layer.forward(x)

        expect(output.shape).toEqual([10, 2, 64])
      })
    })

    test('processes sequence with batch-first format', () => {
      run(() => {
        const layer = new TransformerEncoderLayer(64, 4, { batchFirst: true })
        const x = cpu.randn([2, 10, 64] as const) // [batch, seq, embed]

        const output = layer.forward(x)

        expect(output.shape).toEqual([2, 10, 64])
      })
    })

    test('works with gelu activation', () => {
      run(() => {
        const layer = new TransformerEncoderLayer(64, 4, { activation: 'gelu' })
        const x = cpu.randn([10, 2, 64] as const)

        const output = layer.forward(x)

        expect(output.shape).toEqual([10, 2, 64])
      })
    })

    test('works with pre-norm', () => {
      run(() => {
        const layer = new TransformerEncoderLayer(64, 4, { normFirst: true })
        const x = cpu.randn([10, 2, 64] as const)

        const output = layer.forward(x)

        expect(output.shape).toEqual([10, 2, 64])
      })
    })
  })

  describe('parameters', () => {
    test('has trainable parameters', () => {
      const layer = new TransformerEncoderLayer(64, 4)
      const params = layer.parameters()

      expect(params.length).toBeGreaterThan(0)
    })

    test('all parameters require gradient', () => {
      const layer = new TransformerEncoderLayer(64, 4)
      const params = layer.parameters()

      for (const param of params) {
        expect(param.requiresGrad).toBe(true)
      }
    })
  })

  describe('training mode', () => {
    test('is in training mode by default', () => {
      const layer = new TransformerEncoderLayer(64, 4)

      expect(layer.training).toBe(true)
    })

    test('can switch to eval mode', () => {
      const layer = new TransformerEncoderLayer(64, 4)

      layer.eval()

      expect(layer.training).toBe(false)
    })
  })

  describe('toString', () => {
    test('returns descriptive string', () => {
      const layer = new TransformerEncoderLayer(512, 8)
      const str = layer.toString()

      expect(str).toContain('TransformerEncoderLayer')
      expect(str).toContain('d_model=512')
      expect(str).toContain('nhead=8')
    })
  })
})

describe('TransformerEncoder', () => {
  describe('constructor', () => {
    test('creates encoder with correct number of layers', () => {
      const encoderLayer = new TransformerEncoderLayer(64, 4)
      const encoder = new TransformerEncoder(encoderLayer, 6)

      expect(encoder.numLayers).toBe(6)
    })

    test('throws error for invalid numLayers', () => {
      const encoderLayer = new TransformerEncoderLayer(64, 4)

      expect(() => new TransformerEncoder(encoderLayer, 0)).toThrow('must be positive')
      expect(() => new TransformerEncoder(encoderLayer, -1)).toThrow('must be positive')
    })

    test('accepts optional norm', () => {
      const encoderLayer = new TransformerEncoderLayer(64, 4)
      const norm = new LayerNorm([64])
      const encoder = new TransformerEncoder(encoderLayer, 6, { norm })

      expect(encoder.numLayers).toBe(6)
    })
  })

  describe('forward pass', () => {
    test('processes sequence through all layers', () => {
      run(() => {
        const encoderLayer = new TransformerEncoderLayer(64, 4)
        const encoder = new TransformerEncoder(encoderLayer, 3)
        const x = cpu.randn([10, 2, 64] as const)

        const output = encoder.forward(x)

        expect(output.shape).toEqual([10, 2, 64])
      })
    })

    test('works with batch-first format', () => {
      run(() => {
        const encoderLayer = new TransformerEncoderLayer(64, 4, { batchFirst: true })
        const encoder = new TransformerEncoder(encoderLayer, 3)
        const x = cpu.randn([2, 10, 64] as const)

        const output = encoder.forward(x)

        expect(output.shape).toEqual([2, 10, 64])
      })
    })

    test('applies final norm when provided', () => {
      run(() => {
        const encoderLayer = new TransformerEncoderLayer(64, 4)
        const norm = new LayerNorm([64])
        const encoder = new TransformerEncoder(encoderLayer, 3, { norm })
        const x = cpu.randn([10, 2, 64] as const)

        const output = encoder.forward(x)

        expect(output.shape).toEqual([10, 2, 64])
      })
    })
  })

  describe('parameters', () => {
    test('has parameters from all layers', () => {
      const encoderLayer = new TransformerEncoderLayer(64, 4)
      const encoder = new TransformerEncoder(encoderLayer, 3)
      const params = encoder.parameters()

      // Each layer has: self_attn (in_proj, out_proj), linear1, linear2, norm1, norm2
      // Multiple parameters per submodule
      expect(params.length).toBeGreaterThan(0)
    })
  })

  describe('toString', () => {
    test('returns descriptive string', () => {
      const encoderLayer = new TransformerEncoderLayer(64, 4)
      const encoder = new TransformerEncoder(encoderLayer, 6)
      const str = encoder.toString()

      expect(str).toContain('TransformerEncoder')
      expect(str).toContain('num_layers=6')
    })
  })
})

describe('TransformerDecoderLayer', () => {
  describe('constructor', () => {
    test('creates layer with correct dimensions', () => {
      const layer = new TransformerDecoderLayer(512, 8)

      expect(layer.dModel).toBe(512)
      expect(layer.nHead).toBe(8)
      expect(layer.dimFeedforward).toBe(2048)
      expect(layer.dropoutP).toBe(0.1)
      expect(layer.activation).toBe('relu')
    })

    test('accepts custom options', () => {
      const layer = new TransformerDecoderLayer(256, 4, {
        dimFeedforward: 1024,
        dropout: 0.2,
        activation: 'gelu',
        batchFirst: true,
        normFirst: true,
      })

      expect(layer.dModel).toBe(256)
      expect(layer.nHead).toBe(4)
      expect(layer.dimFeedforward).toBe(1024)
      expect(layer.dropoutP).toBe(0.2)
      expect(layer.activation).toBe('gelu')
      expect(layer.batchFirst).toBe(true)
      expect(layer.normFirst).toBe(true)
    })

    test('throws error for invalid dModel', () => {
      expect(() => new TransformerDecoderLayer(0, 8)).toThrow('must be positive')
    })

    test('throws error for invalid nHead', () => {
      expect(() => new TransformerDecoderLayer(512, 0)).toThrow('must be positive')
    })

    test('throws error when dModel not divisible by nHead', () => {
      expect(() => new TransformerDecoderLayer(512, 7)).toThrow('must be divisible by')
    })
  })

  describe('forward pass', () => {
    test('processes target and memory with seq-first format', () => {
      run(() => {
        const layer = new TransformerDecoderLayer(64, 4)
        const tgt = cpu.randn([10, 2, 64] as const) // [tgt_seq, batch, embed]
        const memory = cpu.randn([20, 2, 64] as const) // [src_seq, batch, embed]

        const output = layer.forward(tgt, memory)

        expect(output.shape).toEqual([10, 2, 64])
      })
    })

    test('processes target and memory with batch-first format', () => {
      run(() => {
        const layer = new TransformerDecoderLayer(64, 4, { batchFirst: true })
        const tgt = cpu.randn([2, 10, 64] as const) // [batch, tgt_seq, embed]
        const memory = cpu.randn([2, 20, 64] as const) // [batch, src_seq, embed]

        const output = layer.forward(tgt, memory)

        expect(output.shape).toEqual([2, 10, 64])
      })
    })

    test('works with gelu activation', () => {
      run(() => {
        const layer = new TransformerDecoderLayer(64, 4, { activation: 'gelu' })
        const tgt = cpu.randn([10, 2, 64] as const)
        const memory = cpu.randn([20, 2, 64] as const)

        const output = layer.forward(tgt, memory)

        expect(output.shape).toEqual([10, 2, 64])
      })
    })

    test('works with pre-norm', () => {
      run(() => {
        const layer = new TransformerDecoderLayer(64, 4, { normFirst: true })
        const tgt = cpu.randn([10, 2, 64] as const)
        const memory = cpu.randn([20, 2, 64] as const)

        const output = layer.forward(tgt, memory)

        expect(output.shape).toEqual([10, 2, 64])
      })
    })
  })

  describe('parameters', () => {
    test('has trainable parameters', () => {
      const layer = new TransformerDecoderLayer(64, 4)
      const params = layer.parameters()

      expect(params.length).toBeGreaterThan(0)
    })
  })

  describe('toString', () => {
    test('returns descriptive string', () => {
      const layer = new TransformerDecoderLayer(512, 8)
      const str = layer.toString()

      expect(str).toContain('TransformerDecoderLayer')
      expect(str).toContain('d_model=512')
      expect(str).toContain('nhead=8')
    })
  })
})

describe('TransformerDecoder', () => {
  describe('constructor', () => {
    test('creates decoder with correct number of layers', () => {
      const decoderLayer = new TransformerDecoderLayer(64, 4)
      const decoder = new TransformerDecoder(decoderLayer, 6)

      expect(decoder.numLayers).toBe(6)
    })

    test('throws error for invalid numLayers', () => {
      const decoderLayer = new TransformerDecoderLayer(64, 4)

      expect(() => new TransformerDecoder(decoderLayer, 0)).toThrow('must be positive')
    })

    test('accepts optional norm', () => {
      const decoderLayer = new TransformerDecoderLayer(64, 4)
      const norm = new LayerNorm([64])
      const decoder = new TransformerDecoder(decoderLayer, 6, { norm })

      expect(decoder.numLayers).toBe(6)
    })
  })

  describe('forward pass', () => {
    test('processes sequence through all layers', () => {
      run(() => {
        const decoderLayer = new TransformerDecoderLayer(64, 4)
        const decoder = new TransformerDecoder(decoderLayer, 3)
        const tgt = cpu.randn([10, 2, 64] as const)
        const memory = cpu.randn([20, 2, 64] as const)

        const output = decoder.forward(tgt, memory)

        expect(output.shape).toEqual([10, 2, 64])
      })
    })

    test('works with batch-first format', () => {
      run(() => {
        const decoderLayer = new TransformerDecoderLayer(64, 4, { batchFirst: true })
        const decoder = new TransformerDecoder(decoderLayer, 3)
        const tgt = cpu.randn([2, 10, 64] as const)
        const memory = cpu.randn([2, 20, 64] as const)

        const output = decoder.forward(tgt, memory)

        expect(output.shape).toEqual([2, 10, 64])
      })
    })

    test('applies final norm when provided', () => {
      run(() => {
        const decoderLayer = new TransformerDecoderLayer(64, 4)
        const norm = new LayerNorm([64])
        const decoder = new TransformerDecoder(decoderLayer, 3, { norm })
        const tgt = cpu.randn([10, 2, 64] as const)
        const memory = cpu.randn([20, 2, 64] as const)

        const output = decoder.forward(tgt, memory)

        expect(output.shape).toEqual([10, 2, 64])
      })
    })
  })

  describe('toString', () => {
    test('returns descriptive string', () => {
      const decoderLayer = new TransformerDecoderLayer(64, 4)
      const decoder = new TransformerDecoder(decoderLayer, 6)
      const str = decoder.toString()

      expect(str).toContain('TransformerDecoder')
      expect(str).toContain('num_layers=6')
    })
  })
})

describe('generateSquareSubsequentMask', () => {
  test('creates correct shape mask', () => {
    run(() => {
      const mask = generateSquareSubsequentMask(10)

      expect(mask.shape).toEqual([10, 10])
    })
  })

  test('creates mask for small sequence', () => {
    run(() => {
      const mask = generateSquareSubsequentMask(4)

      expect(mask.shape).toEqual([4, 4])
    })
  })
})

describe('Transformer end-to-end', () => {
  test('encoder-decoder architecture works together', () => {
    run(() => {
      // Create encoder
      const encoderLayer = new TransformerEncoderLayer(64, 4, { batchFirst: true })
      const encoder = new TransformerEncoder(encoderLayer, 2)

      // Create decoder
      const decoderLayer = new TransformerDecoderLayer(64, 4, { batchFirst: true })
      const decoder = new TransformerDecoder(decoderLayer, 2)

      // Input data
      const src = cpu.randn([2, 20, 64] as const) // [batch, src_seq, embed]
      const tgt = cpu.randn([2, 10, 64] as const) // [batch, tgt_seq, embed]

      // Encode
      const memory = encoder.forward(src)
      expect(memory.shape).toEqual([2, 20, 64])

      // Decode
      const output = decoder.forward(tgt, memory)
      expect(output.shape).toEqual([2, 10, 64])
    })
  })
})
