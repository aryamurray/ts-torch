/**
 * Tests for Embedding layer
 */

import { describe, test, expect } from 'vitest'
import { Embedding, embeddingFromPretrained } from '../embedding.js'
import { Linear } from '../linear.js'
import { device, run, int64 } from '@ts-torch/core'

const cpu = device.cpu()

describe('Embedding', () => {
  describe('constructor', () => {
    test('creates embedding with correct dimensions', () => {
      const embedding = new Embedding(1000, 128)

      expect(embedding.numEmbeddings).toBe(1000)
      expect(embedding.embeddingDim).toBe(128)
      expect(embedding.weight).toBeDefined()
      expect(embedding.weight.data.shape).toEqual([1000, 128])
    })

    test('weight requires gradient by default', () => {
      const embedding = new Embedding(100, 64)

      expect(embedding.weight.requiresGrad).toBe(true)
    })

    test('registers weight parameter', () => {
      const embedding = new Embedding(100, 64)
      const params = embedding.parameters()

      expect(params).toHaveLength(1)
      expect(params).toContain(embedding.weight)
    })

    test('named parameters use correct name', () => {
      const embedding = new Embedding(100, 64)
      const namedParams = embedding.namedParameters()

      expect(namedParams.get('weight')).toBe(embedding.weight)
    })

    test('throws error for invalid numEmbeddings', () => {
      expect(() => new Embedding(0, 64)).toThrow('numEmbeddings must be positive')
      expect(() => new Embedding(-10, 64)).toThrow('numEmbeddings must be positive')
    })

    test('throws error for invalid embeddingDim', () => {
      expect(() => new Embedding(100, 0)).toThrow('embeddingDim must be positive')
      expect(() => new Embedding(100, -10)).toThrow('embeddingDim must be positive')
    })

    test('creates embedding with paddingIdx', () => {
      const embedding = new Embedding(100, 64, { paddingIdx: 0 })

      expect(embedding.paddingIdx).toBe(0)
    })

    test('validates paddingIdx range', () => {
      expect(() => new Embedding(100, 64, { paddingIdx: 100 })).toThrow()
      expect(() => new Embedding(100, 64, { paddingIdx: -101 })).toThrow()
    })

    test('allows negative paddingIdx within valid range', () => {
      const embedding = new Embedding(100, 64, { paddingIdx: -1 })

      expect(embedding.paddingIdx).toBe(-1)
    })

    test('stores maxNorm option', () => {
      const embedding = new Embedding(100, 64, { maxNorm: 1.0 })

      expect(embedding.maxNorm).toBe(1.0)
    })

    test('stores normType option', () => {
      const embedding = new Embedding(100, 64, { normType: 1 })

      expect(embedding.normType).toBe(1)
    })

    test('default normType is 2', () => {
      const embedding = new Embedding(100, 64)

      expect(embedding.normType).toBe(2)
    })
  })

  describe('forward pass', () => {
    test('embeds 1D input correctly', () => {
      run(() => {
        const embedding = new Embedding(100, 16)
        const input = cpu.tensor([0, 1, 2, 3], [4] as const, int64)
        const output = embedding.forward(input)

        expect(output.shape).toEqual([4, 16])
      })
    })

    test('embeds 2D input correctly (batch, seq)', () => {
      run(() => {
        const embedding = new Embedding(1000, 128)
        const input = cpu.tensor(
          [0, 1, 2, 3, 4, 5, 6, 7],
          [2, 4] as const,
          int64,
        )
        const output = embedding.forward(input)

        expect(output.shape).toEqual([2, 4, 128])
      })
    })

    test('handles single token', () => {
      run(() => {
        const embedding = new Embedding(100, 64)
        const input = cpu.tensor([5], [1] as const, int64)
        const output = embedding.forward(input)

        expect(output.shape).toEqual([1, 64])
      })
    })

    test('handles large vocabulary', () => {
      run(() => {
        const embedding = new Embedding(50000, 256)
        const input = cpu.tensor([49999, 0, 12345], [3] as const, int64)
        const output = embedding.forward(input)

        expect(output.shape).toEqual([3, 256])
      })
    })
  })

  describe('training mode', () => {
    test('is in training mode by default', () => {
      const embedding = new Embedding(100, 64)

      expect(embedding.training).toBe(true)
    })

    test('can switch to eval mode', () => {
      const embedding = new Embedding(100, 64)

      embedding.eval()

      expect(embedding.training).toBe(false)
    })

    test('can switch back to training mode', () => {
      const embedding = new Embedding(100, 64)

      embedding.eval()
      embedding.train()

      expect(embedding.training).toBe(true)
    })
  })

  describe('toString', () => {
    test('returns descriptive string', () => {
      const embedding = new Embedding(1000, 128)
      const str = embedding.toString()

      expect(str).toContain('Embedding')
      expect(str).toContain('1000')
      expect(str).toContain('128')
    })

    test('includes padding_idx when specified', () => {
      const embedding = new Embedding(1000, 128, { paddingIdx: 0 })
      const str = embedding.toString()

      expect(str).toContain('padding_idx=0')
    })
  })

  describe('composition', () => {
    test('can be piped with other modules', () => {
      run(() => {
        const embedding = new Embedding(100, 64)
        const linear = new Linear(64, 10)

        const piped = embedding.pipe(linear)

        expect(piped).toBeDefined()
      })
    })
  })
})

describe('embeddingFromPretrained', () => {
  test('creates embedding from pretrained weights', () => {
    run(() => {
      const pretrainedWeights = cpu.randn([100, 64] as const)
      const embedding = embeddingFromPretrained(pretrainedWeights)

      expect(embedding.numEmbeddings).toBe(100)
      expect(embedding.embeddingDim).toBe(64)
    })
  })

  test('freezes weights by default', () => {
    run(() => {
      const pretrainedWeights = cpu.randn([100, 64] as const)
      const embedding = embeddingFromPretrained(pretrainedWeights)

      expect(embedding.weight.requiresGrad).toBe(false)
    })
  })

  test('allows fine-tuning when freeze is false', () => {
    run(() => {
      const pretrainedWeights = cpu.randn([100, 64] as const)
      const embedding = embeddingFromPretrained(pretrainedWeights, {
        freeze: false,
      })

      expect(embedding.weight.requiresGrad).toBe(true)
    })
  })

  test('preserves pretrained options', () => {
    run(() => {
      const pretrainedWeights = cpu.randn([100, 64] as const)
      const embedding = embeddingFromPretrained(pretrainedWeights, {
        paddingIdx: 0,
        maxNorm: 2.0,
      })

      expect(embedding.paddingIdx).toBe(0)
      expect(embedding.maxNorm).toBe(2.0)
    })
  })
})
