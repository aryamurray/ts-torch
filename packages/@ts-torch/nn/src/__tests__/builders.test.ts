/**
 * Tests for config serialization: toJSON / fromJSON
 */

import { describe, test, expect } from 'vitest'
import { nn } from '../builders.js'

describe('Config serialization', () => {
  describe('toJSON', () => {
    test('serializes basic sequence config', () => {
      const config = nn.sequence(
        nn.input(784),
        nn.fc(128).relu(),
        nn.fc(10),
      )

      const json = config.toJSON() as any

      expect(json.format).toBe('ts-torch-sequence')
      expect(json.version).toBe(1)
      expect(json.input).toEqual({ shape: [784] })
      expect(json.blocks).toHaveLength(2)
      expect(json.blocks[0].outFeatures).toBe(128)
      expect(json.blocks[0].activation).toBe('relu')
      expect(json.blocks[1].outFeatures).toBe(10)
      expect(json.blocks[1].activation).toBeUndefined()
    })

    test('only includes non-default fields', () => {
      const config = nn.sequence(
        nn.input(10),
        nn.fc(5),
      )

      const json = config.toJSON() as any
      const block = json.blocks[0]

      // Default values should not be present
      expect(block.bias).toBeUndefined()
      expect(block.init).toBeUndefined()
      expect(block.dropoutP).toBeUndefined()
      expect(block.batchNorm).toBeUndefined()
    })

    test('serializes non-default bias and init', () => {
      const config = nn.sequence(
        nn.input(10),
        nn.fc(5).noBias().withInit('xavier_uniform'),
      )

      const json = config.toJSON() as any
      const block = json.blocks[0]

      expect(block.bias).toBe(false)
      expect(block.init).toBe('xavier_uniform')
    })

    test('serializes dropout', () => {
      const config = nn.sequence(
        nn.input(10),
        nn.fc(5).relu().dropout(0.5),
      )

      const json = config.toJSON() as any
      expect(json.blocks[0].dropoutP).toBe(0.5)
    })

    test('serializes batchNorm', () => {
      const config = nn.sequence(
        nn.input(10),
        nn.fc(5).batchNorm().relu(),
      )

      const json = config.toJSON() as any
      expect(json.blocks[0].batchNorm).toBe(true)
    })

    test('serializes leakyRelu with negativeSlope', () => {
      const config = nn.sequence(
        nn.input(10),
        nn.fc(5).leakyRelu(0.1),
      )

      const json = config.toJSON() as any
      expect(json.blocks[0].activation).toBe('leaky_relu')
      expect(json.blocks[0].negativeSlope).toBe(0.1)
    })

    test('serializes all activation types', () => {
      const activations = ['relu', 'gelu', 'sigmoid', 'tanh'] as const
      for (const act of activations) {
        const config = nn.sequence(
          nn.input(10),
          nn.fc(5)[act](),
        )
        const json = config.toJSON() as any
        expect(json.blocks[0].activation).toBe(act)
      }
    })
  })

  describe('fromJSON', () => {
    test('reconstructs config from JSON', () => {
      const original = nn.sequence(
        nn.input(784),
        nn.fc(128).relu(),
        nn.fc(64).gelu(),
        nn.fc(10),
      )

      const json = original.toJSON()
      const reconstructed = nn.fromJSON(json)

      expect(reconstructed.inputDef.shape).toEqual([784])
      expect(reconstructed.blocks).toHaveLength(3)
      expect(reconstructed.blocks[0]!.outFeatures).toBe(128)
      expect(reconstructed.blocks[0]!.activation).toBe('relu')
      expect(reconstructed.blocks[1]!.outFeatures).toBe(64)
      expect(reconstructed.blocks[1]!.activation).toBe('gelu')
      expect(reconstructed.blocks[2]!.outFeatures).toBe(10)
    })

    test('roundtrips toJSON -> fromJSON -> toJSON', () => {
      const config = nn.sequence(
        nn.input(784),
        nn.fc(128).relu().dropout(0.2).noBias(),
        nn.fc(64).gelu().batchNorm(),
        nn.fc(10).withInit('xavier_normal'),
      )

      const json1 = config.toJSON()
      const reconstructed = nn.fromJSON(json1)
      const json2 = reconstructed.toJSON()

      expect(json2).toEqual(json1)
    })

    test('rejects unknown format', () => {
      expect(() => nn.fromJSON({
        format: 'pytorch',
        version: 1,
        input: { shape: [10] },
        blocks: [{ outFeatures: 5 }],
      })).toThrow(/Unknown config format/)
    })

    test('rejects unsupported version', () => {
      expect(() => nn.fromJSON({
        format: 'ts-torch-sequence',
        version: 999,
        input: { shape: [10] },
        blocks: [{ outFeatures: 5 }],
      })).toThrow(/Unsupported config version/)
    })

    test('rejects invalid input shape', () => {
      expect(() => nn.fromJSON({
        format: 'ts-torch-sequence',
        version: 1,
        input: { shape: [] },
        blocks: [{ outFeatures: 5 }],
      })).toThrow(/Invalid input shape/)

      expect(() => nn.fromJSON({
        format: 'ts-torch-sequence',
        version: 1,
        input: { shape: [-1] },
        blocks: [{ outFeatures: 5 }],
      })).toThrow(/Invalid input shape/)
    })

    test('rejects negative outFeatures', () => {
      expect(() => nn.fromJSON({
        format: 'ts-torch-sequence',
        version: 1,
        input: { shape: [10] },
        blocks: [{ outFeatures: -5 }],
      })).toThrow(/outFeatures must be a positive integer/)
    })

    test('rejects unknown activation', () => {
      expect(() => nn.fromJSON({
        format: 'ts-torch-sequence',
        version: 1,
        input: { shape: [10] },
        blocks: [{ outFeatures: 5, activation: 'swish_magic' }],
      })).toThrow(/unknown activation/)
    })

    test('rejects unknown init strategy', () => {
      expect(() => nn.fromJSON({
        format: 'ts-torch-sequence',
        version: 1,
        input: { shape: [10] },
        blocks: [{ outFeatures: 5, init: 'random_magic' }],
      })).toThrow(/unknown init strategy/)
    })

    test('rejects invalid dropoutP', () => {
      expect(() => nn.fromJSON({
        format: 'ts-torch-sequence',
        version: 1,
        input: { shape: [10] },
        blocks: [{ outFeatures: 5, dropoutP: 1.5 }],
      })).toThrow(/dropoutP must be a number/)
    })

    test('rejects null config', () => {
      expect(() => nn.fromJSON(null)).toThrow()
    })

    test('rejects empty blocks', () => {
      expect(() => nn.fromJSON({
        format: 'ts-torch-sequence',
        version: 1,
        input: { shape: [10] },
        blocks: [],
      })).toThrow(/non-empty/)
    })

    test('ignores unknown fields (forward compat)', () => {
      const config = nn.fromJSON({
        format: 'ts-torch-sequence',
        version: 1,
        input: { shape: [10] },
        blocks: [{ outFeatures: 5, futureField: 'hello' }],
        futureTopLevel: true,
      })

      expect(config.blocks[0]!.outFeatures).toBe(5)
    })
  })
})
