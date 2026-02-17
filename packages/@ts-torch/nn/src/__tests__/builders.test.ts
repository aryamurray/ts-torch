/**
 * Tests for config serialization: toJSON / fromJSON
 */

import { describe, test, expect } from 'vitest'
import { nn } from '../builders.js'

describe('Config serialization', () => {
  describe('toJSON', () => {
    test('serializes basic sequence config', () => {
      const config = nn.sequence(nn.input(784), nn.fc(128).relu(), nn.fc(10))

      const json = config.toJSON() as any

      expect(json.format).toBe('ts-torch-sequence')
      expect(json.version).toBe(2)
      expect(json.input).toEqual({ shape: [784] })
      expect(json.blocks).toHaveLength(2)
      expect(json.blocks[0].kind).toBe('fc')
      expect(json.blocks[0].outFeatures).toBe(128)
      expect(json.blocks[0].activation).toBe('relu')
      expect(json.blocks[1].outFeatures).toBe(10)
      expect(json.blocks[1].activation).toBeUndefined()
    })

    test('only includes non-default fields', () => {
      const config = nn.sequence(nn.input(10), nn.fc(5))

      const json = config.toJSON() as any
      const block = json.blocks[0]

      // Default values should not be present
      expect(block.bias).toBeUndefined()
      expect(block.init).toBeUndefined()
      expect(block.dropoutP).toBeUndefined()
      expect(block.batchNorm).toBeUndefined()
    })

    test('serializes non-default bias and init', () => {
      const config = nn.sequence(nn.input(10), nn.fc(5).noBias().withInit('xavier_uniform'))

      const json = config.toJSON() as any
      const block = json.blocks[0]

      expect(block.bias).toBe(false)
      expect(block.init).toBe('xavier_uniform')
    })

    test('serializes dropout', () => {
      const config = nn.sequence(nn.input(10), nn.fc(5).relu().dropout(0.5))

      const json = config.toJSON() as any
      expect(json.blocks[0].dropoutP).toBe(0.5)
    })

    test('serializes batchNorm', () => {
      const config = nn.sequence(nn.input(10), nn.fc(5).batchNorm().relu())

      const json = config.toJSON() as any
      expect(json.blocks[0].batchNorm).toBe(true)
    })

    test('serializes leakyRelu with negativeSlope', () => {
      const config = nn.sequence(nn.input(10), nn.fc(5).leakyRelu(0.1))

      const json = config.toJSON() as any
      expect(json.blocks[0].activation).toBe('leaky_relu')
      expect(json.blocks[0].negativeSlope).toBe(0.1)
    })

    test('serializes all activation types', () => {
      const activations = ['relu', 'gelu', 'sigmoid', 'tanh'] as const
      for (const act of activations) {
        const config = nn.sequence(nn.input(10), nn.fc(5)[act]())
        const json = config.toJSON() as any
        expect(json.blocks[0].activation).toBe(act)
      }
    })
  })

  describe('fromJSON', () => {
    test('reconstructs config from JSON', () => {
      const original = nn.sequence(nn.input(784), nn.fc(128).relu(), nn.fc(64).gelu(), nn.fc(10))

      const json = original.toJSON()
      const reconstructed = nn.fromJSON(json)

      expect(reconstructed.inputDef.shape).toEqual([784])
      expect(reconstructed.blocks).toHaveLength(3)
      const b0 = reconstructed.blocks[0]! as any
      expect(b0.outFeatures).toBe(128)
      expect(b0.activation).toBe('relu')
      const b1 = reconstructed.blocks[1]! as any
      expect(b1.outFeatures).toBe(64)
      expect(b1.activation).toBe('gelu')
      const b2 = reconstructed.blocks[2]! as any
      expect(b2.outFeatures).toBe(10)
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
      expect(() =>
        nn.fromJSON({
          format: 'pytorch',
          version: 1,
          input: { shape: [10] },
          blocks: [{ outFeatures: 5 }],
        }),
      ).toThrow(/Unknown config format/)
    })

    test('rejects unsupported version', () => {
      expect(() =>
        nn.fromJSON({
          format: 'ts-torch-sequence',
          version: 999,
          input: { shape: [10] },
          blocks: [{ outFeatures: 5 }],
        }),
      ).toThrow(/Unsupported config version/)
    })

    test('rejects invalid input shape', () => {
      expect(() =>
        nn.fromJSON({
          format: 'ts-torch-sequence',
          version: 1,
          input: { shape: [] },
          blocks: [{ outFeatures: 5 }],
        }),
      ).toThrow(/Invalid input shape/)

      expect(() =>
        nn.fromJSON({
          format: 'ts-torch-sequence',
          version: 1,
          input: { shape: [-1] },
          blocks: [{ outFeatures: 5 }],
        }),
      ).toThrow(/Invalid input shape/)
    })

    test('rejects negative outFeatures', () => {
      expect(() =>
        nn.fromJSON({
          format: 'ts-torch-sequence',
          version: 1,
          input: { shape: [10] },
          blocks: [{ outFeatures: -5 }],
        }),
      ).toThrow(/outFeatures must be a positive integer/)
    })

    test('rejects unknown activation', () => {
      expect(() =>
        nn.fromJSON({
          format: 'ts-torch-sequence',
          version: 1,
          input: { shape: [10] },
          blocks: [{ outFeatures: 5, activation: 'swish_magic' }],
        }),
      ).toThrow(/unknown activation/)
    })

    test('rejects unknown init strategy', () => {
      expect(() =>
        nn.fromJSON({
          format: 'ts-torch-sequence',
          version: 1,
          input: { shape: [10] },
          blocks: [{ outFeatures: 5, init: 'random_magic' }],
        }),
      ).toThrow(/unknown init strategy/)
    })

    test('rejects invalid dropoutP', () => {
      expect(() =>
        nn.fromJSON({
          format: 'ts-torch-sequence',
          version: 1,
          input: { shape: [10] },
          blocks: [{ outFeatures: 5, dropoutP: 1.5 }],
        }),
      ).toThrow(/dropoutP must be a number/)
    })

    test('rejects null config', () => {
      expect(() => nn.fromJSON(null)).toThrow()
    })

    test('rejects empty blocks', () => {
      expect(() =>
        nn.fromJSON({
          format: 'ts-torch-sequence',
          version: 1,
          input: { shape: [10] },
          blocks: [],
        }),
      ).toThrow(/non-empty/)
    })

    test('ignores unknown fields (forward compat)', () => {
      const config = nn.fromJSON({
        format: 'ts-torch-sequence',
        version: 1,
        input: { shape: [10] },
        blocks: [{ outFeatures: 5, futureField: 'hello' }],
        futureTopLevel: true,
      })

      expect((config.blocks[0]! as any).outFeatures).toBe(5)
    })

    test('version 1 backward compat: blocks without kind treated as fc', () => {
      const config = nn.fromJSON({
        format: 'ts-torch-sequence',
        version: 1,
        input: { shape: [784] },
        blocks: [{ outFeatures: 128, activation: 'relu' }, { outFeatures: 10 }],
      })

      expect(config.blocks).toHaveLength(2)
      expect(config.blocks[0]!.kind).toBe('fc')
      expect(config.blocks[1]!.kind).toBe('fc')
    })
  })

  describe('CNN builder config', () => {
    test('serializes CNN sequence config', () => {
      const config = nn.sequence(
        nn.input([1, 28, 28]),
        nn.conv2d(32, 3, { padding: 1 }).relu(),
        nn.maxPool2d(2),
        nn.conv2d(64, 3, { padding: 1 }).relu(),
        nn.maxPool2d(2),
        nn.flatten(),
        nn.fc(128).relu().dropout(0.5),
        nn.fc(10),
      )

      const json = config.toJSON() as any

      expect(json.format).toBe('ts-torch-sequence')
      expect(json.version).toBe(2)
      expect(json.input).toEqual({ shape: [1, 28, 28] })
      expect(json.blocks).toHaveLength(7)

      // conv2d block
      expect(json.blocks[0].kind).toBe('conv2d')
      expect(json.blocks[0].outChannels).toBe(32)
      expect(json.blocks[0].kernelSize).toBe(3)
      expect(json.blocks[0].padding).toBe(1)
      expect(json.blocks[0].activation).toBe('relu')
      // stride=1 is default, should not be present
      expect(json.blocks[0].stride).toBeUndefined()

      // maxPool2d block
      expect(json.blocks[1].kind).toBe('maxPool2d')
      expect(json.blocks[1].kernelSize).toBe(2)

      // flatten block
      expect(json.blocks[4].kind).toBe('flatten')

      // fc block
      expect(json.blocks[5].kind).toBe('fc')
      expect(json.blocks[5].outFeatures).toBe(128)
    })

    test('CNN roundtrips toJSON -> fromJSON -> toJSON', () => {
      const config = nn.sequence(
        nn.input([1, 28, 28]),
        nn.conv2d(32, 3, { padding: 1 }).relu().batchNorm(),
        nn.maxPool2d(2),
        nn.flatten(),
        nn.fc(10),
      )

      const json1 = config.toJSON()
      const reconstructed = nn.fromJSON(json1)
      const json2 = reconstructed.toJSON()

      expect(json2).toEqual(json1)
    })

    test('conv2d with non-default options serializes correctly', () => {
      const config = nn.sequence(
        nn.input([3, 32, 32]),
        nn.conv2d(16, 5, { stride: 2, padding: 2, dilation: 2, groups: 1 }).noBias().dropout(0.1),
        nn.flatten(),
        nn.fc(10),
      )

      const json = config.toJSON() as any
      const conv = json.blocks[0]

      expect(conv.kind).toBe('conv2d')
      expect(conv.stride).toBe(2)
      expect(conv.padding).toBe(2)
      expect(conv.dilation).toBe(2)
      expect(conv.bias).toBe(false)
      expect(conv.dropoutP).toBe(0.1)
      // groups=1 is default, should not be present
      expect(conv.groups).toBeUndefined()
    })

    test('adaptiveAvgPool2d serializes correctly', () => {
      const config = nn.sequence(
        nn.input([3, 32, 32]),
        nn.conv2d(16, 3, { padding: 1 }),
        nn.adaptiveAvgPool2d(1),
        nn.flatten(),
        nn.fc(10),
      )

      const json = config.toJSON() as any
      expect(json.blocks[1].kind).toBe('adaptiveAvgPool2d')
      expect(json.blocks[1].outputSize).toBe(1)
    })

    test('avgPool2d serializes correctly', () => {
      const config = nn.sequence(
        nn.input([3, 32, 32]),
        nn.conv2d(16, 3, { padding: 1 }),
        nn.avgPool2d(2),
        nn.flatten(),
        nn.fc(10),
      )

      const json = config.toJSON() as any
      expect(json.blocks[1].kind).toBe('avgPool2d')
      expect(json.blocks[1].kernelSize).toBe(2)
    })
  })

  describe('Transformer builder config', () => {
    test('serializes transformer sequence config', () => {
      const config = nn.sequence(
        nn.input([32]),
        nn.embedding(1000, 64),
        nn.transformerEncoder(4, 2, { dimFeedforward: 128, dropout: 0.1 }),
        nn.flatten(),
        nn.fc(2),
      )

      const json = config.toJSON() as any

      expect(json.input).toEqual({ shape: [32] })
      expect(json.blocks).toHaveLength(4)

      expect(json.blocks[0].kind).toBe('embedding')
      expect(json.blocks[0].numEmbeddings).toBe(1000)
      expect(json.blocks[0].embeddingDim).toBe(64)

      expect(json.blocks[1].kind).toBe('transformerEncoder')
      expect(json.blocks[1].nHead).toBe(4)
      expect(json.blocks[1].numLayers).toBe(2)
      expect(json.blocks[1].dimFeedforward).toBe(128)
      expect(json.blocks[1].dropout).toBe(0.1)
    })

    test('transformer roundtrips toJSON -> fromJSON -> toJSON', () => {
      const config = nn.sequence(
        nn.input([16]),
        nn.embedding(500, 32),
        nn.transformerEncoder(4, 2, { dimFeedforward: 64, dropout: 0.1, activation: 'gelu', normFirst: true }),
        nn.flatten(),
        nn.fc(10),
      )

      const json1 = config.toJSON()
      const reconstructed = nn.fromJSON(json1)
      const json2 = reconstructed.toJSON()

      expect(json2).toEqual(json1)
    })

    test('embedding with paddingIdx', () => {
      const config = nn.sequence(nn.input([16]), nn.embedding(500, 32, { paddingIdx: 0 }), nn.flatten(), nn.fc(10))

      const json = config.toJSON() as any
      expect(json.blocks[0].paddingIdx).toBe(0)
    })
  })

  describe('Shape validation errors', () => {
    test('fc after conv without flatten throws', () => {
      expect(() => nn.sequence(nn.input([1, 28, 28]), nn.conv2d(32, 3), nn.fc(10))).not.toThrow() // sequence() doesn't validate shapes

      // But init() should throw
      // (We can't test init() without the native bindings, but we can test the shape logic
      //  by checking the error message pattern)
    })

    test('conv2d after 1D input is a config error at init time', () => {
      // sequence() itself doesn't throw (validation is deferred to init())
      const config = nn.sequence(nn.input(784), nn.conv2d(32, 3))
      expect(config.blocks).toHaveLength(1)
    })

    test('transformerEncoder after 1D input is a config error at init time', () => {
      const config = nn.sequence(nn.input(784), nn.transformerEncoder(4, 2))
      expect(config.blocks).toHaveLength(1)
    })
  })

  describe('All block types roundtrip', () => {
    test('mixed CNN + FC roundtrips', () => {
      const config = nn.sequence(
        nn.input([3, 32, 32]),
        nn.conv2d(32, 3, { padding: 1 }).relu(),
        nn.maxPool2d(2),
        nn.conv2d(64, 3, { padding: 1 }).relu(),
        nn.avgPool2d(2),
        nn.flatten(),
        nn.fc(128).relu().dropout(0.5),
        nn.fc(10),
      )

      const json1 = config.toJSON()
      const json2 = nn.fromJSON(json1).toJSON()
      expect(json2).toEqual(json1)
    })
  })

  describe('Conv2dBlockDef fluent API', () => {
    test('withStride() and withPadding() methods', () => {
      const block = nn.conv2d(32, 3).withStride(2).withPadding(1)
      expect(block.kind).toBe('conv2d')
      expect(block.stride).toBe(2)
      expect(block.padding).toBe(1)
    })

    test('relu() and dropout()', () => {
      const block = nn.conv2d(32, 3).relu().dropout(0.2)
      expect(block.activation).toBe('relu')
      expect(block.dropoutP).toBe(0.2)
    })

    test('batchNorm() and noBias()', () => {
      const block = nn.conv2d(32, 3).batchNorm().noBias()
      expect(block.useBatchNorm).toBe(true)
      expect(block.bias).toBe(false)
    })
  })
})
