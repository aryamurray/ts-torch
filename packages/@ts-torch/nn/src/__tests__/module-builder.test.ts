/**
 * Tests for nn.heads() builder and HeadedSequential
 */

import { describe, test, expect } from 'vitest'
import { nn } from '../builders.js'

describe('nn.heads() builder', () => {
  describe('config creation', () => {
    test('nn.sequence() without heads returns SequenceDef (unchanged)', () => {
      const config = nn.sequence(nn.input(4), nn.fc(64).tanh(), nn.fc(10))
      expect(config.inputDef.shape).toEqual([4])
      expect(config.blocks).toHaveLength(2)
    })

    test('nn.sequence() with nn.heads() creates config', () => {
      const config = nn.sequence(
        nn.input(4),
        nn.fc(64).tanh(),
        nn.heads({
          pi: nn.sequence(nn.fc(32).relu(), nn.fc(2)),
          vf: nn.sequence(nn.fc(1)),
        }),
      )
      expect(config.blocks).toHaveLength(2) // fc + heads
      expect(config.blocks[1]!.kind).toBe('heads')
    })

    test('nn.heads() must be the last block', () => {
      expect(() => nn.sequence(nn.input(4), nn.heads({ a: nn.sequence(nn.fc(1)) }), nn.fc(10))).toThrow(
        'nn.heads() must be the last block',
      )
    })

    test('nn.heads() requires at least one head', () => {
      expect(() => nn.heads({})).toThrow('at least one head')
    })

    test('nn.heads() rejects invalid defaultHead', () => {
      expect(() => nn.heads({ a: nn.sequence(nn.fc(1)) }, { defaultHead: 'missing' })).toThrow(
        'Default head "missing" not found',
      )
    })

    test('headless nn.sequence(nn.fc(32)) works inside nn.heads()', () => {
      const headless = nn.sequence(nn.fc(32).relu(), nn.fc(2))
      expect(headless.blocks).toHaveLength(2)
      expect((headless.blocks[0]! as any).outFeatures).toBe(32)
    })

    test('headless sequence requires at least one block', () => {
      expect(() => nn.sequence()).toThrow('requires at least one block')
    })

    test('nn.heads() cannot be nested inside headless sequence', () => {
      expect(() => nn.sequence(nn.heads({ a: nn.sequence(nn.fc(1)) }))).toThrow(
        'nn.heads() cannot be used inside a headless sequence',
      )
    })
  })

  describe('toJSON / fromJSON', () => {
    test('without heads, serializes as version 2 (backward compat)', () => {
      const config = nn.sequence(nn.input(4), nn.fc(10))
      const json = config.toJSON() as any
      expect(json.version).toBe(2)
      expect(json.blocks).toHaveLength(1)
      expect(json.blocks[0].kind).toBe('fc')
    })

    test('with heads, serializes as version 3', () => {
      const config = nn.sequence(
        nn.input(4),
        nn.fc(64).tanh(),
        nn.heads({
          pi: nn.sequence(nn.fc(32).relu(), nn.fc(2)),
          vf: nn.sequence(nn.fc(1)),
        }),
      )

      const json = config.toJSON() as any
      expect(json.version).toBe(3)
      expect(json.blocks).toHaveLength(2)
      expect(json.blocks[0].kind).toBe('fc')
      expect(json.blocks[1].kind).toBe('heads')
      expect(json.blocks[1].heads.pi.blocks).toHaveLength(2)
      expect(json.blocks[1].heads.vf.blocks).toHaveLength(1)
    })

    test('toJSON → fromJSON roundtrip with heads', () => {
      const config = nn.sequence(
        nn.input(4),
        nn.fc(64).tanh(),
        nn.heads(
          {
            pi: nn.sequence(nn.fc(32).relu(), nn.fc(2)),
            vf: nn.sequence(nn.fc(1)),
          },
          { defaultHead: 'pi' },
        ),
      )

      const json1 = config.toJSON()
      const reconstructed = nn.fromJSON(json1)
      const json2 = reconstructed.toJSON()
      expect(json2).toEqual(json1)
    })

    test('toJSON → fromJSON roundtrip without heads (version 2)', () => {
      const config = nn.sequence(nn.input(784), nn.fc(128).relu(), nn.fc(10))
      const json1 = config.toJSON()
      const json2 = nn.fromJSON(json1).toJSON()
      expect(json2).toEqual(json1)
    })

    test('version 1 config (no kind field) still loads through v3 code', () => {
      const config = nn.fromJSON({
        format: 'ts-torch-sequence',
        version: 1,
        input: { shape: [784] },
        blocks: [{ outFeatures: 128, activation: 'relu' }, { outFeatures: 10 }],
      })
      expect(config.blocks).toHaveLength(2)
      expect(config.blocks[0]!.kind).toBe('fc')
    })

    test('version 2 config (with kind, no heads) still loads through v3 code', () => {
      const config = nn.fromJSON({
        format: 'ts-torch-sequence',
        version: 2,
        input: { shape: [4] },
        blocks: [
          { kind: 'fc', outFeatures: 64, activation: 'tanh' },
          { kind: 'fc', outFeatures: 10 },
        ],
      })
      expect(config.blocks).toHaveLength(2)
      expect(config.blocks[0]!.kind).toBe('fc')
    })

    test('heads block fromJSON validation: missing heads object', () => {
      expect(() =>
        nn.fromJSON({
          format: 'ts-torch-sequence',
          version: 3,
          input: { shape: [4] },
          blocks: [{ kind: 'heads' }],
        }),
      ).toThrow('requires a "heads" object')
    })

    test('heads block fromJSON validation: empty heads', () => {
      expect(() =>
        nn.fromJSON({
          format: 'ts-torch-sequence',
          version: 3,
          input: { shape: [4] },
          blocks: [{ kind: 'heads', heads: {} }],
        }),
      ).toThrow('at least one head')
    })

    test('heads block fromJSON validation: empty head blocks', () => {
      expect(() =>
        nn.fromJSON({
          format: 'ts-torch-sequence',
          version: 3,
          input: { shape: [4] },
          blocks: [{ kind: 'heads', heads: { pi: { blocks: [] } } }],
        }),
      ).toThrow('non-empty "blocks" array')
    })

    test('serializes defaultHead in heads block', () => {
      const config = nn.sequence(
        nn.input(4),
        nn.heads({ pi: nn.sequence(nn.fc(2)), vf: nn.sequence(nn.fc(1)) }, { defaultHead: 'vf' }),
      )

      const json = config.toJSON() as any
      expect(json.blocks[0].kind).toBe('heads')
      expect(json.blocks[0].defaultHead).toBe('vf')
    })
  })
})
