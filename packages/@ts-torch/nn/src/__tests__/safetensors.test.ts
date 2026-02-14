/**
 * Tests for safetensors encode/decode
 */

import { describe, test, expect } from 'vitest'
import {
  encodeSafetensors,
  decodeSafetensors,
  serializeMetadata,
  deserializeMetadata,
} from '../safetensors.js'
import type { StateDict } from '../safetensors.js'

describe('Safetensors format', () => {
  describe('encode/decode roundtrip', () => {
    test('roundtrips float32 tensors', () => {
      const state: StateDict = {
        'weight': {
          data: new Float32Array([1, 2, 3, 4, 5, 6]),
          shape: [2, 3],
          dtype: 'float32',
        },
      }

      const encoded = encodeSafetensors(state)
      const { tensors: decoded } = decodeSafetensors(encoded)

      expect(decoded['weight']!.dtype).toBe('float32')
      expect(decoded['weight']!.shape).toEqual([2, 3])
      expect(Array.from(decoded['weight']!.data as Float32Array)).toEqual([1, 2, 3, 4, 5, 6])
    })

    test('roundtrips float64 tensors', () => {
      const state: StateDict = {
        'param': {
          data: new Float64Array([1.5, 2.5, 3.5]),
          shape: [3],
          dtype: 'float64',
        },
      }

      const encoded = encodeSafetensors(state)
      const { tensors: decoded } = decodeSafetensors(encoded)

      expect(decoded['param']!.dtype).toBe('float64')
      expect(decoded['param']!.shape).toEqual([3])
      expect(Array.from(decoded['param']!.data as Float64Array)).toEqual([1.5, 2.5, 3.5])
    })

    test('roundtrips int32 tensors', () => {
      const state: StateDict = {
        'indices': {
          data: new Int32Array([10, 20, 30]),
          shape: [3],
          dtype: 'int32',
        },
      }

      const encoded = encodeSafetensors(state)
      const { tensors: decoded } = decodeSafetensors(encoded)

      expect(decoded['indices']!.dtype).toBe('int32')
      expect(Array.from(decoded['indices']!.data as Int32Array)).toEqual([10, 20, 30])
    })

    test('roundtrips uint8 tensors', () => {
      const state: StateDict = {
        'mask': {
          data: new Uint8Array([0, 128, 255]),
          shape: [3],
          dtype: 'uint8',
        },
      }

      const encoded = encodeSafetensors(state)
      const { tensors: decoded } = decodeSafetensors(encoded)

      expect(decoded['mask']!.dtype).toBe('uint8')
      expect(Array.from(decoded['mask']!.data as Uint8Array)).toEqual([0, 128, 255])
    })

    test('roundtrips int64 (BigInt64Array) tensors', () => {
      const state: StateDict = {
        'ids': {
          data: new BigInt64Array([1n, 2n, 3n]),
          shape: [3],
          dtype: 'int64',
        },
      }

      const encoded = encodeSafetensors(state)
      const { tensors: decoded } = decodeSafetensors(encoded)

      expect(decoded['ids']!.dtype).toBe('int64')
      const arr = decoded['ids']!.data as BigInt64Array
      expect(arr[0]).toBe(1n)
      expect(arr[1]).toBe(2n)
      expect(arr[2]).toBe(3n)
    })

    test('roundtrips float16 tensors', () => {
      const raw = new Uint16Array([0x3C00, 0x4000])
      const state: StateDict = {
        'half': {
          data: raw,
          shape: [2],
          dtype: 'float16',
        },
      }

      const encoded = encodeSafetensors(state)
      const { tensors: decoded } = decodeSafetensors(encoded)

      expect(decoded['half']!.dtype).toBe('float16')
      const arr = decoded['half']!.data as Uint16Array
      expect(arr[0]).toBe(0x3C00)
      expect(arr[1]).toBe(0x4000)
    })

    test('roundtrips bfloat16 tensors', () => {
      const raw = new Uint16Array([0x3F80, 0x4000])
      const state: StateDict = {
        'bf': {
          data: raw,
          shape: [2],
          dtype: 'bfloat16',
        },
      }

      const encoded = encodeSafetensors(state)
      const { tensors: decoded } = decodeSafetensors(encoded)

      expect(decoded['bf']!.dtype).toBe('bfloat16')
      const arr = decoded['bf']!.data as Uint16Array
      expect(arr[0]).toBe(0x3F80)
      expect(arr[1]).toBe(0x4000)
    })

    test('roundtrips bool tensors', () => {
      const state: StateDict = {
        'flags': {
          data: new Uint8Array([1, 0, 1, 0]),
          shape: [4],
          dtype: 'bool',
        },
      }

      const encoded = encodeSafetensors(state)
      const { tensors: decoded } = decodeSafetensors(encoded)

      expect(decoded['flags']!.dtype).toBe('bool')
      expect(Array.from(decoded['flags']!.data as Uint8Array)).toEqual([1, 0, 1, 0])
    })

    test('roundtrips multiple tensors', () => {
      const state: StateDict = {
        '0.weight': {
          data: new Float32Array([1, 2, 3, 4, 5, 6]),
          shape: [2, 3],
          dtype: 'float32',
        },
        '0.bias': {
          data: new Float32Array([0.1, 0.2]),
          shape: [2],
          dtype: 'float32',
        },
      }

      const encoded = encodeSafetensors(state)
      const { tensors: decoded } = decodeSafetensors(encoded)

      expect(Object.keys(decoded)).toHaveLength(2)
      expect(decoded['0.weight']!.shape).toEqual([2, 3])
      expect(decoded['0.bias']!.shape).toEqual([2])
    })
  })

  describe('metadata', () => {
    test('preserves metadata in roundtrip', () => {
      const state: StateDict = {
        'w': {
          data: new Float32Array([1]),
          shape: [1],
          dtype: 'float32',
        },
      }

      const metadata = { framework: 'ts-torch', version: '0.1.0' }
      const encoded = encodeSafetensors(state, metadata)

      // Parse header to check metadata is present
      const view = new DataView(encoded.buffer, encoded.byteOffset, encoded.byteLength)
      const headerSize = Number(view.getBigUint64(0, true))
      const headerJson = new TextDecoder().decode(encoded.slice(8, 8 + headerSize))
      const header = JSON.parse(headerJson)

      expect(header.__metadata__).toEqual(metadata)
    })

    test('decodeSafetensors extracts metadata', () => {
      const state: StateDict = {
        'w': {
          data: new Float32Array([1]),
          shape: [1],
          dtype: 'float32',
        },
      }

      const metadata = { framework: 'ts-torch', version: '0.1.0' }
      const encoded = encodeSafetensors(state, metadata)
      const result = decodeSafetensors(encoded)

      expect(result.metadata).toEqual(metadata)
      expect(result.tensors['w']).toBeDefined()
    })

    test('decodeSafetensors returns empty metadata when none present', () => {
      const state: StateDict = {
        'w': {
          data: new Float32Array([1]),
          shape: [1],
          dtype: 'float32',
        },
      }

      const encoded = encodeSafetensors(state)
      const result = decodeSafetensors(encoded)

      expect(result.metadata).toEqual({})
    })
  })

  describe('serializeMetadata', () => {
    test('always adds framework: "ts-torch"', () => {
      const result = serializeMetadata()
      expect(result).toEqual({ framework: 'ts-torch' })
    })

    test('JSON-stringifies string values', () => {
      const result = serializeMetadata({ note: 'hello' })
      expect(result.note).toBe('"hello"')
      expect(result.framework).toBe('ts-torch')
    })

    test('JSON-stringifies non-string values', () => {
      const result = serializeMetadata({ epoch: 5, loss: 0.01, tags: ['a', 'b'] })
      expect(result.epoch).toBe('5')
      expect(result.loss).toBe('0.01')
      expect(result.tags).toBe('["a","b"]')
    })

    test('does not let user override framework tag', () => {
      const result = serializeMetadata({ framework: 'pytorch' })
      expect(result.framework).toBe('ts-torch')
    })
  })

  describe('deserializeMetadata', () => {
    test('strips framework key', () => {
      const result = deserializeMetadata({ framework: 'ts-torch', epoch: '5' })
      expect(result.framework).toBeUndefined()
      expect(result.epoch).toBe(5)
    })

    test('JSON-parses numeric and object values', () => {
      const result = deserializeMetadata({
        framework: 'ts-torch',
        epoch: '5',
        loss: '0.01',
        tags: '["a","b"]',
        config: '{"lr":0.001}',
      })

      expect(result.epoch).toBe(5)
      expect(result.loss).toBe(0.01)
      expect(result.tags).toEqual(['a', 'b'])
      expect(result.config).toEqual({ lr: 0.001 })
    })

    test('passes through plain strings that are not JSON', () => {
      const result = deserializeMetadata({
        framework: 'ts-torch',
        note: 'hello world',
      })

      expect(result.note).toBe('hello world')
    })

    test('handles empty metadata', () => {
      const result = deserializeMetadata({})
      expect(result).toEqual({})
    })

    test('roundtrips string "5" without type coercion', () => {
      const serialized = serializeMetadata({ value: '5' })
      const deserialized = deserializeMetadata(serialized)
      expect(deserialized.value).toBe('5')
      expect(typeof deserialized.value).toBe('string')
    })

    test('roundtrips string "true" without type coercion', () => {
      const serialized = serializeMetadata({ value: 'true' })
      const deserialized = deserializeMetadata(serialized)
      expect(deserialized.value).toBe('true')
      expect(typeof deserialized.value).toBe('string')
    })

    test('roundtrips string "null" without type coercion', () => {
      const serialized = serializeMetadata({ value: 'null' })
      const deserialized = deserializeMetadata(serialized)
      expect(deserialized.value).toBe('null')
      expect(typeof deserialized.value).toBe('string')
    })
  })

  describe('header format', () => {
    test('uses correct safetensors dtype names in header', () => {
      const state: StateDict = {
        'w': {
          data: new Float32Array([1]),
          shape: [1],
          dtype: 'float32',
        },
      }

      const encoded = encodeSafetensors(state)

      // Parse header
      const view = new DataView(encoded.buffer, encoded.byteOffset, encoded.byteLength)
      const headerSize = Number(view.getBigUint64(0, true))
      const headerJson = new TextDecoder().decode(encoded.slice(8, 8 + headerSize))
      const header = JSON.parse(headerJson)

      expect(header['w'].dtype).toBe('F32')
      expect(header['w'].shape).toEqual([1])
      expect(header['w'].data_offsets).toEqual([0, 4])
    })

    test('keys are sorted in output', () => {
      const state: StateDict = {
        'z.weight': { data: new Float32Array([1]), shape: [1], dtype: 'float32' },
        'a.weight': { data: new Float32Array([2]), shape: [1], dtype: 'float32' },
        'm.weight': { data: new Float32Array([3]), shape: [1], dtype: 'float32' },
      }

      const encoded = encodeSafetensors(state)
      const view = new DataView(encoded.buffer, encoded.byteOffset, encoded.byteLength)
      const headerSize = Number(view.getBigUint64(0, true))
      const headerJson = new TextDecoder().decode(encoded.slice(8, 8 + headerSize))
      const header = JSON.parse(headerJson)

      const keys = Object.keys(header)
      expect(keys).toEqual(['a.weight', 'm.weight', 'z.weight'])
    })
  })

  describe('error handling', () => {
    test('throws on buffer too small', () => {
      expect(() => decodeSafetensors(new Uint8Array(4))).toThrow(/too small/)
    })

    test('throws on header size exceeding buffer', () => {
      const buf = new Uint8Array(16)
      const view = new DataView(buf.buffer)
      view.setBigUint64(0, 9999n, true) // huge header size
      expect(() => decodeSafetensors(buf)).toThrow(/header size exceeds/)
    })

    test('throws on unknown dtype for encoding', () => {
      const state: StateDict = {
        'bad': {
          data: new Float32Array([1]),
          shape: [1],
          dtype: 'complex128',
        },
      }

      expect(() => encodeSafetensors(state)).toThrow(/Cannot encode dtype/)
    })

    test('throws on data length mismatch during encoding', () => {
      const state: StateDict = {
        'bad': {
          data: new Float32Array([1, 2, 3]),
          shape: [2, 3],
          dtype: 'float32',
        },
      }

      expect(() => encodeSafetensors(state)).toThrow(/Data length mismatch/)
    })

    test('throws on non-string metadata values during decoding', () => {
      // Craft a safetensors buffer with non-string metadata
      const header = JSON.stringify({
        __metadata__: { bad: 42 },
        w: { dtype: 'F32', shape: [1], data_offsets: [0, 4] },
      })
      const headerBytes = new TextEncoder().encode(header)
      const buf = new Uint8Array(8 + headerBytes.byteLength + 4)
      const view = new DataView(buf.buffer)
      view.setBigUint64(0, BigInt(headerBytes.byteLength), true)
      buf.set(headerBytes, 8)
      // Write one float32 zero for tensor data
      view.setFloat32(8 + headerBytes.byteLength, 0, true)

      expect(() => decodeSafetensors(buf)).toThrow(/Invalid safetensors metadata.*"bad".*non-string/)
    })
  })
})
