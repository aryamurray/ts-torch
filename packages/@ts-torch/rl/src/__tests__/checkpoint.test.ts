import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { mkdir, rm } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import {
  saveCheckpoint,
  loadCheckpoint,
  encodeCheckpoint,
  decodeCheckpoint,
  float32Tensor,
  paramsToTensors,
  type CheckpointData,
} from '@ts-torch/nn'

describe('Checkpoint', () => {
  let testDir: string

  beforeEach(async () => {
    testDir = join(tmpdir(), `checkpoint-test-${Date.now()}`)
    await mkdir(testDir, { recursive: true })
  })

  afterEach(async () => {
    if (existsSync(testDir)) {
      await rm(testDir, { recursive: true })
    }
  })

  describe('encodeCheckpoint / decodeCheckpoint', () => {
    it('roundtrips simple checkpoint', () => {
      const original: CheckpointData = {
        tensors: {
          'layer1.weight': {
            data: new Float32Array([1, 2, 3, 4]),
            shape: [2, 2],
            dtype: 'float32',
          },
          'layer1.bias': {
            data: new Float32Array([0.1, 0.2]),
            shape: [2],
            dtype: 'float32',
          },
        },
        metadata: {
          stepCount: 1000,
          version: '1.0.0',
        },
      }

      const encoded = encodeCheckpoint(original)
      const decoded = decodeCheckpoint(encoded)

      expect(decoded.metadata).toEqual(original.metadata)
      expect(Object.keys(decoded.tensors)).toEqual(Object.keys(original.tensors))

      for (const name of Object.keys(original.tensors)) {
        const origTensor = original.tensors[name]!
        const decTensor = decoded.tensors[name]!

        expect(decTensor.shape).toEqual(origTensor.shape)
        expect(decTensor.dtype).toBe(origTensor.dtype)
        expect(Array.from(decTensor.data as Float32Array)).toEqual(
          Array.from(origTensor.data as Float32Array),
        )
      }
    })

    it('handles empty metadata', () => {
      const original: CheckpointData = {
        tensors: {
          weight: {
            data: new Float32Array([1, 2]),
            shape: [2],
            dtype: 'float32',
          },
        },
      }

      const encoded = encodeCheckpoint(original)
      const decoded = decodeCheckpoint(encoded)

      expect(decoded.tensors['weight']).toBeDefined()
      expect(decoded.metadata).toEqual({})
    })

    it('handles empty tensors', () => {
      const original: CheckpointData = {
        tensors: {},
        metadata: { key: 'value' },
      }

      const encoded = encodeCheckpoint(original)
      const decoded = decodeCheckpoint(encoded)

      expect(Object.keys(decoded.tensors)).toHaveLength(0)
      expect(decoded.metadata).toEqual({ key: 'value' })
    })

    it('preserves different dtypes', () => {
      const original: CheckpointData = {
        tensors: {
          float32: {
            data: new Float32Array([1.5, 2.5]),
            shape: [2],
            dtype: 'float32',
          },
          float64: {
            data: new Float64Array([1.5, 2.5]),
            shape: [2],
            dtype: 'float64',
          },
          int32: {
            data: new Int32Array([1, 2, 3]),
            shape: [3],
            dtype: 'int32',
          },
          uint8: {
            data: new Uint8Array([255, 128, 0]),
            shape: [3],
            dtype: 'uint8',
          },
        },
      }

      const encoded = encodeCheckpoint(original)
      const decoded = decodeCheckpoint(encoded)

      expect(decoded.tensors['float32']!.dtype).toBe('float32')
      expect(decoded.tensors['float64']!.dtype).toBe('float64')
      expect(decoded.tensors['int32']!.dtype).toBe('int32')
      expect(decoded.tensors['uint8']!.dtype).toBe('uint8')

      expect(Array.from(decoded.tensors['float32']!.data as Float32Array)).toEqual([1.5, 2.5])
      expect(Array.from(decoded.tensors['float64']!.data as Float64Array)).toEqual([1.5, 2.5])
      expect(Array.from(decoded.tensors['int32']!.data as Int32Array)).toEqual([1, 2, 3])
      expect(Array.from(decoded.tensors['uint8']!.data as Uint8Array)).toEqual([255, 128, 0])
    })

    it('handles multi-dimensional tensors', () => {
      const original: CheckpointData = {
        tensors: {
          tensor3d: {
            data: new Float32Array(24), // 2 * 3 * 4 = 24 elements
            shape: [2, 3, 4],
            dtype: 'float32',
          },
        },
      }

      const encoded = encodeCheckpoint(original)
      const decoded = decodeCheckpoint(encoded)

      expect(decoded.tensors['tensor3d']!.shape).toEqual([2, 3, 4])
    })

    it('throws for invalid magic bytes', () => {
      const invalid = new Uint8Array([0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00])
      expect(() => decodeCheckpoint(invalid)).toThrow('bad magic bytes')
    })

    it('throws for unsupported version', () => {
      // Create buffer with valid magic but wrong version
      const buffer = new Uint8Array(20)
      buffer.set([0x54, 0x53, 0x4e, 0x4e]) // "TSNN" (nn package magic bytes)
      new DataView(buffer.buffer).setUint32(4, 999, true) // Version 999

      expect(() => decodeCheckpoint(buffer)).toThrow('Unsupported checkpoint version')
    })
  })

  describe('saveCheckpoint / loadCheckpoint', () => {
    it('saves and loads checkpoint file', async () => {
      const path = join(testDir, 'model.ckpt')
      const original: CheckpointData = {
        tensors: {
          weight: {
            data: new Float32Array([1, 2, 3, 4, 5, 6]),
            shape: [2, 3],
            dtype: 'float32',
          },
        },
        metadata: {
          epoch: 10,
        },
      }

      await saveCheckpoint(path, original)
      expect(existsSync(path)).toBe(true)

      const loaded = await loadCheckpoint(path)

      expect(loaded.metadata).toEqual(original.metadata)
      expect(Array.from(loaded.tensors['weight']!.data as Float32Array)).toEqual([1, 2, 3, 4, 5, 6])
    })

    it('handles large tensors', async () => {
      const path = join(testDir, 'large.ckpt')
      const largeData = new Float32Array(10000)
      for (let i = 0; i < largeData.length; i++) {
        largeData[i] = i * 0.001
      }

      const original: CheckpointData = {
        tensors: {
          large: {
            data: largeData,
            shape: [100, 100],
            dtype: 'float32',
          },
        },
      }

      await saveCheckpoint(path, original)
      const loaded = await loadCheckpoint(path)

      expect(loaded.tensors['large']!.data.length).toBe(10000)
      expect(loaded.tensors['large']!.shape).toEqual([100, 100])

      // Check a few values
      const data = loaded.tensors['large']!.data as Float32Array
      expect(Math.abs(data[0]! - 0)).toBeLessThan(0.0001)
      expect(Math.abs(data[1000]! - 1)).toBeLessThan(0.0001)
    })
  })

  describe('float32Tensor()', () => {
    it('creates TensorData from Float32Array', () => {
      const data = new Float32Array([1, 2, 3, 4])
      const tensor = float32Tensor(data, [2, 2])

      expect(tensor.data).toBe(data)
      expect(tensor.shape).toEqual([2, 2])
      expect(tensor.dtype).toBe('float32')
    })
  })

  describe('paramsToTensors()', () => {
    it('converts Map to tensor record', () => {
      const params = new Map<string, Float32Array>([
        ['layer1.weight', new Float32Array([1, 2, 3, 4])],
        ['layer1.bias', new Float32Array([0.1, 0.2])],
      ])

      const shapes = new Map<string, number[]>([
        ['layer1.weight', [2, 2]],
        ['layer1.bias', [2]],
      ])

      const tensors = paramsToTensors(params, shapes)

      expect(tensors['layer1.weight']!.shape).toEqual([2, 2])
      expect(tensors['layer1.bias']!.shape).toEqual([2])
      expect(Array.from(tensors['layer1.weight']!.data as Float32Array)).toEqual([1, 2, 3, 4])
    })

    it('uses default shape [length] if not provided', () => {
      const params = new Map<string, Float32Array>([
        ['weight', new Float32Array([1, 2, 3])],
      ])

      const shapes = new Map<string, number[]>()

      const tensors = paramsToTensors(params, shapes)

      expect(tensors['weight']!.shape).toEqual([3])
    })
  })

  describe('edge cases', () => {
    it('handles tensor with special characters in name', () => {
      const original: CheckpointData = {
        tensors: {
          'layer.0.weight': {
            data: new Float32Array([1, 2]),
            shape: [2],
            dtype: 'float32',
          },
          'module/sub-module/bias': {
            data: new Float32Array([3, 4]),
            shape: [2],
            dtype: 'float32',
          },
        },
      }

      const encoded = encodeCheckpoint(original)
      const decoded = decodeCheckpoint(encoded)

      expect(decoded.tensors['layer.0.weight']).toBeDefined()
      expect(decoded.tensors['module/sub-module/bias']).toBeDefined()
    })

    it('handles complex nested metadata', () => {
      const original: CheckpointData = {
        tensors: {
          weight: {
            data: new Float32Array([1]),
            shape: [1],
            dtype: 'float32',
          },
        },
        metadata: {
          config: {
            nested: {
              deep: {
                value: 42,
              },
            },
            array: [1, 2, 3],
          },
          nullValue: null,
          boolValue: true,
        },
      }

      const encoded = encodeCheckpoint(original)
      const decoded = decodeCheckpoint(encoded)

      expect(decoded.metadata).toEqual(original.metadata)
    })

    it('handles zero-element tensor', () => {
      const original: CheckpointData = {
        tensors: {
          empty: {
            data: new Float32Array(0),
            shape: [0],
            dtype: 'float32',
          },
        },
      }

      const encoded = encodeCheckpoint(original)
      const decoded = decodeCheckpoint(encoded)

      expect(decoded.tensors['empty']!.data.length).toBe(0)
      expect(decoded.tensors['empty']!.shape).toEqual([0])
    })
  })
})
