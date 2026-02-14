/**
 * Tests for validateStateDict — shape, dtype, missing key, unexpected key checks
 */

import { describe, test, expect } from 'vitest'
import { run } from '@ts-torch/core'
import { Linear } from '../modules/linear.js'
import { Sequential } from '../modules/container.js'
import { ReLU } from '../modules/activation.js'
import {
  validateStateDict,
  MissingKeyError,
  UnexpectedKeyError,
  ShapeMismatchError,
  DTypeMismatchError,
  DataLengthMismatchError,
} from '../validation.js'
import type { StateDict } from '../safetensors.js'

describe('validateStateDict', () => {
  test('passes for matching state dict', () => {
    run(() => {
      const model = new Sequential(new Linear(4, 3))
      const state = model.stateDict()

      // Should not throw
      expect(() => validateStateDict(model, state)).not.toThrow()
    })
  })

  test('passes for multi-layer model', () => {
    run(() => {
      const model = new Sequential(new Linear(8, 4), new ReLU(), new Linear(4, 2))
      const state = model.stateDict()

      expect(() => validateStateDict(model, state)).not.toThrow()
    })
  })

  test('throws MissingKeyError when state dict is missing a key', () => {
    run(() => {
      const model = new Sequential(new Linear(4, 3))
      const state: StateDict = {}

      expect(() => validateStateDict(model, state)).toThrow(MissingKeyError)
      expect(() => validateStateDict(model, state)).toThrow(/Missing key/)
    })
  })

  test('throws UnexpectedKeyError when state dict has extra key', () => {
    run(() => {
      const model = new Sequential(new Linear(4, 3))
      const state = model.stateDict()

      // Add an extra key
      state['extra.weight'] = {
        data: new Float32Array([1, 2, 3]),
        shape: [3],
        dtype: 'float32',
      }

      expect(() => validateStateDict(model, state)).toThrow(UnexpectedKeyError)
      expect(() => validateStateDict(model, state)).toThrow(/Unexpected key/)
    })
  })

  test('throws ShapeMismatchError when shapes differ', () => {
    run(() => {
      const model = new Sequential(new Linear(4, 3))
      const state = model.stateDict()

      // Corrupt the shape of the weight tensor
      const weightKey = Object.keys(state).find((k) => k.includes('weight'))!
      state[weightKey] = {
        data: new Float32Array(6 * 4),
        shape: [6, 4], // Wrong shape — model expects [3, 4]
        dtype: 'float32',
      }

      expect(() => validateStateDict(model, state)).toThrow(ShapeMismatchError)
      expect(() => validateStateDict(model, state)).toThrow(/Shape mismatch/)
    })
  })

  test('throws DTypeMismatchError when dtypes differ', () => {
    run(() => {
      const model = new Sequential(new Linear(4, 3))
      const state = model.stateDict()

      // Change dtype of weight tensor
      const weightKey = Object.keys(state).find((k) => k.includes('weight'))!
      const original = state[weightKey]!
      state[weightKey] = {
        data: new Float64Array(Array.from(original.data as Float32Array)),
        shape: original.shape,
        dtype: 'float64',
      }

      expect(() => validateStateDict(model, state)).toThrow(DTypeMismatchError)
      expect(() => validateStateDict(model, state)).toThrow(/DType mismatch/)
    })
  })

  describe('strict: false', () => {
    test('allows missing keys when strict is false', () => {
      run(() => {
        const model = new Sequential(new Linear(4, 3), new ReLU(), new Linear(3, 2))
        const state = model.stateDict()

        // Remove the second layer's keys
        const keysToRemove = Object.keys(state).filter((k) => k.startsWith('2.'))
        for (const key of keysToRemove) {
          delete state[key]
        }

        // strict=false should not throw for missing keys
        expect(() => validateStateDict(model, state, false)).not.toThrow()
      })
    })

    test('allows extra keys when strict is false', () => {
      run(() => {
        const model = new Sequential(new Linear(4, 3))
        const state = model.stateDict()
        state['extra.param'] = {
          data: new Float32Array([1]),
          shape: [1],
          dtype: 'float32',
        }

        expect(() => validateStateDict(model, state, false)).not.toThrow()
      })
    })

    test('still throws ShapeMismatchError when strict is false', () => {
      run(() => {
        const model = new Sequential(new Linear(4, 3))
        const state = model.stateDict()

        const weightKey = Object.keys(state).find((k) => k.includes('weight'))!
        state[weightKey] = {
          data: new Float32Array(6 * 4),
          shape: [6, 4],
          dtype: 'float32',
        }

        expect(() => validateStateDict(model, state, false)).toThrow(ShapeMismatchError)
      })
    })

    test('still throws DTypeMismatchError when strict is false', () => {
      run(() => {
        const model = new Sequential(new Linear(4, 3))
        const state = model.stateDict()

        const weightKey = Object.keys(state).find((k) => k.includes('weight'))!
        const original = state[weightKey]!
        state[weightKey] = {
          data: new Float64Array(Array.from(original.data as Float32Array)),
          shape: original.shape,
          dtype: 'float64',
        }

        expect(() => validateStateDict(model, state, false)).toThrow(DTypeMismatchError)
      })
    })
  })

  test('throws DataLengthMismatchError when data length does not match shape', () => {
    run(() => {
      const model = new Sequential(new Linear(4, 3))
      const state = model.stateDict()

      const weightKey = Object.keys(state).find((k) => k.includes('weight'))!
      state[weightKey] = {
        data: new Float32Array(3), // 3 elements, but shape [3, 4] expects 12
        shape: [3, 4],
        dtype: 'float32',
      }

      expect(() => validateStateDict(model, state)).toThrow(DataLengthMismatchError)
      expect(() => validateStateDict(model, state)).toThrow(/Data length mismatch/)
    })
  })
})
