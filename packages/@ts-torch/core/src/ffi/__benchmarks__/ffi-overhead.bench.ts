/**
 * FFI Overhead Benchmarks
 *
 * Measures the overhead of FFI calls, buffer allocation, and tensor operations
 * to validate optimization improvements.
 *
 * Run with: bun run vitest bench packages/@ts-torch/core/src/ffi/__benchmarks__/
 */

import { bench, describe } from 'vitest'
import { zeros, ones, randn, rand, fromArray } from '../../tensor/factory.js'
import { createError, ERROR_STRUCT_SIZE } from '../error.js'
import { errorPool, shapeCache } from '../buffer-pool.js'

describe('Buffer Allocation', () => {
  bench('error buffer: new ArrayBuffer (baseline)', () => {
    const buffer = new ArrayBuffer(ERROR_STRUCT_SIZE)
    new DataView(buffer).setInt32(0, 0, true)
  })

  bench('error buffer: pool acquire/release', () => {
    const buffer = errorPool.acquire()
    errorPool.release(buffer)
  })

  bench('error buffer: createError (deprecated)', () => {
    createError()
  })

  bench('shape buffer: new BigInt64Array [3]', () => {
    const arr = new BigInt64Array(3)
    arr[0] = 64n
    arr[1] = 128n
    arr[2] = 256n
  })

  bench('shape buffer: pool fillShape [3]', () => {
    const buffer = shapeCache.fillShape([64, 128, 256])
    shapeCache.release(buffer)
  })

  bench('shape buffer: new BigInt64Array [4]', () => {
    const arr = new BigInt64Array(4)
    arr[0] = 32n
    arr[1] = 64n
    arr[2] = 128n
    arr[3] = 256n
  })

  bench('shape buffer: pool fillShape [4]', () => {
    const buffer = shapeCache.fillShape([32, 64, 128, 256])
    shapeCache.release(buffer)
  })
})

describe('Tensor Creation', () => {
  bench('zeros [64, 64]', () => {
    zeros([64, 64] as const)
  })

  bench('zeros [32, 32, 32]', () => {
    zeros([32, 32, 32] as const)
  })

  bench('ones [64, 64]', () => {
    ones([64, 64] as const)
  })

  bench('randn [64, 64]', () => {
    randn([64, 64] as const)
  })

  bench('rand [64, 64]', () => {
    rand([64, 64] as const)
  })

  bench('fromArray Float32 [64, 64]', () => {
    const data = new Float32Array(64 * 64)
    fromArray(data, [64, 64] as const)
  })
})

describe('Element-wise Operations', () => {
  const a = zeros([64, 64] as const)
  const b = zeros([64, 64] as const)

  bench('add (64x64)', () => {
    a.add(b)
  })

  bench('sub (64x64)', () => {
    a.sub(b)
  })

  bench('mul (64x64)', () => {
    a.mul(b)
  })

  bench('div (64x64)', () => {
    a.div(b)
  })
})

describe('Matrix Operations', () => {
  const a = randn([64, 64] as const)
  const b = randn([64, 64] as const)

  bench('matmul (64x64) @ (64x64)', () => {
    a.matmul(b)
  })

  const c = randn([128, 128] as const)
  const d = randn([128, 128] as const)

  bench('matmul (128x128) @ (128x128)', () => {
    c.matmul(d)
  })
})

describe('Memory Pressure', () => {
  bench('100 tensor creations [32, 32]', () => {
    for (let i = 0; i < 100; i++) {
      zeros([32, 32] as const)
    }
  })

  bench('1000 tensor creations [8, 8]', () => {
    for (let i = 0; i < 1000; i++) {
      zeros([8, 8] as const)
    }
  })
})

describe('Activation Functions', () => {
  const t = randn([64, 64] as const)

  bench('relu (64x64)', () => {
    t.relu()
  })

  bench('sigmoid (64x64)', () => {
    t.sigmoid()
  })

  bench('tanh (64x64)', () => {
    t.tanh()
  })
})
