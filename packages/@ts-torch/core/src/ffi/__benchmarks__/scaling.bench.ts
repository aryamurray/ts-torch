/**
 * Scaling Benchmarks
 *
 * Tests how performance scales with tensor size and operation complexity.
 */

import { bench, describe } from 'vitest'
import { zeros, randn } from '../../tensor/factory.js'

describe('Tensor Size Scaling - zeros', () => {
  bench('zeros [32, 32] (1K elements)', () => {
    zeros([32, 32] as const)
  })

  bench('zeros [64, 64] (4K elements)', () => {
    zeros([64, 64] as const)
  })

  bench('zeros [128, 128] (16K elements)', () => {
    zeros([128, 128] as const)
  })

  bench('zeros [256, 256] (65K elements)', () => {
    zeros([256, 256] as const)
  })

  bench('zeros [512, 512] (262K elements)', () => {
    zeros([512, 512] as const)
  })

  bench('zeros [1024, 1024] (1M elements)', () => {
    zeros([1024, 1024] as const)
  })
})

describe('Tensor Size Scaling - randn', () => {
  bench('randn [32, 32] (1K elements)', () => {
    randn([32, 32] as const)
  })

  bench('randn [64, 64] (4K elements)', () => {
    randn([64, 64] as const)
  })

  bench('randn [128, 128] (16K elements)', () => {
    randn([128, 128] as const)
  })

  bench('randn [256, 256] (65K elements)', () => {
    randn([256, 256] as const)
  })

  bench('randn [512, 512] (262K elements)', () => {
    randn([512, 512] as const)
  })

  bench('randn [1024, 1024] (1M elements)', () => {
    randn([1024, 1024] as const)
  })
})

describe('Matmul Scaling', () => {
  const a32 = randn([32, 32] as const)
  const b32 = randn([32, 32] as const)
  bench('matmul [32, 32] @ [32, 32]', () => {
    a32.matmul(b32)
  })

  const a64 = randn([64, 64] as const)
  const b64 = randn([64, 64] as const)
  bench('matmul [64, 64] @ [64, 64]', () => {
    a64.matmul(b64)
  })

  const a128 = randn([128, 128] as const)
  const b128 = randn([128, 128] as const)
  bench('matmul [128, 128] @ [128, 128]', () => {
    a128.matmul(b128)
  })

  const a256 = randn([256, 256] as const)
  const b256 = randn([256, 256] as const)
  bench('matmul [256, 256] @ [256, 256]', () => {
    a256.matmul(b256)
  })

  const a512 = randn([512, 512] as const)
  const b512 = randn([512, 512] as const)
  bench('matmul [512, 512] @ [512, 512]', () => {
    a512.matmul(b512)
  })
})

describe('Element-wise Scaling', () => {
  const a64 = randn([64, 64] as const)
  const b64 = randn([64, 64] as const)
  bench('add [64, 64]', () => {
    a64.add(b64)
  })

  const a256 = randn([256, 256] as const)
  const b256 = randn([256, 256] as const)
  bench('add [256, 256]', () => {
    a256.add(b256)
  })

  const a1024 = randn([1024, 1024] as const)
  const b1024 = randn([1024, 1024] as const)
  bench('add [1024, 1024]', () => {
    a1024.add(b1024)
  })
})

describe('Batch Dimension Scaling', () => {
  // Simulating batch processing scenarios
  const batch8 = randn([8, 64, 64] as const)
  bench('sum over batch=8 [8, 64, 64]', () => {
    batch8.sum()
  })

  const batch32 = randn([32, 64, 64] as const)
  bench('sum over batch=32 [32, 64, 64]', () => {
    batch32.sum()
  })

  const batch128 = randn([128, 64, 64] as const)
  bench('sum over batch=128 [128, 64, 64]', () => {
    batch128.sum()
  })
})

describe('Activation Scaling', () => {
  const t64 = randn([64, 64] as const)
  bench('relu [64, 64]', () => {
    t64.relu()
  })

  const t256 = randn([256, 256] as const)
  bench('relu [256, 256]', () => {
    t256.relu()
  })

  const t1024 = randn([1024, 1024] as const)
  bench('relu [1024, 1024]', () => {
    t1024.relu()
  })
})

describe('Realistic ML Workloads', () => {
  // Mini-batch forward pass simulation
  const input = randn([32, 784] as const)  // batch=32, MNIST input
  const w1 = randn([784, 256] as const)
  const w2 = randn([256, 10] as const)

  bench('MLP forward: [32,784] @ [784,256] @ [256,10]', () => {
    const h = input.matmul(w1).relu()
    h.matmul(w2)
  })

  // Conv-like workload (flattened)
  const _convInput = randn([16, 3, 32, 32] as const)  // batch=16, 3 channels, 32x32
  bench('4D tensor creation [16, 3, 32, 32]', () => {
    randn([16, 3, 32, 32] as const)
  })
})
