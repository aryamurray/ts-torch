/**
 * Matrix Operation Benchmarks
 *
 * Tests matmul, transpose, reshape operations.
 */

import { Bench } from 'tinybench'
import { device, run } from '@ts-torch/core'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'

const cpu = device.cpu()

export const suite: BenchmarkSuite = {
  name: 'Matrix Operations',
  category: 'core',

  async run(config: BenchmarkConfig) {
    const bench = new Bench({
      time: config.time ?? 1000,
      warmup: config.warmup ?? true,
    })

    // Matrix multiplication - various sizes
    const matmulSizes: Array<[number, number, number]> = [
      [32, 32, 32], // Small square
      [64, 64, 64],
      [128, 128, 128],
      [256, 256, 256],
      [512, 512, 512],
    ]

    for (const [m, k, n] of matmulSizes) {
      bench.add(`matmul [${m}, ${k}] @ [${k}, ${n}]`, () => {
        run(() => {
          const a = cpu.randn([m, k] as const)
          const b = cpu.randn([k, n] as const)
          return a.matmul(b)
        })
      })
    }

    // Rectangular matrices
    bench.add('matmul [128, 64] @ [64, 256]', () => {
      run(() => {
        const a = cpu.randn([128, 64] as const)
        const b = cpu.randn([64, 256] as const)
        return a.matmul(b)
      })
    })

    bench.add('matmul [256, 128] @ [128, 64]', () => {
      run(() => {
        const a = cpu.randn([256, 128] as const)
        const b = cpu.randn([128, 64] as const)
        return a.matmul(b)
      })
    })

    // Vector-matrix multiplication
    bench.add('matmul [1, 256] @ [256, 256]', () => {
      run(() => {
        const a = cpu.randn([1, 256] as const)
        const b = cpu.randn([256, 256] as const)
        return a.matmul(b)
      })
    })

    // Transpose operations
    const transposeSizes: Array<[number, number]> = [
      [128, 128],
      [256, 256],
      [512, 512],
      [1024, 1024],
    ]

    for (const [m, n] of transposeSizes) {
      bench.add(`transpose [${m}, ${n}]`, () => {
        run(() => {
          const a = cpu.randn([m, n] as const)
          return a.transpose(0, 1)
        })
      })
    }

    // Reshape operations (should be fast, view-based)
    bench.add('reshape [256, 256] -> [65536]', () => {
      run(() => {
        const a = cpu.randn([256, 256] as const)
        return a.reshape([65536] as const)
      })
    })

    bench.add('reshape [1024, 1024] -> [1, 1048576]', () => {
      run(() => {
        const a = cpu.randn([1024, 1024] as const)
        return a.reshape([1, 1048576] as const)
      })
    })

    bench.add('reshape [256, 256] -> [64, 1024]', () => {
      run(() => {
        const a = cpu.randn([256, 256] as const)
        return a.reshape([64, 1024] as const)
      })
    })

    // Flatten
    bench.add('flatten [32, 32, 32]', () => {
      run(() => {
        const a = cpu.randn([32, 32, 32] as const)
        return a.flatten()
      })
    })

    await bench.run()
    return bench
  },
}

export default suite
