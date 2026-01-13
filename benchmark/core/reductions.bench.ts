/**
 * Reduction Operation Benchmarks
 *
 * Tests sum, mean, and dimensional reductions.
 */

import { Bench } from 'tinybench'
import { device, run } from '@ts-torch/core'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'

const cpu = device.cpu()
import { STANDARD_SIZES } from '../lib/utils.js'

export const suite: BenchmarkSuite = {
  name: 'Reduction Operations',
  category: 'core',

  async run(config: BenchmarkConfig) {
    const bench = new Bench({
      time: config.time ?? 1000,
      warmup: config.warmup ?? true,
    })

    const sizes = config.sizes ?? STANDARD_SIZES

    // Global reductions
    for (const [m, n] of sizes) {
      const label = `[${m}, ${n}]`

      bench.add(`sum ${label}`, () => {
        run(() => {
          const a = cpu.randn([m, n] as const)
          return a.sum()
        })
      })

      bench.add(`mean ${label}`, () => {
        run(() => {
          const a = cpu.randn([m, n] as const)
          return a.mean()
        })
      })
    }

    // 3D tensor reductions
    bench.add('sum [64, 128, 128]', () => {
      run(() => {
        const a = cpu.randn([64, 128, 128] as const)
        return a.sum()
      })
    })

    bench.add('mean [64, 128, 128]', () => {
      run(() => {
        const a = cpu.randn([64, 128, 128] as const)
        return a.mean()
      })
    })

    // 4D tensor reductions (batch, channel, height, width)
    bench.add('sum [32, 64, 32, 32]', () => {
      run(() => {
        const a = cpu.randn([32, 64, 32, 32] as const)
        return a.sum()
      })
    })

    bench.add('mean [32, 64, 32, 32]', () => {
      run(() => {
        const a = cpu.randn([32, 64, 32, 32] as const)
        return a.mean()
      })
    })

    await bench.run()
    return bench
  },
}

export default suite
