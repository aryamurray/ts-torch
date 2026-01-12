/**
 * Element-wise Operation Benchmarks
 *
 * Tests add, sub, mul, div operations at various sizes.
 */

import { Bench } from 'tinybench'
import { torch, run } from '@ts-torch/core'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'
import { STANDARD_SIZES } from '../lib/utils.js'

export const suite: BenchmarkSuite = {
  name: 'Element-wise Operations',
  category: 'core',

  async run(config: BenchmarkConfig) {
    const bench = new Bench({
      time: config.time ?? 1000,
      warmup: config.warmup ?? true,
    })

    const sizes = config.sizes ?? STANDARD_SIZES

    for (const [m, n] of sizes) {
      const label = `[${m}, ${n}]`

      // Tensor + Tensor operations
      bench.add(`add ${label}`, () => {
        run(() => {
          const a = torch.ones([m, n] as const)
          const b = torch.ones([m, n] as const)
          return a.add(b)
        })
      })

      bench.add(`sub ${label}`, () => {
        run(() => {
          const a = torch.ones([m, n] as const)
          const b = torch.ones([m, n] as const)
          return a.sub(b)
        })
      })

      bench.add(`mul ${label}`, () => {
        run(() => {
          const a = torch.ones([m, n] as const)
          const b = torch.ones([m, n] as const)
          return a.mul(b)
        })
      })

      bench.add(`div ${label}`, () => {
        run(() => {
          const a = torch.ones([m, n] as const)
          const b = torch.ones([m, n] as const).addScalar(1) // Avoid div by zero
          return a.div(b)
        })
      })
    }

    // Scalar operations
    const scalarSizes: Array<[number, number]> = [
      [128, 128],
      [512, 512],
    ]

    for (const [m, n] of scalarSizes) {
      const label = `[${m}, ${n}]`

      bench.add(`addScalar ${label}`, () => {
        run(() => {
          const a = torch.ones([m, n] as const)
          return a.addScalar(5)
        })
      })

      bench.add(`mulScalar ${label}`, () => {
        run(() => {
          const a = torch.ones([m, n] as const)
          return a.mulScalar(2.5)
        })
      })

      bench.add(`divScalar ${label}`, () => {
        run(() => {
          const a = torch.ones([m, n] as const)
          return a.divScalar(2)
        })
      })
    }

    // Chained operations
    bench.add('chain: a.add(b).mul(c) [256, 256]', () => {
      run(() => {
        const a = torch.ones([256, 256] as const)
        const b = torch.ones([256, 256] as const)
        const c = torch.ones([256, 256] as const)
        return a.add(b).mul(c)
      })
    })

    bench.add('chain: 5 ops [256, 256]', () => {
      run(() => {
        const a = torch.ones([256, 256] as const)
        const b = torch.ones([256, 256] as const)
        return a.add(b).mulScalar(2).sub(a).addScalar(1).divScalar(2)
      })
    })

    await bench.run()
    return bench
  },
}

export default suite
