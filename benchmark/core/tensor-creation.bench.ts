/**
 * Tensor Creation Benchmarks
 *
 * Tests factory function performance at various sizes.
 */

import { Bench } from 'tinybench'
import { torch, run } from '@ts-torch/core'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'
import { STANDARD_SIZES } from '../lib/utils.js'

export const suite: BenchmarkSuite = {
  name: 'Tensor Creation',
  category: 'core',

  async run(config: BenchmarkConfig) {
    const bench = new Bench({
      time: config.time ?? 1000,
      warmup: config.warmup ?? true,
    })

    const sizes = config.sizes ?? STANDARD_SIZES

    for (const [m, n] of sizes) {
      const label = `[${m}, ${n}]`

      bench.add(`zeros ${label}`, () => {
        run(() => {
          torch.zeros([m, n] as const)
        })
      })

      bench.add(`ones ${label}`, () => {
        run(() => {
          torch.ones([m, n] as const)
        })
      })

      bench.add(`empty ${label}`, () => {
        run(() => {
          torch.empty([m, n] as const)
        })
      })

      bench.add(`randn ${label}`, () => {
        run(() => {
          torch.randn([m, n] as const)
        })
      })
    }

    // fromArray benchmark
    const testData = new Float32Array(1024).fill(1.0)
    bench.add('tensor() from Float32Array [32, 32]', () => {
      run(() => {
        torch.tensor(testData, [32, 32] as const)
      })
    })

    const largeData = new Float32Array(65536).fill(1.0)
    bench.add('tensor() from Float32Array [256, 256]', () => {
      run(() => {
        torch.tensor(largeData, [256, 256] as const)
      })
    })

    // arange benchmark
    bench.add('arange(0, 100)', () => {
      run(() => {
        torch.arange(0, 100)
      })
    })

    bench.add('arange(0, 1000)', () => {
      run(() => {
        torch.arange(0, 1000)
      })
    })

    await bench.run()
    return bench
  },
}

export default suite
