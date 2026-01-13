/**
 * Tensor Creation Benchmarks
 *
 * Tests factory function performance at various sizes.
 */

import { Bench } from 'tinybench'
import { device, run } from '@ts-torch/core'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'
import { STANDARD_SIZES } from '../lib/utils.js'

const cpu = device.cpu()

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
          cpu.zeros([m, n] as const)
        })
      })

      bench.add(`ones ${label}`, () => {
        run(() => {
          cpu.ones([m, n] as const)
        })
      })

      bench.add(`empty ${label}`, () => {
        run(() => {
          cpu.empty([m, n] as const)
        })
      })

      bench.add(`randn ${label}`, () => {
        run(() => {
          cpu.randn([m, n] as const)
        })
      })
    }

    // fromArray benchmark
    const testData = new Float32Array(1024).fill(1.0)
    bench.add('tensor() from Float32Array [32, 32]', () => {
      run(() => {
        cpu.tensor(testData, [32, 32] as const)
      })
    })

    const largeData = new Float32Array(65536).fill(1.0)
    bench.add('tensor() from Float32Array [256, 256]', () => {
      run(() => {
        cpu.tensor(largeData, [256, 256] as const)
      })
    })

    await bench.run()
    return bench
  },
}

export default suite
