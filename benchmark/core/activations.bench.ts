/**
 * Activation Function Benchmarks
 *
 * Tests relu, sigmoid, tanh, softmax, etc.
 */

import { Bench } from 'tinybench'
import { torch, run } from '@ts-torch/core'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'
import { STANDARD_SIZES } from '../lib/utils.js'

export const suite: BenchmarkSuite = {
  name: 'Activation Functions',
  category: 'core',

  async run(config: BenchmarkConfig) {
    const bench = new Bench({
      time: config.time ?? 1000,
      warmup: config.warmup ?? true,
    })

    const sizes = config.sizes ?? STANDARD_SIZES

    for (const [m, n] of sizes) {
      const label = `[${m}, ${n}]`

      bench.add(`relu ${label}`, () => {
        run(() => {
          const a = torch.randn([m, n] as const)
          return a.relu()
        })
      })

      bench.add(`sigmoid ${label}`, () => {
        run(() => {
          const a = torch.randn([m, n] as const)
          return a.sigmoid()
        })
      })
    }

    // Additional activations at medium size
    const mediumSize: [number, number] = [256, 256]
    const label = '[256, 256]'

    bench.add(`tanh ${label}`, () => {
      run(() => {
        const a = torch.randn(mediumSize as const)
        return a.tanh()
      })
    })

    bench.add(`exp ${label}`, () => {
      run(() => {
        const a = torch.randn(mediumSize as const)
        return a.exp()
      })
    })

    bench.add(`log ${label}`, () => {
      run(() => {
        // Use abs to avoid log of negative numbers
        const a = torch.randn(mediumSize as const)
        return a.mul(a).addScalar(0.1).log() // log(x^2 + 0.1)
      })
    })

    bench.add(`sqrt ${label}`, () => {
      run(() => {
        const a = torch.randn(mediumSize as const)
        return a.mul(a).sqrt() // sqrt(x^2) = |x|
      })
    })

    bench.add(`neg ${label}`, () => {
      run(() => {
        const a = torch.randn(mediumSize as const)
        return a.neg()
      })
    })

    // Softmax benchmarks (includes reduction)
    const softmaxSizes: Array<[number, number]> = [
      [32, 10], // Small batch, 10 classes
      [128, 100], // Medium batch, 100 classes
      [256, 1000], // Large batch, 1000 classes (ImageNet)
    ]

    for (const [batch, classes] of softmaxSizes) {
      bench.add(`softmax [${batch}, ${classes}] dim=1`, () => {
        run(() => {
          const a = torch.randn([batch, classes] as const)
          return a.softmax(1)
        })
      })

      bench.add(`logSoftmax [${batch}, ${classes}] dim=1`, () => {
        run(() => {
          const a = torch.randn([batch, classes] as const)
          return a.logSoftmax(1)
        })
      })
    }

    await bench.run()
    return bench
  },
}

export default suite
