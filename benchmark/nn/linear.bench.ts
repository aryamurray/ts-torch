/**
 * Linear Layer Benchmarks
 *
 * Tests Linear layer forward pass at various sizes.
 */

import { Bench } from 'tinybench'
import { device, run } from '@ts-torch/core'
import { Linear } from '@ts-torch/nn'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'

const cpu = device.cpu()

export const suite: BenchmarkSuite = {
  name: 'Linear Layer',
  category: 'nn',

  async run(config: BenchmarkConfig) {
    const bench = new Bench({
      time: config.time ?? 1000,
      warmup: config.warmup ?? true,
    })

    // Common layer configurations
    const configs: Array<{ batch: number; inFeatures: number; outFeatures: number }> = [
      { batch: 1, inFeatures: 784, outFeatures: 128 }, // Single sample
      { batch: 32, inFeatures: 784, outFeatures: 128 }, // Small batch
      { batch: 128, inFeatures: 784, outFeatures: 128 }, // Medium batch
      { batch: 256, inFeatures: 784, outFeatures: 128 }, // Large batch
      { batch: 32, inFeatures: 128, outFeatures: 10 }, // Output layer
      { batch: 32, inFeatures: 256, outFeatures: 256 }, // Hidden layer
      { batch: 32, inFeatures: 512, outFeatures: 512 }, // Large hidden
      { batch: 32, inFeatures: 1024, outFeatures: 1024 }, // Very large
    ]

    for (const { batch, inFeatures, outFeatures } of configs) {
      const label = `Linear(${inFeatures}->${outFeatures}) batch=${batch}`

      bench.add(label, () => {
        run(() => {
          const layer = new Linear(inFeatures, outFeatures)
          const x = cpu.randn([batch, inFeatures] as const)
          return layer.forward(x)
        })
      })
    }

    // Pre-created layer (measure forward pass only, no layer creation)
    const preCreatedLayer = new Linear(256, 128)
    bench.add('Linear(256->128) forward only, batch=64', () => {
      run(() => {
        const x = cpu.randn([64, 256] as const)
        return preCreatedLayer.forward(x)
      })
    })

    // With bias disabled
    const noBiasLayer = new Linear(256, 128, { bias: false })
    bench.add('Linear(256->128) no bias, batch=64', () => {
      run(() => {
        const x = cpu.randn([64, 256] as const)
        return noBiasLayer.forward(x)
      })
    })

    // MLP pattern (multiple layers)
    bench.add('MLP 784->256->128->10, batch=32', () => {
      run(() => {
        const l1 = new Linear(784, 256)
        const l2 = new Linear(256, 128)
        const l3 = new Linear(128, 10)

        const x = cpu.randn([32, 784] as const)
        let h = l1.forward(x).relu()
        h = l2.forward(h).relu()
        return l3.forward(h)
      })
    })

    await bench.run()
    return bench
  },
}

export default suite
