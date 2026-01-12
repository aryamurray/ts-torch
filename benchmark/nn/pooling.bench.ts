/**
 * Pooling Layer Benchmarks
 *
 * Tests MaxPool2d, AvgPool2d, AdaptiveAvgPool2d.
 */

import { Bench } from 'tinybench'
import { torch, run } from '@ts-torch/core'
import { MaxPool2d, AvgPool2d, AdaptiveAvgPool2d } from '@ts-torch/nn'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'

export const suite: BenchmarkSuite = {
  name: 'Pooling Layers',
  category: 'nn',

  async run(config: BenchmarkConfig) {
    const bench = new Bench({
      time: config.time ?? 1000,
      warmup: config.warmup ?? true,
    })

    // Input sizes to test
    const inputConfigs: Array<{ batch: number; channels: number; size: number }> = [
      { batch: 32, channels: 64, size: 32 },
      { batch: 32, channels: 128, size: 16 },
      { batch: 32, channels: 256, size: 8 },
      { batch: 64, channels: 64, size: 56 }, // ImageNet-like
    ]

    // MaxPool2d
    for (const { batch, channels, size } of inputConfigs) {
      bench.add(`MaxPool2d k=2 [${batch}, ${channels}, ${size}, ${size}]`, () => {
        run(() => {
          const pool = new MaxPool2d([2, 2])
          const x = torch.randn([batch, channels, size, size] as const)
          return pool.forward(x)
        })
      })
    }

    // AvgPool2d
    for (const { batch, channels, size } of inputConfigs) {
      bench.add(`AvgPool2d k=2 [${batch}, ${channels}, ${size}, ${size}]`, () => {
        run(() => {
          const pool = new AvgPool2d([2, 2])
          const x = torch.randn([batch, channels, size, size] as const)
          return pool.forward(x)
        })
      })
    }

    // AdaptiveAvgPool2d
    const adaptiveConfigs: Array<{ batch: number; channels: number; inputSize: number; outputSize: number }> = [
      { batch: 32, channels: 512, inputSize: 7, outputSize: 1 }, // Common in ResNet
      { batch: 32, channels: 256, inputSize: 14, outputSize: 1 },
      { batch: 32, channels: 128, inputSize: 28, outputSize: 7 },
    ]

    for (const { batch, channels, inputSize, outputSize } of adaptiveConfigs) {
      bench.add(`AdaptiveAvgPool2d ${inputSize}x${inputSize}->${outputSize}x${outputSize} [${batch}, ${channels}]`, () => {
        run(() => {
          const pool = new AdaptiveAvgPool2d([outputSize, outputSize])
          const x = torch.randn([batch, channels, inputSize, inputSize] as const)
          return pool.forward(x)
        })
      })
    }

    // Pre-created layers (forward only)
    const maxPool = new MaxPool2d([2, 2], { stride: [2, 2] })
    bench.add('MaxPool2d forward only k=2 s=2, batch=64', () => {
      run(() => {
        const x = torch.randn([64, 64, 32, 32] as const)
        return maxPool.forward(x)
      })
    })

    const avgPool = new AvgPool2d([2, 2], { stride: [2, 2] })
    bench.add('AvgPool2d forward only k=2 s=2, batch=64', () => {
      run(() => {
        const x = torch.randn([64, 64, 32, 32] as const)
        return avgPool.forward(x)
      })
    })

    await bench.run()
    return bench
  },
}

export default suite
