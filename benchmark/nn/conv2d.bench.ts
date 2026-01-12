/**
 * Conv2d Layer Benchmarks
 *
 * Tests Conv2d forward pass at various configurations.
 */

import { Bench } from 'tinybench'
import { torch, run } from '@ts-torch/core'
import { Conv2d } from '@ts-torch/nn'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'

export const suite: BenchmarkSuite = {
  name: 'Conv2d Layer',
  category: 'nn',

  async run(config: BenchmarkConfig) {
    const bench = new Bench({
      time: config.time ?? 1000,
      warmup: config.warmup ?? true,
    })

    // Common configurations
    const configs: Array<{
      batch: number
      inChannels: number
      outChannels: number
      inputSize: number
      kernelSize: number
      label: string
    }> = [
      // Small input
      { batch: 1, inChannels: 1, outChannels: 32, inputSize: 28, kernelSize: 3, label: 'MNIST first conv' },
      { batch: 32, inChannels: 1, outChannels: 32, inputSize: 28, kernelSize: 3, label: 'MNIST batch=32' },

      // Medium input (CIFAR-like)
      { batch: 32, inChannels: 3, outChannels: 64, inputSize: 32, kernelSize: 3, label: 'CIFAR batch=32' },
      { batch: 64, inChannels: 3, outChannels: 64, inputSize: 32, kernelSize: 3, label: 'CIFAR batch=64' },

      // Deeper layers
      { batch: 32, inChannels: 64, outChannels: 128, inputSize: 16, kernelSize: 3, label: 'Deep 64->128' },
      { batch: 32, inChannels: 128, outChannels: 256, inputSize: 8, kernelSize: 3, label: 'Deep 128->256' },

      // Large input (ImageNet-like)
      { batch: 8, inChannels: 3, outChannels: 64, inputSize: 224, kernelSize: 7, label: 'ImageNet first' },
      { batch: 16, inChannels: 64, outChannels: 64, inputSize: 56, kernelSize: 3, label: 'ImageNet mid' },
    ]

    for (const { batch, inChannels, outChannels, inputSize, kernelSize, label } of configs) {
      bench.add(`Conv2d ${label}`, () => {
        run(() => {
          const conv = new Conv2d(inChannels, outChannels, [kernelSize, kernelSize])
          const x = torch.randn([batch, inChannels, inputSize, inputSize] as const)
          return conv.forward(x)
        })
      })
    }

    // With stride and padding
    bench.add('Conv2d 3->64 k=3 s=2 p=1, 32x32 batch=32', () => {
      run(() => {
        const conv = new Conv2d(3, 64, [3, 3], {
          stride: [2, 2],
          padding: [1, 1],
        })
        const x = torch.randn([32, 3, 32, 32] as const)
        return conv.forward(x)
      })
    })

    // Pre-created layer (forward only)
    const preCreatedConv = new Conv2d(64, 128, [3, 3], { padding: [1, 1] })
    bench.add('Conv2d forward only 64->128 k=3 p=1, 16x16 batch=32', () => {
      run(() => {
        const x = torch.randn([32, 64, 16, 16] as const)
        return preCreatedConv.forward(x)
      })
    })

    await bench.run()
    return bench
  },
}

export default suite
