/**
 * Normalization Layer Benchmarks
 *
 * Tests BatchNorm1d, BatchNorm2d, LayerNorm.
 */

import { Bench } from 'tinybench'
import { torch, run } from '@ts-torch/core'
import { BatchNorm1d, BatchNorm2d, LayerNorm } from '@ts-torch/nn'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'

export const suite: BenchmarkSuite = {
  name: 'Normalization Layers',
  category: 'nn',

  async run(config: BenchmarkConfig) {
    const bench = new Bench({
      time: config.time ?? 1000,
      warmup: config.warmup ?? true,
    })

    // BatchNorm1d (for fully connected layers)
    const bn1dConfigs: Array<{ batch: number; features: number }> = [
      { batch: 32, features: 128 },
      { batch: 64, features: 256 },
      { batch: 128, features: 512 },
      { batch: 256, features: 1024 },
    ]

    for (const { batch, features } of bn1dConfigs) {
      bench.add(`BatchNorm1d(${features}) batch=${batch}`, () => {
        run(() => {
          const bn = new BatchNorm1d(features)
          const x = torch.randn([batch, features] as const)
          return bn.forward(x)
        })
      })
    }

    // BatchNorm2d (for conv layers)
    const bn2dConfigs: Array<{ batch: number; channels: number; size: number }> = [
      { batch: 32, channels: 64, size: 32 },
      { batch: 32, channels: 128, size: 16 },
      { batch: 32, channels: 256, size: 8 },
      { batch: 64, channels: 64, size: 56 },
    ]

    for (const { batch, channels, size } of bn2dConfigs) {
      bench.add(`BatchNorm2d(${channels}) [${batch}, ${channels}, ${size}, ${size}]`, () => {
        run(() => {
          const bn = new BatchNorm2d(channels)
          const x = torch.randn([batch, channels, size, size] as const)
          return bn.forward(x)
        })
      })
    }

    // LayerNorm
    const lnConfigs: Array<{ batch: number; seq: number; hidden: number }> = [
      { batch: 32, seq: 128, hidden: 256 }, // Small transformer
      { batch: 32, seq: 512, hidden: 512 }, // Medium transformer
      { batch: 16, seq: 512, hidden: 768 }, // BERT-base like
    ]

    for (const { batch, seq, hidden } of lnConfigs) {
      bench.add(`LayerNorm([${hidden}]) [${batch}, ${seq}, ${hidden}]`, () => {
        run(() => {
          const ln = new LayerNorm([hidden])
          const x = torch.randn([batch, seq, hidden] as const)
          return ln.forward(x)
        })
      })
    }

    // Pre-created layers (forward only)
    const bn1d = new BatchNorm1d(256)
    bench.add('BatchNorm1d(256) forward only batch=128', () => {
      run(() => {
        const x = torch.randn([128, 256] as const)
        return bn1d.forward(x)
      })
    })

    const bn2d = new BatchNorm2d(128)
    bench.add('BatchNorm2d(128) forward only [64, 128, 16, 16]', () => {
      run(() => {
        const x = torch.randn([64, 128, 16, 16] as const)
        return bn2d.forward(x)
      })
    })

    const ln = new LayerNorm([512])
    bench.add('LayerNorm([512]) forward only [32, 256, 512]', () => {
      run(() => {
        const x = torch.randn([32, 256, 512] as const)
        return ln.forward(x)
      })
    })

    await bench.run()
    return bench
  },
}

export default suite
