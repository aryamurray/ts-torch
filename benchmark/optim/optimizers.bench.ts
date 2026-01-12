/**
 * Optimizer Benchmarks
 *
 * Tests SGD, Adam, RMSprop step performance.
 */

import { Bench } from 'tinybench'
import { torch, run } from '@ts-torch/core'
import { Linear } from '@ts-torch/nn'
import { SGD, Adam, RMSprop } from '@ts-torch/optim'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'

export const suite: BenchmarkSuite = {
  name: 'Optimizers',
  category: 'optim',

  async run(config: BenchmarkConfig) {
    const bench = new Bench({
      time: config.time ?? 1000,
      warmup: config.warmup ?? true,
    })

    // Test configurations: different parameter sizes
    const paramConfigs: Array<{ inFeatures: number; outFeatures: number; label: string }> = [
      { inFeatures: 128, outFeatures: 64, label: 'small (8K params)' },
      { inFeatures: 512, outFeatures: 256, label: 'medium (131K params)' },
      { inFeatures: 1024, outFeatures: 512, label: 'large (524K params)' },
    ]

    for (const { inFeatures, outFeatures, label } of paramConfigs) {
      // Create model for each optimizer test
      const createModel = () => new Linear(inFeatures, outFeatures)

      // SGD
      bench.add(`SGD step ${label}`, () => {
        run(() => {
          const model = createModel()
          const optimizer = new SGD(model.parameters(), { lr: 0.01 })

          // Simulate gradient computation by setting requires_grad
          const x = torch.randn([32, inFeatures] as const)
          const y = model.forward(x)
          const loss = y.sum()
          loss.backward()

          optimizer.step()
          return loss
        })
      })

      // SGD with momentum
      bench.add(`SGD+momentum step ${label}`, () => {
        run(() => {
          const model = createModel()
          const optimizer = new SGD(model.parameters(), { lr: 0.01, momentum: 0.9 })

          const x = torch.randn([32, inFeatures] as const)
          const y = model.forward(x)
          const loss = y.sum()
          loss.backward()

          optimizer.step()
          return loss
        })
      })

      // Adam
      bench.add(`Adam step ${label}`, () => {
        run(() => {
          const model = createModel()
          const optimizer = new Adam(model.parameters(), { lr: 0.001 })

          const x = torch.randn([32, inFeatures] as const)
          const y = model.forward(x)
          const loss = y.sum()
          loss.backward()

          optimizer.step()
          return loss
        })
      })

      // RMSprop
      bench.add(`RMSprop step ${label}`, () => {
        run(() => {
          const model = createModel()
          const optimizer = new RMSprop(model.parameters(), { lr: 0.01 })

          const x = torch.randn([32, inFeatures] as const)
          const y = model.forward(x)
          const loss = y.sum()
          loss.backward()

          optimizer.step()
          return loss
        })
      })
    }

    // Multiple steps comparison (simulates training loop)
    bench.add('SGD 10 steps Linear(256->128)', () => {
      run(() => {
        const model = new Linear(256, 128)
        const optimizer = new SGD(model.parameters(), { lr: 0.01 })

        for (let i = 0; i < 10; i++) {
          optimizer.zeroGrad()
          const x = torch.randn([32, 256] as const)
          const y = model.forward(x)
          const loss = y.sum()
          loss.backward()
          optimizer.step()
        }

        return model
      })
    })

    bench.add('Adam 10 steps Linear(256->128)', () => {
      run(() => {
        const model = new Linear(256, 128)
        const optimizer = new Adam(model.parameters(), { lr: 0.001 })

        for (let i = 0; i < 10; i++) {
          optimizer.zeroGrad()
          const x = torch.randn([32, 256] as const)
          const y = model.forward(x)
          const loss = y.sum()
          loss.backward()
          optimizer.step()
        }

        return model
      })
    })

    await bench.run()
    return bench
  },
}

export default suite
