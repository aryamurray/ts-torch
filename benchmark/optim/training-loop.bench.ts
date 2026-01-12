/**
 * Training Loop Benchmarks
 *
 * Tests full forward + backward + optimizer step performance.
 */

import { Bench } from 'tinybench'
import { torch, run } from '@ts-torch/core'
import { Linear, Conv2d, MaxPool2d } from '@ts-torch/nn'
import { SGD, Adam } from '@ts-torch/optim'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'

export const suite: BenchmarkSuite = {
  name: 'Training Loop',
  category: 'optim',

  async run(config: BenchmarkConfig) {
    const bench = new Bench({
      time: config.time ?? 1000,
      warmup: config.warmup ?? true,
    })

    // Simple MLP training step
    bench.add('MLP 784->256->10 train step, batch=32', () => {
      run(() => {
        const l1 = new Linear(784, 256)
        const l2 = new Linear(256, 10)
        const optimizer = new SGD([...l1.parameters(), ...l2.parameters()], { lr: 0.01 })

        optimizer.zeroGrad()

        // Forward pass
        const x = torch.randn([32, 784] as const)
        let h = l1.forward(x).relu()
        const logits = l2.forward(h)

        // Compute loss (using sum as simple loss)
        const target = torch.zeros([32, 10] as const)
        const loss = logits.sub(target).mul(logits.sub(target)).mean()

        // Backward pass
        loss.backward()

        // Optimizer step
        optimizer.step()

        return loss
      })
    })

    // MLP with Adam
    bench.add('MLP 784->256->10 train step (Adam), batch=32', () => {
      run(() => {
        const l1 = new Linear(784, 256)
        const l2 = new Linear(256, 10)
        const optimizer = new Adam([...l1.parameters(), ...l2.parameters()], { lr: 0.001 })

        optimizer.zeroGrad()

        const x = torch.randn([32, 784] as const)
        let h = l1.forward(x).relu()
        const logits = l2.forward(h)

        const target = torch.zeros([32, 10] as const)
        const loss = logits.sub(target).mul(logits.sub(target)).mean()

        loss.backward()
        optimizer.step()

        return loss
      })
    })

    // Deeper MLP
    bench.add('MLP 784->512->256->128->10 train step, batch=64', () => {
      run(() => {
        const l1 = new Linear(784, 512)
        const l2 = new Linear(512, 256)
        const l3 = new Linear(256, 128)
        const l4 = new Linear(128, 10)
        const params = [...l1.parameters(), ...l2.parameters(), ...l3.parameters(), ...l4.parameters()]
        const optimizer = new SGD(params, { lr: 0.01, momentum: 0.9 })

        optimizer.zeroGrad()

        const x = torch.randn([64, 784] as const)
        let h = l1.forward(x).relu()
        h = l2.forward(h).relu()
        h = l3.forward(h).relu()
        const logits = l4.forward(h)

        const loss = logits.sum()
        loss.backward()
        optimizer.step()

        return loss
      })
    })

    // CNN training step (simplified)
    bench.add('CNN conv->pool->linear train step, batch=16', () => {
      run(() => {
        const conv = new Conv2d(1, 32, [3, 3])
        const pool = new MaxPool2d({ kernelSize: [2, 2] })
        const fc = new Linear(32 * 13 * 13, 10)

        const params = [...conv.parameters(), ...fc.parameters()]
        const optimizer = new SGD(params, { lr: 0.01 })

        optimizer.zeroGrad()

        // Input: [batch, channels, height, width]
        const x = torch.randn([16, 1, 28, 28] as const)

        // Forward: conv -> pool -> flatten -> fc
        let h = conv.forward(x).relu()
        h = pool.forward(h)
        const flat = h.flatten(1) // Flatten spatial dims
        const logits = fc.forward(flat)

        const loss = logits.sum()
        loss.backward()
        optimizer.step()

        return loss
      })
    })

    // Multiple training steps
    bench.add('5 training iterations MLP 256->128, batch=32', () => {
      run(() => {
        const model = new Linear(256, 128)
        const optimizer = new Adam(model.parameters(), { lr: 0.001 })

        for (let i = 0; i < 5; i++) {
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

    // Inference vs training comparison
    bench.add('Linear(256->128) inference only, batch=32', () => {
      run(() => {
        const model = new Linear(256, 128)
        const x = torch.randn([32, 256] as const)
        return model.forward(x)
      })
    })

    bench.add('Linear(256->128) full train step, batch=32', () => {
      run(() => {
        const model = new Linear(256, 128)
        const optimizer = new SGD(model.parameters(), { lr: 0.01 })

        optimizer.zeroGrad()

        const x = torch.randn([32, 256] as const)
        const y = model.forward(x)
        const loss = y.sum()

        loss.backward()
        optimizer.step()

        return loss
      })
    })

    await bench.run()
    return bench
  },
}

export default suite
